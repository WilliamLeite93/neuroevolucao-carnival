from __future__ import annotations

import argparse
import copy
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import ale_py
import gymnasium as gym
import numpy as np


ENV_ID = "ALE/Carnival-v5"
NUM_ENTRADAS = 128
NUM_OCULTOS = 16
NUM_ACOES = 6
TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
NUM_ELITE = 2
LIMITE_PASSOS_EPISODIO = 500
PASTA_RESULTADOS = Path("resultados")


@dataclass(frozen=True)
class ConfiguracaoExperimento:
    nome: str
    taxa_mutacao: float
    intensidade_mutacao: float


EXPERIMENTOS = {
    "A": ConfiguracaoExperimento("A", 0.01, 0.05),
    "B": ConfiguracaoExperimento("B", 0.10, 0.20),
    "C": ConfiguracaoExperimento("C", 0.40, 0.50),
}


def criar_ambiente(render_mode: str | None = None):
    return gym.make(
        ENV_ID,
        obs_type="ram",
        render_mode=render_mode,
        max_episode_steps=LIMITE_PASSOS_EPISODIO,
    )


class Agente:

    def __init__(self) -> None:
        self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS)
        self.b1 = np.random.randn(NUM_OCULTOS)
        self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES)
        self.b2 = np.random.randn(NUM_ACOES)
        self.fitness = 0.0

    def agir(self, estado: np.ndarray) -> int:
        estado_normalizado = np.asarray(estado, dtype=np.float32) / 255.0
        oculto = np.tanh(estado_normalizado @ self.W1 + self.b1)
        saida = oculto @ self.W2 + self.b2
        return int(np.argmax(saida))


def avaliar(agente: Agente, seed: int | None = None) -> float:
    """Executa um episódio, soma rewards e retorna essa soma como fitness."""
    env = criar_ambiente()
    try:
        estado, _info = env.reset(seed=seed)
        if np.asarray(estado).shape != (NUM_ENTRADAS,):
            raise ValueError(
                f"Esperava observação RAM com 128 valores; recebido: {np.asarray(estado).shape}"
            )
        recompensa_total = 0.0
        terminou = False
        while not terminou:
            acao = agente.agir(estado)
            estado, recompensa, terminated, truncated, _info = env.step(acao)
            recompensa_total += float(recompensa)
            terminou = terminated or truncated
        agente.fitness = recompensa_total
        return recompensa_total
    finally:
        env.close()


def avaliar_aleatorio(numero_episodios: int, seed: int) -> tuple[float, list[float]]:
    """Baseline: amostra ações uniformemente do espaço de ações do Carnival."""
    resultados: list[float] = []
    for episodio in range(numero_episodios):
        env = criar_ambiente()
        try:
            episodio_seed = seed + episodio
            _estado, _info = env.reset(seed=episodio_seed)
            env.action_space.seed(episodio_seed)
            total = 0.0
            terminou = False
            while not terminou:
                _estado, recompensa, terminated, truncated, _info = env.step(
                    env.action_space.sample()
                )
                total += float(recompensa)
                terminou = terminated or truncated
            resultados.append(total)
        finally:
            env.close()
    return float(np.mean(resultados)), resultados


def salvar_agente(agente: Agente, caminho: Path, experimento: str) -> None:
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        caminho,
        W1=agente.W1,
        b1=agente.b1,
        W2=agente.W2,
        b2=agente.b2,
        fitness_treino=agente.fitness,
        experimento=experimento,
    )


def carregar_agente(caminho: Path) -> Agente:
    dados = np.load(caminho)
    agente = Agente()
    agente.W1 = dados["W1"]
    agente.b1 = dados["b1"]
    agente.W2 = dados["W2"]
    agente.b2 = dados["b2"]
    agente.fitness = float(dados["fitness_treino"])
    return agente


def avaliar_em_seeds(agente: Agente, numero_episodios: int, seed: int) -> list[float]:
    fitness_treino = agente.fitness
    try:
        return [avaliar(agente, seed=seed + episodio) for episodio in range(numero_episodios)]
    finally:
        agente.fitness = fitness_treino


def criar_populacao() -> list[Agente]:
    return [Agente() for _ in range(TAMANHO_POPULACAO)]


def selecionar(populacao: list[Agente]) -> list[Agente]:
    quantidade_pais = max(2, TAMANHO_POPULACAO // 4)
    return sorted(populacao, key=lambda agente: agente.fitness, reverse=True)[:quantidade_pais]


def mutar(agente: Agente, configuracao: ConfiguracaoExperimento) -> None:
    for nome in ("W1", "b1", "W2", "b2"):
        parametros = getattr(agente, nome)
        mascara = np.random.random(parametros.shape) < configuracao.taxa_mutacao
        ruido = np.random.normal(0.0, configuracao.intensidade_mutacao, parametros.shape)
        parametros += mascara * ruido


def nova_geracao(
    populacao: list[Agente], configuracao: ConfiguracaoExperimento
) -> list[Agente]:
    pais = selecionar(populacao)
    ordenados = sorted(populacao, key=lambda agente: agente.fitness, reverse=True)
    proxima = [copy.deepcopy(agente) for agente in ordenados[:NUM_ELITE]]
    while len(proxima) < TAMANHO_POPULACAO:
        filho = copy.deepcopy(random.choice(pais))
        mutar(filho, configuracao)
        proxima.append(filho)
    return proxima


def executar_experimento(
    configuracao: ConfiguracaoExperimento, seed: int
) -> tuple[list[dict[str, float | int | str]], Agente]:
    random.seed(seed)
    np.random.seed(seed)
    populacao = criar_populacao()
    historico: list[dict[str, float | int | str]] = []
    melhor_global: Agente | None = None

    for indice_geracao in range(NUM_GERACOES):
        for indice, agente in enumerate(populacao):
            agente.fitness = avaliar(
                agente, seed=seed + indice_geracao * TAMANHO_POPULACAO + indice
            )

        valores = [agente.fitness for agente in populacao]
        melhor_da_geracao = max(populacao, key=lambda agente: agente.fitness)
        if melhor_global is None or melhor_da_geracao.fitness > melhor_global.fitness:
            melhor_global = copy.deepcopy(melhor_da_geracao)
        registro: dict[str, float | int | str] = {
            "experimento": configuracao.nome,
            "geracao": indice_geracao + 1,
            "melhor": max(valores),
            "media": float(np.mean(valores)),
            "pior": min(valores),
        }
        historico.append(registro)
        print(
            f"Experimento {configuracao.nome} | Geração {indice_geracao + 1:02d}/{NUM_GERACOES} | "
            f"Melhor: {max(valores):.2f} | Média: {np.mean(valores):.2f} | Pior: {min(valores):.2f}"
        )
        populacao = nova_geracao(populacao, configuracao)

    assert melhor_global is not None
    return historico, melhor_global


def salvar_csv(historicos: list[dict[str, float | int | str]]) -> Path:
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_RESULTADOS / "fitness.csv"
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        campos = ["experimento", "geracao", "melhor", "media", "pior"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(historicos)
    return destino


def salvar_comparacao_csv(
    baseline: list[float], resultados: dict[str, list[float]], seed: int, experimento: str
) -> Path:
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_RESULTADOS / "comparacao_baseline.csv"
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        campos = ["episodio", "seed", "fitness_aleatorio", "fitness_evoluido", "experimento"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for indice, valor_aleatorio in enumerate(baseline):
            escritor.writerow({
                "episodio": indice + 1,
                "seed": seed + indice,
                "fitness_aleatorio": valor_aleatorio,
                "fitness_evoluido": resultados[experimento][indice],
                "experimento": experimento,
            })
    return destino


def salvar_grafico(historicos: list[dict[str, float | int | str]]) -> Path:
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = PASTA_RESULTADOS / "fitness.png"
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        _salvar_grafico_png_basico(historicos, destino)
        return destino

    fig, eixos = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for eixo, nome in zip(eixos, ("melhor", "media")):
        for codigo in EXPERIMENTOS:
            linhas = [linha for linha in historicos if linha["experimento"] == codigo]
            eixo.plot(
                [int(linha["geracao"]) for linha in linhas],
                [float(linha[nome]) for linha in linhas],
                label=f"Experimento {codigo}",
            )
        eixo.set_title("Melhor fitness" if nome == "melhor" else "Fitness médio")
        eixo.set_xlabel("Geração")
        eixo.set_ylabel("Soma de recompensas (fitness)")
        eixo.grid(True, alpha=0.3)
        eixo.legend()
    fig.suptitle("Neuroevolução no Atari Carnival")
    fig.tight_layout()
    fig.savefig(str(destino), dpi=150)
    plt.close(fig)
    return destino


def _salvar_grafico_png_basico(
    historicos: list[dict[str, float | int | str]], destino: Path
) -> None:
    import struct
    import zlib

    largura, altura = 1100, 540
    pixels = bytearray([255] * (largura * altura * 3))
    fonte = {
        "A": (14,17,31,31,17,17,17), "B": (30,17,17,30,17,17,30), "C": (14,17,16,16,16,17,14),
        "D": (30,17,17,17,17,17,30), "E": (31,16,16,30,16,16,31), "F": (31,16,16,30,16,16,16),
        "G": (14,17,16,23,17,17,15), "H": (17,17,17,31,17,17,17), "I": (14,4,4,4,4,4,14),
        "L": (16,16,16,16,16,16,31), "M": (17,27,21,21,17,17,17), "N": (17,25,25,21,19,19,17),
        "O": (14,17,17,17,17,17,14), "P": (30,17,17,30,16,16,16), "R": (30,17,17,30,20,18,17),
        "S": (15,16,16,14,1,1,30), "T": (31,4,4,4,4,4,4), "X": (17,17,10,4,10,17,17),
        "0": (14,17,19,21,25,17,14), "1": (4,12,4,4,4,4,14), "2": (14,17,1,2,4,8,31),
        "3": (30,1,1,14,1,1,30), "4": (2,6,10,18,31,2,2), "5": (31,16,16,30,1,1,30),
        "6": (14,16,16,30,17,17,14), "7": (31,1,2,4,8,8,8), "8": (14,17,17,14,17,17,14),
        "9": (14,17,17,15,1,1,14), "-": (0,0,0,31,0,0,0), ".": (0,0,0,0,0,12,12), " ": (0,0,0,0,0,0,0),
    }

    def ponto(x: int, y: int, cor: tuple[int, int, int]) -> None:
        if 0 <= x < largura and 0 <= y < altura:
            inicio = (y * largura + x) * 3
            pixels[inicio : inicio + 3] = bytes(cor)

    def linha(x0: int, y0: int, x1: int, y1: int, cor: tuple[int, int, int]) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        erro = dx + dy
        while True:
            ponto(x0, y0, cor)
            if x0 == x1 and y0 == y1:
                break
            dobro = 2 * erro
            if dobro >= dy:
                erro += dy
                x0 += sx
            if dobro <= dx:
                erro += dx
                y0 += sy

    def texto(x: int, y: int, valor: str, cor: tuple[int, int, int], escala: int = 1) -> None:
        cursor = x
        for caractere in valor.upper():
            for yi, bits in enumerate(fonte.get(caractere, fonte[" "])):
                for xi in range(5):
                    if bits & (1 << (4 - xi)):
                        for dy in range(escala):
                            for dx in range(escala):
                                ponto(cursor + xi * escala + dx, y + yi * escala + dy, cor)
            cursor += 6 * escala

    cores = {"A": (35,105,190), "B": (220,80,55), "C": (45,150,90)}
    for painel, campo in enumerate(("melhor", "media")):
        esquerda = 65 + painel * 520
        topo, direita, base = 75, esquerda + 430, 430
        texto(esquerda + 135, 30, "BEST FITNESS" if campo == "melhor" else "MEAN FITNESS", (20,20,20), 2)
        valores = [float(item[campo]) for item in historicos]
        minimo, maximo = min(valores, default=0.0), max(valores, default=1.0)
        if maximo == minimo:
            maximo += 1.0
        for divisao in range(6):
            y = topo + (base - topo) * divisao // 5
            for x in range(esquerda, direita + 1):
                ponto(x, y, (225,225,225))
            texto(esquerda - 37, y - 3, str(round(maximo - (maximo - minimo) * divisao / 5)), (60,60,60))
        linha(esquerda, topo, esquerda, base, (50,50,50))
        linha(esquerda, base, direita, base, (50,50,50))
        for codigo, cor in cores.items():
            serie = sorted((r for r in historicos if r["experimento"] == codigo), key=lambda r: int(r["geracao"]))
            pontos = []
            for item in serie:
                x = esquerda + (int(item["geracao"]) - 1) * (direita-esquerda) // max(1, NUM_GERACOES-1)
                y = base - round((float(item[campo])-minimo) * (base-topo) / (maximo-minimo))
                pontos.append((x,y))
            for inicio, fim in zip(pontos, pontos[1:]):
                linha(inicio[0], inicio[1], fim[0], fim[1], cor)
        texto(esquerda + 170, 445, "GENERATION", (20,20,20))

    x_legenda = 390
    for codigo, cor in cores.items():
        linha(x_legenda, 500, x_legenda + 24, 500, cor)
        texto(x_legenda + 30, 496, f"EXPERIMENT {codigo}", (20,20,20))
        x_legenda += 125

    def chunk(tipo: bytes, dados: bytes) -> bytes:
        conteudo = tipo + dados
        return struct.pack(">I", len(dados)) + conteudo + struct.pack(">I", zlib.crc32(conteudo) & 0xFFFFFFFF)

    dados_png = b"".join(b"\x00" + pixels[y*largura*3:(y+1)*largura*3] for y in range(altura))
    destino.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", largura, altura, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(dados_png))
        + chunk(b"IEND", b"")
    )


def executar_melhor(agente: Agente) -> float:
    env = criar_ambiente(render_mode="human")
    try:
        estado, _info = env.reset()
        total = 0.0
        terminou = False
        while not terminou:
            estado, recompensa, terminated, truncated, _info = env.step(agente.agir(estado))
            total += float(recompensa)
            terminou = terminated or truncated
        return total
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuroevolução no Carnival: evolução de redes neurais no ambiente ALE/Carnival-v5."
    )
    parser.add_argument(
        "--experimento",
        choices=["A", "B", "C", "todos"],
        default="todos",
        help="Configuração de mutação a executar (padrão: os três experimentos)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodução dos resultados")
    parser.add_argument(
        "--episodios-baseline",
        type=int,
        default=10,
        help="Número de episódios do agente aleatório para calcular o baseline",
    )
    parser.add_argument(
        "--render-melhor",
        action="store_true",
        help="Ao final do treino, abre o Carnival para demonstrar o melhor agente evoluído",
    )
    parser.add_argument(
        "--demonstrar-salvo",
        action="store_true",
        help="Carrega resultados/melhor_agente.npz e demonstra sem treinar novamente",
    )
    args = parser.parse_args()
    if args.episodios_baseline < 1:
        parser.error("--episodios-baseline deve ser pelo menos 1")

    gym.register_envs(ale_py)
    if args.demonstrar_salvo:
        caminho = PASTA_RESULTADOS / "melhor_agente.npz"
        if not caminho.exists():
            parser.error(f"Modelo salvo não encontrado: {caminho}. Execute o treinamento primeiro.")
        agente = carregar_agente(caminho)
        print(f"Demonstração do modelo salvo: {caminho}")
        print(f"Recompensa: {executar_melhor(agente):.2f}")
        return

    with criar_ambiente() as env:
        print(f"Ambiente: {ENV_ID} | ações: {env.action_space} | observação RAM: {env.observation_space.shape}")
        numero_acoes = cast(gym.spaces.Discrete, env.action_space).n
        if numero_acoes != NUM_ACOES:
            raise RuntimeError(
                f"Esperadas 6 ações do Carnival-v5; o ambiente apresentou {numero_acoes}."
            )

    seed_validacao = args.seed + 1_000_000
    baseline_media, baseline = avaliar_aleatorio(args.episodios_baseline, seed_validacao)
    print(f"Baseline aleatório | média: {baseline_media:.2f} | episódios: {baseline}")

    codigos = list(EXPERIMENTOS) if args.experimento == "todos" else [args.experimento]
    historicos: list[dict[str, float | int | str]] = []
    melhores: list[tuple[str, Agente]] = []
    for codigo in codigos:
        historico, melhor = executar_experimento(EXPERIMENTOS[codigo], args.seed)
        historicos.extend(historico)
        melhores.append((codigo, melhor))
        salvar_agente(melhor, PASTA_RESULTADOS / f"melhor_{codigo}.npz", codigo)

    csv_path = salvar_csv(historicos)
    grafico_path = salvar_grafico(historicos)
    validacoes: dict[str, list[float]] = {}
    for codigo, agente in melhores:
        validacoes[codigo] = avaliar_em_seeds(agente, args.episodios_baseline, seed_validacao)
        print(
            f"Agente {codigo} em validação | média: {np.mean(validacoes[codigo]):.2f} | "
            f"episódios: {validacoes[codigo]}"
        )

    melhor_codigo, melhor_agente = max(melhores, key=lambda item: item[1].fitness)
    melhor_media = float(np.mean(validacoes[melhor_codigo]))
    salvar_agente(melhor_agente, PASTA_RESULTADOS / "melhor_agente.npz", melhor_codigo)
    comparacao_path = salvar_comparacao_csv(baseline, validacoes, seed_validacao, melhor_codigo)
    print(
        f"Comparação pareada ({args.episodios_baseline} episódios, mesmas seeds) | "
        f"baseline: {baseline_media:.2f} | melhor evoluído ({melhor_codigo}): {melhor_media:.2f}"
    )
    print(f"Modelo vencedor salvo em {PASTA_RESULTADOS / 'melhor_agente.npz'}.")
    print(f"Histórico: {csv_path} | gráfico: {grafico_path} | comparação: {comparacao_path}.")
    if args.render_melhor:
        print(f"Recompensa na demonstração: {executar_melhor(melhor_agente):.2f}")


if __name__ == "__main__":
    main()
