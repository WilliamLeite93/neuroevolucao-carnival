# Neuroevolução no Carnival

O projeto é uma pequena rede neural evoluída por **neuroevolução**. Ele se resume a uma população de agentes que joga o **Carnival** do Arcade Learning Environment (ALE) e os melhores são selecionados e seus pesos são copiados e mutados ao longo de várias gerações. Todo o código está em `main.py` e é executável sem nenhuma edição manual.

## Ambiente utilizado e forma de observação

- **Ambiente:** `ALE/Carnival-v5` (Gymnasium + Arcade Learning Environment). Constantes em `main.py:18` e criação em `criar_ambiente` (`main.py:43`).
- **Observação:** RAM do Atari (`obs_type="ram"`), um vetor de **128 valores** entre 0 e 255. Em `Agente.agir` (`main.py:63`) a RAM é convertida para `float32` e normalizada para `[0, 1]` antes de entrar na rede.
- **Ações:** as **seis ações nativas** do Carnival: `0 = NOOP`, `1 = FIRE`, `2 = RIGHT`, `3 = LEFT`, `4 = RIGHTFIRE`, `5 = LEFTFIRE` (`NUM_ACOES = 6`, `main.py:21`).
- **Rede (DNA do indivíduo):** 128 entradas → 16 neurônios ocultos com `tanh` → 6 saídas. O maior valor de saída (`argmax`) define a ação. Pesos e biases `W1`, `b1`, `W2`, `b2` (`Agente.__init__`, `main.py:56-60`).
- **Recompensa e fitness:** a recompensa é o sinal retornado pelo ambiente a cada passo; o **fitness** é a soma das recompensas do episódio (`avaliar`, `main.py:70`).
- **Fim do episódio:** termina em `terminated` ou `truncated`. Como o Carnival não impõe um limite padrão, o projeto limita cada episódio a **500 ações** (`LIMITE_PASSOS_EPISODIO`, `main.py:25`). O mesmo limite é usado no treino, na validação e no baseline.

## Configuração evolutiva

| Parâmetro | Valor | Onde |
| --- | --- | --- |
| Tamanho da população | 20 indivíduos | `TAMANHO_POPULACAO`, `main.py:22` |
| Número de gerações | 30 | `NUM_GERACOES`, `main.py:23` |
| Indivíduos preservados por elitismo | 2 (sem mutação) | `NUM_ELITE`, `main.py:24` |
| Estratégia de seleção | os **25% melhores** por fitness (5 pais) | `selecionar`, `main.py:152` |
| Reprodução | cópia independente (`deepcopy`) de um pai sorteado entre os selecionados | `nova_geracao`, `main.py:167` |
| Mutação | máscara aleatória por parâmetro + ruído gaussiano | `mutar`, `main.py:158` |

**Seleção:** `selecionar` (`main.py:152`) ordena a população pelo fitness e devolve os `max(2, 20 // 4) = 5` melhores como conjunto de pais. Esse critério foi escolhido por ser simples de explicar e por dar mais chance de reprodução a quem obteve mais recompensa.

**Elitismo:** `nova_geracao` (`main.py:167`) copia os `NUM_ELITE = 2` melhores **sem mutação**, garantindo que a melhor solução nunca se perca. Em seguida, completa a população com filhos (cópias independentes mutadas) até voltar a 20 indivíduos.

**Mutação:** em `mutar` (`main.py:158`), cada peso/bias tem probabilidade igual à **taxa de mutação** de ser alterado; quando é alterado, recebe um ruído gaussiano de média 0 e desvio-padrão igual à **intensidade de mutação**.

## Experimentos de mutação (obrigatórios)

| Experimento | Taxa | Intensidade | Hipótese |
| --- | ---: | ---: | --- |
| A | 1% | 0,05 | Poucas alterações e pequenas; pouca exploração, progresso lento. |
| B | 10% | 0,20 | Configuração intermediária. |
| C | 40% | 0,50 | Muita diversidade, com risco de destruir boas soluções. |

Definidos em `EXPERIMENTOS` (`main.py:36`).

## Como executar (sem alterar o código)

Requer Python 3.10 ou superior. Em um terminal aberto na pasta do projeto:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

`python main.py` executa o baseline aleatório e os três experimentos (20 indivíduos × 30 gerações cada), salva os resultados e compara o melhor agente com o baseline. A execução pode levar alguns minutos. O comportamento pode ser ajustado apenas por argumentos de linha de comando, sem editar o arquivo:

```bash
python main.py --experimento A          # rodar somente o experimento A (ou B, ou C)
python main.py --seed 7                 # outra semente (padrão: 42)
python main.py --episodios-baseline 20  # mais episódios de baseline/validação
python main.py --render-melhor          # abrir o Carnival e demonstrar o melhor ao final
python main.py --demonstrar-salvo       # demonstrar o modelo salvo, sem treinar de novo
```

## Resultado dos três experimentos

Execução padrão (seed 42, 10 episódios de baseline/validação, limite de 500 ações por episódio):

| Experimento | Melhor episódio no treino | Média de fitness na geração 30 | Média na validação (10 episódios) |
| --- | ---: | ---: | ---: |
| A (1% / 0,05) | 1.580 | 977 | 808 |
| B (10% / 0,20) | 1.620 | 654 | 602 |
| C (40% / 0,50) | 1.900 | 486 | 728 |
| Baseline aleatório | — | — | 820 |

O melhor indivíduo pelo fitness de treino foi **C** (1.900 em um episódio isolado). Avaliado nos mesmos dez episódios de validação, sua média foi **728**, contra **820** do agente aleatório: **não superou o baseline** nesta execução (diferença de −92). Os valores pareados estão em `resultados/comparacao_baseline.csv`. A melhor configuração por média de validação foi **A** (808); a escolha do agente a comparar com o baseline usa o fitness de treino, sem escolher o vencedor após olhar a validação.

### Gráfico com melhor fitness e fitness médio

`resultados/fitness.png` apresenta **dois painéis**: o melhor fitness por geração e o fitness médio por geração, com as curvas dos experimentos A, B e C e legenda. O gráfico é gerado por `salvar_grafico` (`main.py:252`) com Matplotlib; se o Matplotlib não estiver disponível, um gerador de PNG interno (`_salvar_grafico_png_basico`, `main.py:284`) produz o mesmo conteúdo mínimo.

## Análise sobre exploração e convergência

**Exploração:** surge em dois momentos — na **população inicial aleatória** (`Agente.__init__`, `main.py:56-60`) e nas **mutações** dos descendentes (`mutar`, `main.py:158`). A escolha aleatória de um pai entre os selecionados também injeta variação. Taxa de mutação próxima de zero reduz a diversidade e pode estagnar a busca; taxa e intensidade muito altas alteram muitos parâmetros e podem destruir combinações úteis.

**Convergência:** **não há evidência de convergência estável** nesta execução. As médias das três populações partiram de 178 na primeira geração e chegaram, na geração 30, a 977 (A), 654 (B) e 486 (C) — houve melhora em relação ao início. Porém, as curvas oscilam bastante e o melhor fitness não cresce de forma monotônica, sem estabilização por várias gerações. A teve a maior média final e de validação; C gerou o maior episódio isolado, mas com média final e de validação menores. Como cada indivíduo de treino é avaliado em apenas um episódio, há bastante ruído — uma melhoria natural seria avaliar cada indivíduo em vários episódios durante a evolução. Confira `resultados/fitness.csv` e `resultados/fitness.png`.

## Arquivos gerados

Após rodar `python main.py`, a pasta `resultados/` contém:

- `fitness.csv`: melhor, média e pior fitness de cada geração (1–30) e experimento.
- `fitness.png`: gráfico de melhor fitness e fitness médio por geração.
- `comparacao_baseline.csv`: baseline aleatório e melhor agente evoluído, nos mesmos episódios de validação.
- `melhor_A.npz`, `melhor_B.npz`, `melhor_C.npz`: pesos do melhor agente de cada experimento.
- `melhor_agente.npz`: pesos do agente vencedor, usado por `--demonstrar-salvo`.

## Estrutura do projeto

```
projeto/
├── main.py               # ambiente, agente, evolução, baseline, persistência e gráfico
├── README.md             # este arquivo
├── requirements.txt      # dependências Python
├── RESPOSTAS_DEFESA.md   # respostas às perguntas da defesa
└── resultados/
    ├── fitness.csv
    ├── fitness.png
    ├── comparacao_baseline.csv
    └── melhor_*.npz
```

## Perguntas da defesa

## 1. O que é o agente e qual estado ele recebe do ambiente?

O agente é a política implementada na classe `Agente` (`main.py:53`). Seu método `agir(estado)` (`main.py:63`) recebe a observação do ambiente — um vetor de **128 bytes da RAM do Atari** (`obs_type="ram"`) — e devolve um inteiro de ação.

Os valores da RAM (0–255) são normalizados para `[0, 1]` logo no início de `agir`. A rede é 128 entradas → 16 neurônios ocultos com `tanh` → 6 saídas; o índice da maior saída (`argmax`) determina a ação do Carnival (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE ou LEFTFIRE). O estado é entregue ao agente no laço de `avaliar` (`main.py:82`): `acao = agente.agir(estado)`.

## 2. Qual a diferença entre ação, recompensa e fitness?

- **Ação:** comando escolhido pelo agente em um passo (`argmax` das saídas, `main.py:67`).
- **Recompensa (reward):** sinal numérico que o ambiente retorna após cada ação — obtido em `env.step(acao)` (`main.py:83`).
- **Fitness:** medida usada pelo algoritmo evolutivo para comparar indivíduos. Aqui é a **soma das recompensas do episódio**, acumulada em `recompensa_total` e gravada em `agente.fitness` (`main.py:84-86`).

Ou seja: a recompensa é recebida a cada passo; o fitness é o acumulado do episódio e é o valor que a seleção usa. Recompensa ≠ fitness.

## 3. Como indivíduo, população e geração aparecem no código?

- **Indivíduo:** cada instância de `Agente` com seus próprios `W1`, `b1`, `W2`, `b2` — esses parâmetros são o "DNA" (`main.py:56-60`).
- **População:** a lista de 20 agentes criada por `criar_populacao` (`main.py:148`) e mantida em `executar_experimento`.
- **Geração:** uma iteração completa do laço `for indice_geracao in range(NUM_GERACOES)` (`main.py:190`): avalia todos os indivíduos, registra as estatísticas e chama `nova_geracao` (`main.py:215`). O programa repete isso por 30 gerações em cada configuração de mutação.

## 4. Como a seleção foi implementada e por que esse critério foi escolhido?

`selecionar` (`main.py:152`) ordena a população por fitness (maior primeiro) e devolve os `max(2, TAMANHO_POPULACAO // 4) = 5` melhores, ou seja, os **25% melhores**. Esses formam o conjunto de pais. Na reprodução (`nova_geracao`, `main.py:174-177`), cada filho é um `deepcopy` de um pai **sorteado aleatoriamente** entre os selecionados, seguido de mutação.

Esse critério foi escolhido porque é simples de explicar, dá maior chance de reprodução a quem obteve mais recompensa e mantém mais de um possível pai (preservando diversidade em vez de um único vencedor).

## 5. O que aconteceria se não existisse elitismo?

Sem elitismo, o melhor DNA de uma geração poderia desaparecer na próxima por causa da seleção aleatória, da reprodução ou da mutação. Como a avaliação tem aleatoriedade, a população poderia perder uma solução boa e apresentar queda de desempenho.

No código, `nova_geracao` (`main.py:173`) copia os `NUM_ELITE = 2` melhores **sem nenhuma mutação** para a geração seguinte — é a garantia de que a melhor solução não regride. É por isso que o melhor fitness registrado nunca cai abaixo do melhor já encontrado.

## 6. Como taxa e intensidade de mutação afetam os descendentes?

Em `mutar` (`main.py:158-164`):

- **Taxa** (`configuracao.taxa_mutacao`): probabilidade de cada peso/bias ser alterado. Implementada pela máscara `np.random.random(shape) < taxa`.
- **Intensidade** (`configuracao.intensidade_mutacao`): desvio-padrão do ruído gaussiano somado aos parâmetros sorteados: `ruido = np.random.normal(0.0, intensidade, shape)`.

Foram comparadas as três configurações do enunciado (`EXPERIMENTOS`, `main.py:36`):

| Experimento | Taxa | Intensidade | Efeito observado |
| --- | ---: | ---: | --- |
| A | 1% | 0,05 | Poucas mudanças, pequenas; seleção conservadora. Teve a maior média final e de validação. |
| B | 10% | 0,20 | Intermediária; média final 654. |
| C | 40% | 0,50 | Mais diversidade; gerou o maior episódio isolado (1.900), mas a menor média final (486). |

## 7. Onde ocorre exploração no algoritmo?

A exploração acontece em três pontos:

1. **População inicial aleatória** — cada agente começa com pesos `np.random.randn(...)` (`main.py:57-60`).
2. **Mutações** — os descendentes recebem ruído gaussiano, introduzindo novas soluções (`mutar`, `main.py:158`).
3. **Sorteio do pai** entre os selecionados — permite que diferentes indivíduos contribuam para as próximas gerações (`main.py:175`).

## 8. Os resultados indicam convergência? Qual evidência sustenta a resposta?

**Não há evidência de convergência estável.** A média da primeira geração foi 178 nos três experimentos. Na geração 30, as médias foram 977 (A), 654 (B) e 486 (C) — houve melhora em relação ao início. Porém, as curvas oscilam bastante e o melhor fitness não cresce de forma monotônica, sem estabilização consistente por várias gerações.

Essa conclusão é sustentada por `resultados/fitness.csv` (melhor, média e pior por geração) e por `resultados/fitness.png` (curvas de melhor e média). Além disso, a evidência vem de uma única seed, então não se pode afirmar que a mesma tendência ocorrerá em toda repetição.

## 9. O melhor agente superou o agente aleatório? Qual evidência mostra isso?

**Não nesta execução.** O melhor agente pelo fitness observado no treino foi o do experimento **C**, com 1.900 em um episódio isolado. Avaliado em dez episódios de validação, sua média foi **728**, enquanto o baseline aleatório, nas mesmas dez seeds e com o mesmo limite de passos, obteve média **820**. Logo, o agente evoluído ficou **92 pontos abaixo** do baseline.

A evidência pareada está em `resultados/comparacao_baseline.csv`. Importante: o valor 1.900 é um episódio isolado de treino e **não** deve ser comparado diretamente com a média do baseline. As médias de validação dos melhores agentes de cada experimento foram A: 808, B: 602 e C: 728 — a escolha do agente a comparar com o baseline segue o fitness de treino, sem escolher o vencedor depois de examinar a validação.

## 10. Qual mudança vocês fariam para tentar melhorar o resultado?

Avaliar cada indivíduo em **vários episódios** durante o próprio processo evolutivo e usar a **média** como fitness. Hoje cada indivíduo é avaliado em um único episódio (`avaliar(agente, seed=...)`, `main.py:194`), o que introduz bastante ruído na seleção — um agente pode vencer por sorte em uma partida. Usar várias seeds reduziria a variância e faria a seleção favorecer quem é consistente. Como complemento, rodar o experimento com **múltiplas seeds** (`--seed`) para confirmar se as tendências se mantêm, em vez de tirar conclusões de uma única execução.

---

## Critério central: conceito → código → evidência

| Conceito | Trecho do código | Evidência |
| --- | --- | --- |
| Agente | `class Agente` / `agir` (`main.py:53-67`) | ação escolhida por `argmax` em cada passo |
| Ambiente | `criar_ambiente` (`main.py:43`) | `ALE/Carnival-v5`, RAM `(128,)`, `Discrete(6)` |
| Ação | `int(np.argmax(saida))` (`main.py:67`) | 0–5 traduzidos em NOOP/FIRE/RIGHT/LEFT/RIGHTFIRE/LEFTFIRE |
| Recompensa | `env.step(...)` (`main.py:83`) | sinal somado a cada passo |
| Fitness | `recompensa_total` → `agente.fitness` (`main.py:86`) | coluna `melhor`/`media` em `fitness.csv` |
| Indivíduo | `W1`, `b1`, `W2`, `b2` (`main.py:56-60`) | modelos `melhor_*.npz` |
| População | `criar_populacao` (`main.py:148`) | 20 agentes por geração |
| Geração | laço em `executar_experimento` (`main.py:190`) | 30 gerações por experimento |
| Seleção | `selecionar` (`main.py:152`) | 25% melhores como pais |
| Elitismo | `nova_geracao` (`main.py:173`) | 2 melhores preservados sem mutação |
| Mutação | `mutar` (`main.py:158`) | taxa/intensidade dos experimentos A, B, C |
| Exploração | população inicial + `mutar` + sorteio do pai | diversidade entre gerações |
| Convergência | análise de `fitness.csv`/`fitness.png` | curvas oscilam; sem estabilização |
| Baseline | `avaliar_aleatorio` (`main.py:92`) | `comparacao_baseline.csv`: 820 vs 728 |
