# savi-2025-2026-trabalho1-grupo10

### Registo 3D com ICP e Esfera Englobante Mínima  
**Autores:** João Menício (93300) e Pascoal Sumbo (123190)  
**Unidade Curricular:** Sistemas Autónomos e Visão Inteligente (SAVI) – 2025

#  Introdução

Este repositório contém a resolução do Trabalho Prático SAVI, focado no registo de nuvens de pontos 3D obtidas a partir de scans RGB-D. O trabalho explora o algoritmo Iterative Closest Point (ICP) em diferentes níveis de abstração, desde o uso de ferramentas nativas do Open3D até a implementação de um otimizador personalizado com scipy.optimize.least_squares. Por fim, é implementada uma otimização de minimização para encontrar a Esfera Englobante Mínima das nuvens de pontos registradas.
Este trabalho aborda três tarefas fundamentais no contexto de registo 3D e reconstrução geométrica:

O objetivo deste trabalho é implementar e comparar diferentes abordagens de registo 3D com base em nuvens de pontos obtidas a partir de imagens RGB-D, bem como explorar uma aplicação simples de otimização geométrica (cálculo da esfera englobante mínima).

O trabalho está dividido em três tarefas:

1. **Tarefa 1** – ICP utilizando as funções nativas do Open3D.  
2. **Tarefa 2** – ICP personalizado, implementado “à mão” com otimização de mínimos quadrados.  
3. **Tarefa 3** – Cálculo da esfera englobante mínima após o alinhamento das nuvens.

## Dependências
 As seguintes bibliotecas são necessárias para executar os scripts:

 + open3d

 + NumPy

 + scipy

 + opencv-python

 + matplotlib
   
## 1 Tarefa 1 – ICP Nativo do Open3D

### 1.1. Abordagem

O ficheiro `main_ipc.py` implementa o registo entre duas vistas RGB-D:

1. Leitura das imagens `1.png`, `depth1.png`, `2.png`, `depth2.png`.
2. Conversão para `RGBDImage` com `create_from_tum_format`.
3. Geração de nuvens de pontos `pcd1` e `pcd2` com intrínsecos `PrimeSenseDefault`.
4. Aplicação de uma transformação de *flip* no eixo Z para compatibilizar o referencial do TUM dataset com o do Open3D.
5. **Downsampling** com voxel grid (`voxel_size = 0.025`) para reduzir ruído e número de pontos.
6. Estimativa de normais em ambas as clouds.
7. Execução do ICP do Open3D, com:
   - método **point-to-plane**  
   - `max_correspondence_distance = 0.6`  
   - até 500 iterações
8. Visualização final:  
   - alvo (target) a verde  
   - fonte (source) alinhada a vermelho.

### Resultados Obtidos
A execução do script produziu: 

### Visualização 1
![Resultado ICP 2](images/tarefa1.111.png)
### Visualização 2
![Resultado ICP 1](images/tarefa1.11.png)

### visualização
A figura sugere que:

As superfícies coincidem na maior parte das regiões

As discrepâncias existentes surgem apenas em zonas com ruído ou ausência de profundidade

As cores misturam-se (verde+vermelho → amarelo), indicando bom alinhamento
### Interpretação dos Resultados

### Fitness = 0.9964

Indica que 99.64% dos pontos da fonte encontraram correspondências válidas na target dentro do limite de 0.6 m.

Este valor é excelente, e significa que:

as duas clouds são geometricamente compatíveis, a sobreposição é muito elevada e há poucas áreas sem match.

### Inlier RMSE = 0.107 m

O erro médio entre correspondências após o alinhamento é cerca de 10.7 cm.

Este valor é consistente com:

Ruído da depth map

Descontinuidades

Superfícies inclinadas

Passages oclusas entre as imagens.

### Transformação estimada

A matriz de rotação aproxima-se de uma matriz de rotação válida (ortogonal):

rotação em torno do eixo Y ≈ −11°

Ligeiras rotações nos restantes eixos

tradução:

x = +0.958 m

y = +0.014 m

z = −0.036 m

O método Point-to-Plane provou ser o mais adequado para estas clouds densas.

A convergência atingiu valores ótimos (fitness > 0.99).

A transformação encontrada é fisicamente plausível e consistente com o movimento da câmera.

O ICP nativo é extremamente estável e fornece um baseline ideal para comparar com a implementação personalizada da Tarefa 2.

### Desafios Encontrados

O ruído nas imagens depth exigiu downsampling para estabilizar o ICP.

A necessidade de corrigir o sistema de coordenadas (eixo Z invertido).

Os parâmetros (max_correspondence_distance e voxel_size) influenciam bastante a convergência — valores muito pequenos impediam a formação de pares suficientes. 



