# SeismicPyGL 3D — Simulador de Terremotos (PyOpenGL + Pygame)

Simulador 3D de terremotos de alto desempenho em Python com **Pygame + PyOpenGL**, utilizando **pipeline programável moderno (GLSL 3.3 Core)**, buffers VBO/VAO, câmera com screen shake por trauma amortecido por Ruído de Perlin, deformação de terreno por onda senoidal diretamente na GPU, modelo físico de colapso de edifícios, partículas de poeira e fumaça com billboarding esférico e alpha blending, e HUD 2D com isolamento de profundidade.

![Demonstração da Simulação](screenshot_simulation.png)

---

## 🎮 Controles Interativos

| Tecla / Ação | Função |
| :--- | :--- |
| `W` | Anda para frente na direção da visão |
| `A` | Anda para a esquerda |
| `D` | Anda para a direita |
| `S` | Anda para trás (a câmera permanece na altura do chão) |
| `SHIFT` | Corrida rápida (sprint) |
| `MOUSE` | Olha ao redor com sensibilidade controlada (limite vertical para evitar inversão) |
| `SCROLL` | Zoom na câmera (FOV dinâmico) |
| `ESPAÇO` | Dispara terremoto de magnitude intermediária (5.5 Richter) |
| `1` a `5` | Dispara terremotos em intensidades calibradas (3.0 a 8.5 na Escala Richter) |
| `R` | Reseta a cidade, remove destroços e restaura a câmera inicial |
| `ESC` | Fecha o simulador com liberação segura de recursos |

---

## 🛠️ Tecnologias e Arquitetura

- **Pipeline Programável Moderno (GLSL 3.3 Core):** Sem `glBegin/glEnd` ou matrizes de função fixa legadas.
- **Deformação de Terreno na GPU (`assets/shaders/ground.vert`):** A equação de onda radial senoidal e o recálculo analítico dos vetores normais são executados inteiramente nos núcleos da GPU, sobre uma malha do chão indexada via EBO (vértices compartilhados entre células vizinhas, em vez de duplicados por triângulo).
- **Screen Shake por Trauma (`src/core/camera.py` + `src/core/math_utils.py`):** Modelo de trauma $[0.0, 1.0]$ com queda cúbica ($Trauma^3$) e amostragem de Ruído de Perlin para dessincronizar Pitch, Yaw, Roll e Translação. A câmera também colide com prédios, casas, árvores, a montanha e postes de iluminação em pé — não atravessa mais a geometria do mundo.
- **Colapso Estrutural com Efeito Chicote (`src/world/building.py`, `mountain.py`, `nature.py`, `light_pole.py`):** Edifícios fatiados que acumulam dano mecânico, sofrem inclinação progressiva e afundamento na base no eixo Y ($M = T \times R \times S$); árvores e postes caem fisicamente sob tremores fortes.
- **Escombros Instanciados (`src/world/debris_renderer.py`):** Todos os pedaços de entulho de prédios e rochas da montanha são desenhados em 1-2 `glDrawArraysInstanced`, em vez de uma chamada de desenho por pedaço — essencial durante o colapso simultâneo de vários prédios.
- **Sistema de Partículas Vetorizado (`src/simulation/particles.py`):** Armazenamento em arrays NumPy (Structure of Arrays) em vez de uma lista de objetos Python; `emit`/`update`/montagem do buffer de instância totalmente vetorizados.
- **Shadow Mapping Direcional (`src/rendering/shadow_map.py`):** Passe de profundidade dedicado (framebuffer + textura de depth) antes do passe principal, com binds de textura/uniforms de material pulados nesse passe (não são lidos pelo shader de sombra).
- **HUD 2D Ortográfico (`src/rendering/hud.py`):** Exibe métricas em tempo real (Escala Richter, trauma da câmera, contador de prédios/casas/postes e FPS) com isolamento total do buffer de profundidade (`glDisable(GL_DEPTH_TEST)`).
- **Seleção Automática de GPU Dedicada:** Sob WSL, força o Mesa a usar a GPU via tradução D3D12; em Linux nativo com driver NVIDIA detectado (notebooks híbridos Intel+NVIDIA sob Wayland), ativa PRIME render offload por padrão — sem precisar de flags manuais na maioria dos casos.
- **Suíte de Testes Automatizados (`tests/`, `pytest`):** Testes unitários para toda a lógica pura do projeto (física, geração procedural, matemática) e testes de fumaça com um contexto OpenGL real para as classes de renderização.

---

## 📦 Como Instalar e Executar

### 1. Criar ambiente virtual (recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Pré-processamento das texturas PBR (conversão EXR -> PNG 2K)

Antes de rodar pela primeira vez com os pacotes PBR, execute o conversor otimizado de EXR para PNG:

```bash
python tools/convert_exr_textures.py
```
*(Para manter resolução 4K máxima sem redimensionamento para 2048x2048, use a flag `--full-4k`)*

### 4. Executar o simulador

O jogo tenta usar a GPU dedicada automaticamente: sob WSL, força o Mesa a
acessar a placa via tradução D3D12; em Linux nativo com driver NVIDIA
detectado (ex.: notebooks híbridos Intel+NVIDIA sob Wayland), ativa PRIME
render offload por padrão. Confira a linha `[Hardware 3D] GPU:` impressa no
console ao iniciar para conferir qual GPU está sendo usada.

```bash
python main.py
```

### 5. Rodar os testes automatizados

```bash
pytest tests/ -v
```

---

## 📁 Estrutura do Projeto

```
SeismicPyGL/
├── main.py                      # Loop de eventos, orquestração e render pass (shadow + cena + HUD)
├── src/
│   ├── core/                    # Matemática 3D, câmera, shaders, malhas, texturas
│   │   ├── camera.py            # FreeCamera: Euler angles, Trauma/Perlin Screen Shake, colisão com o mundo
│   │   ├── math_utils.py        # Matrizes 4x4 (MVP) e Perlin Noise 1D/2D puro
│   │   ├── mesh.py              # VAO/VBO/EBO com dados entrelaçados
│   │   ├── obj_loader.py        # Leitor Wavefront .obj e geradores procedurais de malha
│   │   ├── shader.py            # Compilador/gerenciador de Programas GLSL
│   │   └── texture.py           # Loader PIL + fallback procedural (com aviso de textura ausente)
│   ├── simulation/              # Física sísmica e partículas
│   │   ├── earthquake.py        # Propagação de onda circular e parâmetros Richter
│   │   └── particles.py         # ParticleSystem vetorizado (NumPy SoA) com Billboards
│   ├── world/                   # Entidades da cena e geração procedural
│   │   ├── building.py          # Colapso estrutural e escombros
│   │   ├── ground.py            # Terreno deformável (malha indexada via EBO)
│   │   ├── nature.py            # Árvores (queda física)
│   │   ├── light_pole.py        # Postes de iluminação (queda física)
│   │   ├── mountain.py          # Montanha procedural e queda de rochas
│   │   ├── debris_renderer.py   # Escombros de prédios/montanha em 1-2 draw calls instanciados
│   │   ├── village.py           # generate_village: prédios, casas, ruas e postes
│   │   └── shared.py            # Malhas/materiais PBR compartilhados (cache)
│   └── rendering/                # HUD, sombra e céu
│       ├── hud.py
│       ├── shadow_map.py         # Shadow mapping direcional (passe de profundidade)
│       └── sky.py                # Céu HDRI panorâmico
├── tests/                        # Suíte pytest: lógica pura + testes de fumaça com GL real
├── assets/
│   ├── shaders/                  # scene, ground, billboard, hud, shadow, sky, debris (.vert/.frag)
│   ├── textures/                 # Pacotes PBR (albedo/normal/roughness) + fallback procedural
│   └── models/                   # Modelos 3D .obj
├── requirements.txt              # Dependências fixadas
└── README.md
```
