# Análise Técnica — SeismicPyGL

> Documento de estudo do código-fonte do simulador **SeismicPyGL**: como o
> projeto está organizado, como cada módulo funciona e o que pode ser
> melhorado. Escrito a partir da leitura integral de `main.py`, dos pacotes
> `src/core`, `src/simulation`, `src/rendering`, `src/world`, dos 15 shaders
> GLSL em `assets/shaders/` e da ferramenta `tools/convert_exr_textures.py`
> (estado do repositório em 2026-09-08, branch `master`, após a rodada de
> otimização de performance descrita na Seção 12).

---

## 1. Visão geral

SeismicPyGL é um simulador 3D de terremotos escrito em Python, usando:

- **Pygame** para janela, input e contexto OpenGL;
- **PyOpenGL** para o pipeline gráfico programável (**GLSL 3.3 Core**, sem
  `glBegin/glEnd` nem matrizes de função fixa);
- **NumPy** para toda a álgebra linear (matrizes 4x4, vetores) e geração
  procedural de malhas/texturas;
- **Pillow (PIL)** para carregar/gerar texturas e fazer upload para a GPU.

O "produto" é uma vila 3D (prédios, casas, ruas, postes, árvores, uma
montanha) que pode ser sacudida por terremotos de magnitude Richter
configurável. O tremor deforma o terreno na GPU, danifica e derruba
edifícios/postes/árvores progressivamente, emite partículas de poeira/fumaça,
e é acompanhado por *screen shake* de câmera baseado em Ruído de Perlin.

Como rodar (resumo do `README.md`, que está atualizado no essencial mas
ainda referencia a antiga estrutura de arquivos "flat" pré-refactor):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/convert_exr_textures.py   # opcional, só se houver .exr em assets/textures/pbr
python main.py
```

`SEISMICPYGL_RESOLUTION=4k python main.py` ativa janela 4K e texturas
procedurais em maior resolução.

---

## 2. Arquitetura de pastas

```
main.py                    # loop principal e orquestração de todo o frame
src/
├── core/                  # matemática, câmera, shader, mesh, obj loader, textura
├── simulation/             # física do terremoto e sistema de partículas
├── rendering/              # shadow map, céu HDRI, HUD 2D
└── world/                  # entidades da cena (chão, prédios, ruas, postes,
                             # árvores, montanha) + geradores procedurais de vila
assets/
├── shaders/                # 6 pares .vert/.frag (scene, ground, billboard, shadow, sky, hud)
├── textures/pbr/<material>/textures/  # pacotes PBR (albedo/normal/roughness)
└── models/                 # .obj (praticamente não usado — geometria é procedural)
tools/convert_exr_textures.py   # pré-processamento EXR → PNG
```

Cada subpacote expõe sua API via `__init__.py`, e `main.py` importa tudo por
cima (`from src.core import ...`, `from src.world import ...`), o que mantém
os imports do orquestrador limpos. Internamente os módulos usam import duplo
(`try: from ..core.x import Y except ImportError: from x import Y`) para
funcionar tanto como pacote (`python main.py`) quanto se um arquivo for
executado isoladamente — ver seção de falhas para uma ressalva sobre isso.

---

## 3. `main.py` — orquestração e ciclo de frame

### 3.1 Setup de plataforma (linhas 12–39)

Antes de importar Pygame/OpenGL, o módulo detecta o ambiente e ajusta
variáveis de ambiente em dois casos distintos (corrigido na Seção 12 — antes
o hack de WSL disparava cegamente em qualquer Linux, ver Seção 10.1):

```python
def _running_under_wsl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if "microsoft" in platform.uname().release.lower():
        return True
    return os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")

if _running_under_wsl():
    os.environ["GALLIUM_DRIVER"] = "d3d12"
    os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
    os.environ["PYOPENGL_PLATFORM"] = "glx"
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + ...
elif sys.platform.startswith("linux") and os.environ.get("SEISMICPYGL_FORCE_X11") == "1":
    os.environ["SDL_VIDEODRIVER"] = "x11"
```

O primeiro bloco só existe para fazer o Mesa usar a GPU NVIDIA dedicada
**dentro do WSL** (que expõe a GPU via D3D12), e agora só ativa sob WSL de
verdade. O segundo é um **opt-in** (`SEISMICPYGL_FORCE_X11=1 python main.py`)
para um problema diferente e mais sutil descoberto em notebooks híbridos
Intel+NVIDIA rodando Wayland: o SDL pode criar o contexto OpenGL pelo
caminho EGL nativo do Wayland, que não consulta a seleção de GPU do GLVND
(`__GLX_VENDOR_LIBRARY_NAME`, usada por PRIME render offload) — nesse caso o
jogo roda silenciosamente na GPU integrada mesmo com uma dedicada disponível
e "configurada" no sistema. Forçar `SDL_VIDEODRIVER=x11` faz o SDL usar
XWayland/GLX, que respeita essa seleção. Não é o padrão porque, em pelo
menos um ambiente de teste, essa flag quebrou a criação do contexto OpenGL
(`OpenGL.error.Error: Attempt to retrieve context when no valid context`) —
ver Seção 12.1.

### 3.2 Inicialização (`init_opengl`, `main`)

- Cria a janela com `DOUBLEBUF | OPENGL` e MSAA 4x (`GL_MULTISAMPLESAMPLES`);
- Ativa `GL_DEPTH_TEST` e `GL_MULTISAMPLE`, define a cor de fundo (céu azul
  claro, embora o céu real venha do skybox HDRI depois);
- Prende o cursor do mouse (`event.set_grab` + `mouse.set_visible(False)`)
  para visão em primeira pessoa;
- Instancia todo o mundo: `EarthquakeSimulator`, `Ground` (grade 220×220
  divisões), `generate_village(...)` (9 prédios + 24 casas + ruas + postes),
  uma `Mountain` fora do centro, uma floresta de 120 árvores evitando as
  zonas ocupadas, o `ParticleSystem` (até 6000 partículas), `HUD`,
  `ShadowMap` e `Sky`.
- Pré-calcula `light_space_matrix` (posição de luz fixa em `(42, 65, 36)`
  olhando para a origem, projeção ortográfica ±72 unidades) — a cena inteira
  usa uma única luz direcional estática, sem ciclo dia/noite.

### 3.3 Loop principal (`while running`)

Por frame (`dt` travado em 60 FPS via `clock.tick(60)`, e clampado a 0.05s
para evitar "saltos" físicos em caso de lag):

1. **Eventos**: `ESC` sai; `SPACE` dispara terremoto magnitude 5.5;
   `1`–`5` disparam magnitudes calibradas (3.0 a 8.5 Richter) e injetam
   trauma de câmera correspondente; `R` reseta todos os objetos e a câmera;
   scroll ajusta o FOV (zoom).
2. **Câmera**: `process_mouse` (rotação), `process_keyboard` (movimento
   WASD + sprint), `update_trauma` (decaimento + shake via Perlin).
3. **Física**: cada `Building`, `Tree`, `LightPole` e a `Mountain` chamam
   `update(earthquake, elapsed_time, dt, ...)` — cada um consulta
   independentemente `earthquake.get_offset(x, z, t)` para saber quanto a
   onda sísmica deslocou aquele ponto do mundo naquele instante.
   `ParticleSystem.update` avança partículas existentes e, fora de
   terremoto, `emit_ambient` gera poeira/pólen e folhas caindo perto da
   câmera para dar vida à cena parada.
4. **Renderização**, em ordem:
   - **Shadow pass**: liga o FBO de profundidade (`ShadowMap.begin`),
     desenha ruas/prédios/montanha/árvores/postes só com posição (shader
     `shadow.vert/frag`, sem cor) — cada `draw(..., shadow_pass=True)` pula
     todo o bind de textura/uniforms PBR nesse passe, já que o shader de
     sombra nem os lê (ver Seção 12.2) — mais `debris_renderer.draw_shadow(...)`
     para os escombros, e fecha o FBO;
   - **Sky**: desenha o skybox equirretangular sem depth test/write, usando
     só a parte de rotação da view matrix (translação zerada) para simular
     "infinito";
   - **Ground**: chão deformável, com shader dedicado (recebe os parâmetros
     do terremoto e a shadow map);
   - **Scene**: um único `scene_shader` é reutilizado para ruas, postes,
     prédios/casas, montanha e árvores — cada `draw()` seta um conjunto de
     uniforms/flags (`u_is_street`, `u_is_house`, `u_building_facade`,
     `u_mountain_stratum`, `u_is_foliage`) que o fragment shader usa para
     ramificar comportamento (ver Seção 5). Postes e árvores passam por um
     **culling barato por ângulo de visão** (`_in_view`, Seção 12.3) antes de
     serem desenhados nesse passe;
   - **Escombros**: `debris_renderer.draw(...)` desenha todo o escombro de
     prédios/montanha vivo naquele frame em 1–2 draw calls instanciados
     (Seção 7.8), depois do `scene_shader.stop()`;
   - **Partículas**: desenhadas por último com blending e sem escrita no
     depth buffer;
   - **HUD**: desenhado por cima de tudo, com depth test desabilitado
     explicitamente (é reativado no fim do método `HUD.draw`).
5. `pygame.display.flip()`.

### 3.4 Encerramento

Chama `cleanup()` em `ground`, `mountain`, `particle_system`,
`debris_renderer`, `hud`, `shadow_map`, `sky`, `scene_shader` e
`cleanup_textures()` (globais de `core/texture.py`) antes de `pygame.quit()`.
Ver Seção 6.3 para o que fica de fora dessa limpeza.

---

## 4. `src/core` — fundações

### 4.1 `math_utils.py`

Implementa manualmente (sem `glm`/`pyrr`) as matrizes que qualquer motor 3D
precisa: `perspective`, `look_at`, `ortho`, `translate`, `scale`,
`rotate_x/y/z`. As matrizes são armazenadas como arrays NumPy "linha a
linha" (row-major, do jeito que é natural de ler/escrever em Python) e
convertidas para o layout column-major que o OpenGL espera só na hora de
subir para a GPU, via `to_gl_matrix` (`matrix.T` contíguo). Essa escolha
separa "como eu penso a matriz" de "como o driver espera os bytes" — comum
em implementações didáticas.

Também contém uma implementação pura em Python/NumPy do **Ruído de Perlin
clássico** (`PerlinNoise`, com `fade`, `lerp`, gradientes 1D/2D), usada tanto
pelo *screen shake* da câmera quanto pela perturbação da malha da montanha.
Uma instância global (`_default_perlin`, seed 42) é reaproveitada pelas
funções de conveniência `perlin1d`/`perlin2d`.

### 4.2 `camera.py` — `FreeCamera`

Câmera em primeira pessoa com:
- Movimento WASD relativo ao yaw (vetor "direita" calculado como perpendicular
  ao "forward" projetado no plano XZ), com sprint (Shift) e clamping da
  posição dentro de `world_bounds = 68` unidades e altura mínima
  `eye_height = 2.5`;
- Rotação por mouse direta (yaw/pitch), com pitch limitado a ±84° e um filtro
  simples que ignora deltas anômalos (`> 250px`, típico ao capturar/soltar o
  foco da janela);
- Zoom por FOV dinâmico (`fov_min=20`, `fov_max=90`);
- **Trauma & Screen Shake**: um valor de trauma em `[0, 1]` decai
  linearmente (`trauma_decay`) e alimenta um shake com queda **cúbica**
  (`trauma**3`, mais dramático perto de 1.0). Os offsets de yaw/pitch/roll e
  translação são amostrados do mesmo `perlin1d`, mas em *offsets de fase*
  diferentes (`+113.7`, `+227.4`, ...) para os eixos não oscilarem em
  sincronia — uma técnica clássica de screen shake ("trauma" no estilo GDC
  talk da Squirrel Eiserloh).
- `get_view_matrix()` monta a matriz look-at a partir de eye/target/up já com
  o shake aplicado; `apply()` existe como *fallback* para pipeline fixo via
  `gluLookAt`, mas não é usado em lugar nenhum do fluxo atual (o projeto é
  100% pipeline programável).

### 4.3 `shader.py` — `ShaderProgram`

Compila vertex+fragment shader, faz *link*, valida erros de compilação
(imprime o log do driver e levanta `RuntimeError`) e cacheia
`glGetUniformLocation` por nome (`_uniform_cache`) para evitar round-trips
repetidos à GPU. Expõe setters tipados (`set_uniform_mat4/vec2/vec3/vec4/
float/int`, e um `set_uniform_vec2_array` usado para os pontos de contato de
oclusão ambiente do chão). Todo setter verifica `loc != -1` antes de
escrever, então uniforms inexistentes/otimizadas pelo driver são ignoradas
silenciosamente — conveniente, mas também pode esconder um typo de nome de
uniform sem qualquer aviso.

### 4.4 `mesh.py` + `obj_loader.py` — geometria

- Layout de vértice **entrelaçado**: posição (3f) + UV (2f) + normal (3f) +
  tangente opcional (3f) = stride de 32 ou 44 bytes;
- `parse_obj` lê Wavefront `.obj` linha a linha, suporta triangulação em
  leque para faces com mais de 3 vértices e os três formatos de índice
  (`v`, `v/vt`, `v/vt/vn`, `v//vn`);
- `compute_tangents` calcula o vetor tangente por triângulo a partir das
  derivadas de UV (necessário para normal mapping via matriz TBN nos
  shaders), com fallback geométrico quando o determinante UV é degenerado;
- Geradores procedurais: `create_cube_mesh`, `create_plane_mesh` (grade para
  o chão, sempre triangulada — é nela que a deformação sísmica acontece no
  vertex shader), `create_quad_mesh` (billboards/HUD), `create_cylinder_mesh`
  (também serve como cone/pirâmide dependendo de `top_radius`).
- Todos os desenhos passam por `glDrawArrays` (não há Element Buffer Object /
  índices) — ver Seção 6.5.

### 4.5 `texture.py`

- `load_texture_set` carrega um conjunto PBR completo (albedo `_diff_`,
  normal `_nor_gl_`, roughness `_rough_`) de uma pasta, com cache por
  caminho absoluto e fallback procedural coerente por tipo de material
  (asfalto/grama/concreto) quando os arquivos não existem;
- Geradores procedurais (`create_concrete_texture`, `create_grass_texture`,
  `create_asphalt_texture` com faixa amarela tracejada, e
  `create_smoke_particle_texture` com gradiente radial suave para as
  partículas) usam ruído NumPy vetorizado — exceto a textura de fumaça, que
  itera pixel a pixel em Python puro (64×64, então é barato, mas destoa do
  resto vetorizado);
- `cleanup_textures()` limpa a lista global `_loaded_textures`, mas não os
  outros caches globais do módulo (ver Seção 6.3).

---

## 5. `src/simulation` — física do terremoto e partículas

### 5.1 `earthquake.py` — `EarthquakeSimulator`

Modela o terremoto como uma **onda circular** que se propaga a partir de um
epicentro `(x, z)`:

```
dist            = distância euclidiana do ponto ao epicentro
arrival         = dist / wave_speed              # tempo até a onda chegar
local_t         = (tempo desde o trigger) - arrival
temporal_decay  = exp(-damping * local_t)
spatial_decay   = exp(-spatial_falloff * dist)
amplitude       = magnitude_visual * temporal_decay * spatial_decay
dy = amplitude * sin(2π * frequency * local_t)   # vertical
dx, dz = componentes horizontais defasadas (0.35× amplitude, fase deslocada)
```

`trigger(magnitude=...)` converte a magnitude Richter (ex.: 3.0–8.5) em uma
amplitude visual (`max(0.4, magnitude * 0.42)`) e ajusta o amortecimento
(`damping`) para ser menor — ou seja, o tremor demora mais para morrer — em
terremotos ≥ 7.0 Richter. `get_crack_intensity()` mapeia a maior magnitude já
disparada (`max_richter`) para `[0, 1]` a partir de 3.8 Richter, controlando
as rachaduras visuais no chão/asfalto nos shaders.

Cada entidade do mundo chama `get_offset(x, z, t)` independentemente — não há
um "solver" central que atualiza todo mundo de uma vez; a simulação é
avaliada como uma função fechada do tempo e da posição (útil porque também é
replicável no shader do chão sem qualquer estado compartilhado).

### 5.2 `particles.py` — `ParticleSystem`

- `Particle` (com `__slots__` para reduzir overhead de memória/CPU) simula
  posição, velocidade com atrito exponencial, expansão de tamanho e fade
  alpha com curva de potência — tudo em Python puro, por partícula, a cada
  `update(dt)`;
- Renderização via **instancing** (`glDrawArraysInstanced`): um único quad é
  compartilhado (`Mesh.create_quad`) e um VBO dinâmico
  (`GL_DYNAMIC_DRAW`, atualizado a cada frame com `glBufferSubData`) carrega
  posição/tamanho/alpha/cor por instância nos atributos 3–6, com
  `glVertexAttribDivisor(..., 1)`. Isso permite desenhar milhares de
  partículas em uma única draw call;
- O **billboard esférico** é feito no vertex shader
  (`billboard.vert`), somando `camera_right`/`camera_up` (extraídos
  diretamente da view matrix) escalados pelo tamanho da partícula — assim o
  quad sempre encara a câmera, independente da rotação;
- `emit()` dispara uma rajada (usada no colapso de prédios/postes);
  `emit_ambient()` gera poeira/pólen e folhas caindo continuamente ao redor
  da câmera, com orçamento por tempo (`_ambient_dust_budget`,
  `_leaf_budget`) para controlar a taxa de emissão de forma independente do
  framerate.

---

## 6. `src/rendering` — sombra, céu, HUD

### 6.1 `shadow_map.py`

FBO **somente profundidade** (`GL_DEPTH_COMPONENT`, sem anexo de cor,
`glDrawBuffer(GL_NONE)`), resolução fixa 1024×1024, `GL_CLAMP_TO_BORDER` para
evitar *wrap* fora da área iluminada. `begin()`/`end()` trocam viewport e FBO
ativo. Usa `shadow.vert/frag`, o shader mais simples do projeto (o fragment
shader é vazio — só a profundidade escrita pelo pipeline fixo importa).

### 6.2 `sky.py`

Skybox equirretangular (textura HDRI convertida para JPG,
`meadow_2_sky.jpg`) desenhado como um quad de tela cheia: o vertex shader
recebe a inversa de `projection @ view_sem_translação` e "des-projeta" os
cantos da tela para raios de direção no espaço do mundo; o fragment shader
converte essa direção em coordenadas UV equirretangulares
(`atan(z,x)`, `asin(y)`). Depth test/write são desligados e restaurados ao
redor do desenho, e há uma leve neblina de horizonte por `smoothstep`.

### 6.3 `hud.py` — `HUD` / `TextTexture`

HUD 2D ortográfico "isolado" do depth buffer (comentário no código referencia
"Issue #18"). Texto é renderizado via `pygame.font` para uma `Surface`,
convertido para bytes RGBA (`pygame.image.tostring`) e subido como textura
OpenGL (`TextTexture`), com cache por chave (`_get_label`) para não recriar a
textura todo frame quando o texto não muda. Desenha dois painéis: métricas
sísmicas (magnitude Richter ativa, trauma da câmera com barra de cor
verde→vermelho, contagem de prédios/casas/postes intactos, FPS) e um guia de
controles fixo na parte inferior.

---

## 7. `src/world` — entidades da cena

### 7.1 `shared.py`

Registro central de recursos **compartilhados e cacheados por processo**:
meshes básicos (cubo, cone, cilindro, pirâmide, um prisma triangular
"gable" para telhados de duas águas, feito à mão como array de vértices) e
conjuntos de materiais PBR (`get_pbr_set(nome)`, que resolve
`assets/textures/pbr/<nome>/textures/` e delega para
`load_texture_set`). Como tudo é geometria idêntica reaproveitada (todo
prédio usa o mesmo cubo, só com uma matriz de modelo diferente), o número de
VAOs/VBOs alocados é pequeno independentemente de quantos prédios/árvores
existam na cena.

### 7.2 `ground.py` — `Ground`

Grade (`create_plane_mesh`, 220×220 divisões por padrão em `main.py`) cujo
shader (`ground.vert`) desloca `pos.y` **na GPU** segundo a mesma fórmula de
onda do `earthquake.py` (parâmetros repassados via uniforms), recalculando a
normal analiticamente a partir da derivada espacial da onda — em vez de
normal mapping ou recomputação por diferenças finitas, deriva-se a inclinação
diretamente da equação senoidal (`dy_ddist`), o que é elegante e barato.

O fragment shader (`ground.frag`) combina:
- Normal mapping + roughness PBR de grama esparsa (`sparse_grass`), com UV
  mapeado em espaço de mundo (`v_frag_pos.xz * 0.08`) para eliminar
  repetição visível de ladrilhos, independentemente da densidade da grade;
- **Rachaduras sísmicas**: um padrão de Voronoi (`voronoi_edge`, calculado no
  próprio shader, idêntico ao usado em `ground.vert` e em `scene.frag`) cuja
  intensidade de exposição de "solo" vs. "fissura profunda" depende de
  `u_crack_intensity` (função da maior magnitude já disparada) e da
  distância ao epicentro;
- **Oclusão de contato**: até 40 pontos (posições X/Z de prédios/casas
  próximos, arbitrariamente limitado pelo array do shader) escurecem o chão
  ao redor da base dos objetos (`contact_ao()`), simulando sombra de contato
  sem qualquer AO real calculado;
- PCF 3×3 (soma de 9 amostras da shadow map) para suavizar a borda da sombra
  direcional.

### 7.3 `building.py` — `Building`, `create_house`, `BuildingDebris`

O modelo de colapso é o núcleo "de gameplay" do projeto:

- Cada prédio tem uma **resistência aleatória** (`RESISTANCE_RANGE`) e
  acumula `damage` proporcional à amplitude do deslocamento sísmico local
  (`amplitude * dt * DAMAGE_MULTIPLIER`) sempre que essa amplitude excede um
  limiar (`0.15`). Ao atingir a resistência, `collapsing = True` e um
  temporizador de 2s (`COLLAPSE_DURATION`) avança `collapse_progress`;
- O edifício é desenhado como uma pilha de **fatias horizontais** (`slices`),
  cada uma sua própria instância do cubo compartilhado com uma matriz de
  modelo própria. Conforme `collapse_progress` avança, cada fatia "cai" com
  atraso proporcional à sua altura (`piece_fall`, dependente de `t`), gerando
  o efeito de colapso progressivo de cima para baixo com inclinação
  (`lean_angle_x/z`) e afundamento da base — é um efeito inteiramente
  procedural/artístico, **não uma simulação física real** (sem
  rigidez/momento/colisão entre fatias);
- Ao ultrapassar 12% do progresso de colapso, `_spawn_debris()` cria de 60 a
  90 `BuildingDebris` (cubos pequenos com física de queda livre simplificada
  — gravidade constante, quique amortecido no chão, rotação constante) e o
  `ParticleSystem` recebe uma rajada de poeira. Cada `BuildingDebris` só
  expõe `instance_data()` (matriz de modelo + cor) — quem desenha todos os
  escombros da cena é o `DebrisRenderer` centralizado (Seção 7.8), não mais
  um `draw()` por peça;
- Textura tem **crossfade de dano**: os shaders recebem simultaneamente o
  material intacto e o danificado (`u_damage_blend` interpola albedo/normal/
  roughness entre eles), então o prédio "envelhece" visualmente antes mesmo
  de começar a desabar, e passa a usar a textura de concreto rachado assim
  que `collapsing = True`;
- Casas (`create_house`) são o mesmo `Building` com `is_house=True`,
  dimensões menores, paleta de tijolo e um telhado de duas águas (mesh
  "gable" compartilhado) + chaminé desenhados só quando intactas.

### 7.4 `street.py`, `light_pole.py` — mobiliário urbano

- `Street` é só um cubo achatado texturizado com asfalto PBR, sem física
  própria — mas o `scene.frag` usa `u_is_street` para desenhar as mesmas
  rachaduras de Voronoi do chão diretamente no asfalto;
- `LightPole`/`LampPost` (alias) tem um modelo de queda por probabilidade:
  acima de um limiar de amplitude sísmica local, a cada frame há uma chance
  de começar a cair (`falling=True`), então gira em torno do pivô da base até
  um ângulo alvo aleatório (`target_tilt`), com a lâmpada "piscando"
  aleatoriamente enquanto cai e emitindo poeira de impacto ao tocar o chão.
  Quando não está caindo mas o terremoto está ativo, balança (`sway`) com
  uma senoide simples baseada no tempo — uma aproximação visual, não a mesma
  fórmula do `earthquake.py`.

### 7.5 `mountain.py` — `Mountain`, `RockDebris`

Malha pré-computada **uma única vez no `__init__`** (não recalculada por
frame): um cone segmentado em `bands` × `slices`, com o raio de cada anel
perturbado por `perlin2d` para parecer rochoso, e a normal aproximada
analiticamente (`[cos, 0.35, sin]`, não recomputada a partir da malha real
perturbada — uma simplificação intencional). O bioma nevado no topo é feito
inteiramente no fragment shader (`scene.frag`, seção "Montanha rochosa"), não
na geometria/textura. Sob abalo intenso, gera `RockDebris` com física
simplificada similar aos escombros de prédio — e, como `BuildingDebris`, só
expõe `instance_data()` para o `DebrisRenderer` (Seção 7.8) em vez de
desenhar a si mesmo.

### 7.6 `nature.py` — `Tree`, `generate_forest`

Árvore procedural (tronco cilíndrico + camadas de cone para a copa,
compartilhando meshes globais). O modelo de queda é probabilístico
(similar ao poste): acima de um limiar de amplitude, chance de iniciar
queda, com ângulo de queda crescendo com `t²` (progressão acelerada) e, ao
concluir, desenha um "buraco" de terra revolvida sob a base usando o material
`dry_river_pebbles`. `generate_forest()` faz *rejection sampling* em anel
(distância `radius_range` do centro), testando cada candidato contra zonas
circulares de exclusão, retângulos de prédios/casas e faixas de ruas antes de
aceitar a posição — sem otimização espacial (grid/quadtree), mas com apenas
~120 árvores e algumas dezenas de obstáculos isso é O(n·m) irrelevante em
custo.

### 7.7 `village.py` — `generate_village`, `generate_city`

Gera proceduralmente: um núcleo de prédios em grade quadrada (linhas/colunas
derivadas de `sqrt(building_count)`), anéis concêntricos de casas ao redor
(posições distribuídas por ângulo com jitter aleatório), uma malha de ruas
ortogonais espaçadas por `block_spacing + street_width`, e postes de
iluminação distribuídos ao longo de cada rua alternando o lado da calçada.
`generate_city()` é mantida só por "compatibilidade retroativa" e não é usada
por `main.py` (ver Seção 8).

### 7.8 `debris_renderer.py` — `DebrisRenderer`

Adicionado na otimização de performance da Seção 12: junta todo o escombro
vivo da cena (de todos os prédios/casas colapsados, mais as rochas soltas da
montanha) num único VBO de instância e desenha tudo em **1–2 draw calls**,
em vez de um `glDrawArrays` por peça. Segue exatamente o padrão já usado por
`ParticleSystem` (Seção 5.2): `glDrawArraysInstanced` +
`glVertexAttribDivisor`, só que aqui a malha instanciada é um cubo (não um
quad-billboard) e o dado por instância é a **matriz de modelo 4x4 completa**
(4 atributos `vec4`, locations 4–7 — a técnica padrão de "4 colunas" para
instancing) mais uma cor `vec4` (location 8), em vez de posição/tamanho/
alpha/cor separados.

- `collect(buildings)` / `collect_rocks(mountain)`: percorrem as listas de
  debris existentes e empacotam `(model_matrix, color)` de cada peça viva
  num array NumPy contíguo (chamado uma vez por frame em `main.py`, antes da
  renderização);
- `draw(...)`: escombro de prédio usa um único material PBR compartilhado
  (`cracked_concrete_02` — todo `Building.collapse_set` já apontava para o
  mesmo material antes desta mudança, então isso não altera a aparência) e é
  desenhado com textura; rochas não têm textura, só cor por instância;
- `draw_shadow(...)`: mesma ideia com um shader dedicado mais simples
  (`debris_shadow.vert` + o `shadow.frag` já existente) para o passe de
  profundidade;
- A malha de cubo e os shaders são próprios do `DebrisRenderer` (não
  reaproveita o cubo compartilhado de `shared.py`), pelo mesmo motivo que o
  `ParticleSystem` também tem seu próprio quad: os atributos de instância
  extras (locations 4–8) são específicos desse VAO e não podem conviver com
  o cubo compartilhado, que é desenhado sem instancing em várias outras
  centenas de chamadas (prédios, postes, ruas...).

---

## 8. Shaders GLSL — resumo técnico

| Par | Papel |
|---|---|
| `scene.vert/frag` | Shader "genérico" reutilizado por ruas, postes, prédios, montanha e árvores. Faz Blinn-Phong com PBR (normal+roughness maps), crossfade de dano (`u_damage_blend`), rachaduras de Voronoi na rua, neve por altura na montanha, folhagem colorida por altura/ruído, e até janelas/portas "processuais" na fachada dos prédios (padrão repetido calculado a partir de `v_frag_pos`, sem UV de fachada dedicado). |
| `ground.vert/frag` | Deformação de vértice pela equação de onda sísmica + recálculo analítico de normal; rachaduras de Voronoi; PBR de grama; AO de contato por lista de pontos. |
| `billboard.vert/frag` | Billboarding esférico instanciado para partículas; alpha vem da textura de gradiente radial × alpha da partícula. |
| `shadow.vert/frag` | Passe de profundidade puro para o shadow mapping direcional (fragment shader vazio). |
| `sky.vert/frag` | Skybox equirretangular via des-projeção do far-plane; sem depth test. |
| `hud.vert/frag` | Quad ortográfico simples multiplicando textura × cor uniforme — usado tanto para texto (via `TextTexture`) quanto para os painéis sólidos do HUD. |
| `debris.vert/frag` | Versão instanciada do shader de cena para escombros (Seção 7.8): cada instância traz sua matriz de modelo 4x4 (4 `vec4`, locations 4–7) e cor (location 8) em vez de um `u_model` por draw call. A iluminação Blinn-Phong replica exatamente `scene.frag` (mesmo `spec_power`, damping e AO de contato por altura) para não introduzir diferença visual. |
| `debris_shadow.vert` | Mesma ideia instanciada, só para o passe de profundidade dos escombros (usa o `shadow.frag` vazio já existente). |

Ponto notável: a função `voronoi_edge`/`random2` para as rachaduras está
**duplicada literalmente** em `ground.vert`, `ground.frag` e `scene.frag`
(com o nome renomeado para `_scene` nesse último para evitar colisão, já que
não há um sistema de `#include` em GLSL puro). Funciona, mas qualquer ajuste
na fórmula do rachado precisa ser replicado nos três lugares manualmente.

---

## 9. `tools/convert_exr_textures.py`

Script standalone (fora do pacote `src`) que varre `assets/textures/pbr/`
por arquivos `.exr`, decodifica (via OpenCV se disponível, com fallback para
`OpenEXR`+`Imath`), converte para 8 bits e redimensiona para 2048×2048 por
padrão (`--full-4k` mantém a resolução original). Existe porque texturas PBR
de fontes como Poly Haven costumam vir em EXR de alta profundidade de bits, e
o motor de runtime só consome PNG/JPG via PIL (`texture.py` não lê `.exr`
diretamente).

---

## 10. Melhorias e falhas identificadas

### 10.1 Portabilidade quebrada por padrão (✅ corrigido na Seção 12)

`main.py:16-24` forçava incondicionalmente, em **qualquer** Linux:

```python
if sys.platform.startswith("linux"):
    os.environ["GALLIUM_DRIVER"] = "d3d12"
    os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"] = "NVIDIA"
```

O comentário e o README afirmavam que isso "detecta e ativa automaticamente" a
GPU dedicada, mas não havia nenhuma checagem real de que o processo estava
rodando sob WSL. Em Linux nativo, forçar o backend D3D12 do Mesa podia fazer o
OpenGL falhar ao inicializar ou usar um backend pior do que o driver nativo
escolheria sozinho. **Corrigido**: `_running_under_wsl()` agora detecta WSL de
verdade antes de setar essas variáveis (Seção 3.1). A investigação que levou a
essa correção (Seção 12) também revelou um problema relacionado, mais sutil,
em notebooks híbridos Intel+NVIDIA sob Wayland — ver Seção 12.1.

### 10.2 Docstring desalinhada com o código (severidade: baixa, mas engana leitor)

`camera.py:1-10` e o `__init__` (linhas 41-44) descrevem "Rotação via mouse
suave com interpolação exponencial", com estado dedicado
(`_smooth_yaw_vel`, `_smooth_pitch_vel`, `smooth_alpha`). Mas `process_mouse`
(linha 111 em diante) aplica o delta bruto do mouse diretamente ao
yaw/pitch, e o próprio comentário no método diz "sem acúmulo de
velocidade/inércia". Esses três atributos nunca são lidos em lugar nenhum —
são estado morto. **Sugestão**: remover os campos não usados ou implementar
de fato a suavização exponencial que a docstring promete.

### 10.3 Cleanup assimétrico de recursos GPU (severidade: baixa)

`main.py` (linhas 294-305) libera `ground`, `mountain`, `particle_system`,
`hud`, `shadow_map`, `sky`, `scene_shader` e as texturas soltas
(`cleanup_textures()`), mas nunca libera:
- os meshes compartilhados de `world/shared.py` (`_shared_cube_mesh`,
  `_shared_cone_mesh`, `_shared_cylinder_mesh`, `_shared_pyramid_mesh`,
  `_shared_gable_mesh`);
- os IDs cacheados em `_pbr_set_cache`, `_flat_normal_id` e
  `_default_roughness_id` de `core/texture.py` (o `cleanup_textures()`
  só esvazia `_loaded_textures`, uma lista separada).

Em um processo que só roda uma vez e termina (`sys.exit()` logo depois), o
sistema operacional recupera a VRAM de qualquer forma — não é um bug visível
hoje. Mas é um sinal de que os módulos de cache global não têm uma função
`reset()`/`cleanup()` simétrica à de alocação, o que se tornaria um problema
real se o jogo precisasse reiniciar dentro do mesmo processo (testes
automatizados, hot-reload, um menu de "nova simulação" sem reiniciar o
executável). **Sugestão**: adicionar `reset_shared_resources()` em
`shared.py` e `texture.py`, chamado no cleanup de `main.py`.

### 10.4 Checagem morta em `main.py:296-298`

```python
for b in all_buildings:
    if hasattr(b, "cleanup"):
        b.cleanup()
```

`Building` nunca define um método `cleanup()` (ele só usa meshes
compartilhados, então de fato não precisa de um) — logo essa condição nunca é
verdadeira e o bloco é código morto. O comentário ao lado ("desnecessário
para ruas") também confunde, pois o loop é sobre prédios, não ruas.
**Sugestão**: remover o loop e o comentário, ou documentar explicitamente
"Building não possui recursos GPU próprios — nada a limpar aqui" sem o
`hasattr` supérfluo.

### 10.5 Nenhum buffer de índices (EBO) em toda a geometria

`mesh.py`/`obj_loader.py` só usam `glDrawArrays` — toda malha (incluindo o
cubo compartilhado por centenas de instâncias de prédios/postes/casas e a
grade do chão com até 220×220 células) armazena vértices duplicados por
triângulo, sem `glDrawElements` + Element Buffer Object. Para geometria tão
pequena e reaproveitada (poucos VAOs no total, já que tudo compartilha
meshes de `shared.py`), o impacto prático é baixo, mas é uma otimização
padrão de baixo custo de implementação (o cubo, por exemplo, tem 36 vértices
em vez dos 8 que um índice permitiria). **Sugestão**: não é urgente, mas vale
considerar para o `create_plane_mesh` do chão, que é a malha com mais
vértices duplicados do projeto (2 triângulos × 3 vértices por célula, sem
compartilhar os vértices entre células vizinhas).

### 10.6 Loop Python não vetorizado no sistema de partículas

`particles.py:193-195`:

```python
instance_data = np.empty((len(self.particles), 8), dtype=np.float32)
for i, p in enumerate(self.particles):
    instance_data[i] = (p.x, p.y, p.z, p.size, p.alpha, *p.color)
```

É a única parte do sistema de partículas que não está vetorizada — com até
6000 partículas simultâneas (limite configurado em `main.py`), esse laço
Python roda a cada frame. Como cada `Particle` já usa `__slots__` e o próprio
`update()` de cada partícula também é um laço Python por partícula
(`particles.py:143-145`), o gargalo de CPU em cenas de colapso maciço
provavelmente está mais nesses updates do que na montagem do buffer — mas
ambos poderiam ser convertidos para arrays estruturados NumPy (SoA em vez de
lista de objetos) se o profiling mostrar que isso limita o FPS.
**Sugestão**: só otimizar se houver medição real de gargalo — é fácil
prematuramente complicar esse sistema, que hoje é legível.

### 10.7 Fórmula da onda sísmica duplicada em três lugares

A física do terremoto (`amplitude = magnitude * exp(-damping*t) *
exp(-falloff*dist) * sin(2π*freq*t)`) existe:
1. Em Python, `EarthquakeSimulator.get_offset()` — usada por
   prédios/árvores/postes/montanha/câmera (indiretamente, via objetos);
2. Em GLSL, `ground.vert` — para deformar o chão diretamente na GPU (correto
   que exista aqui, pois rodar isso por vértice em Python seria inviável);
3. Aproximada de forma **diferente e mais simples** em `light_pole.py`
   (`sway_ang = amp * 18.0 * sin(current_time * 16.0 + self.x * 0.5)`) e
   implicitamente na leitura de amplitude em `mountain.py`/`nature.py` (que
   reusam `get_offset` mas decidem seus próprios limiares/probabilidades de
   reação).

Não há bug funcional aqui — cada consumidor usa a mesma fonte de amplitude
(`get_offset`) exceto o balanço do poste, que é deliberadamente uma
aproximação visual mais barata. O risco é de manutenção: mudar `damping`,
`frequency` ou `spatial_falloff` em `EarthquakeSimulator.__init__` sincroniza
Python e chão automaticamente (o chão lê os mesmos atributos via uniforms),
mas **não** afeta o "sway" do poste, que tem sua própria constante `18.0`
hardcoded. **Sugestão**: nada crítico, mas vale um comentário explícito em
`light_pole.py` deixando claro que aquele sway é intencionalmente
independente da física real, para não confundir o próximo mantenedor.

### 10.8 Ausência de testes automatizados e CI

Não existe pasta `tests/`, nenhum framework de teste nas dependências, nem
workflow de CI (`.github/workflows/`) no repositório. Funções puras e fáceis
de testar sem contexto OpenGL já existem prontas para isso — `math_utils.py`
inteiro (matrizes, Perlin) e `earthquake.py.get_offset` não tocam GPU/Pygame
e poderiam ter testes unitários simples (ex.: `perspective()` produz a matriz
esperada para parâmetros conhecidos; `get_offset` retorna zero antes da
chegada da onda; `Building.update` acumula dano e dispara colapso ao exceder
a resistência). **Sugestão**: começar por esses três módulos, que são puros
e não exigem um contexto OpenGL mockado.

### 10.9 Sem tratamento de redimensionamento de janela

`ASPECT_RATIO` é calculado uma única vez a partir de `WINDOW_SIZE`
(`main.py:52`) e nunca recalculado; não há handler para
`pygame.VIDEORESIZE`. Se a janela puder ser redimensionada pela plataforma
(depende do driver/gerenciador de janelas — o modo atual não pede
`RESIZABLE` explicitamente, então pode nem ser reproduzível em todo SO), a
projeção perspectiva ficaria com proporção incorreta. **Sugestão**: baixa
prioridade dado que a janela não é criada com a flag `RESIZABLE`; documentar
essa limitação ou tratar o evento se resize for um requisito futuro.

### 10.10 `generate_city()` morta e com assinatura confusa

`village.py:81-83`:

```python
def generate_city(rows=5, cols=5, spacing=5.5):
    """Compatibilidade retroativa: gera cidade simples."""
    return generate_village(center=(0.0, 0.0), building_count=rows * cols)[0]
```

`spacing` é aceito mas nunca repassado a `generate_village` (que usa seu
próprio `block_spacing` padrão), e `cols`/`rows` só existem para calcular
`building_count` — a função ignora completamente a intenção de "grade
rows×cols" que seu nome sugere. Ela não é chamada por `main.py` nem por
nenhum outro módulo lido nesta análise. **Sugestão**: remover se
"compatibilidade retroativa" não tem mais nenhum consumidor real, ou corrigir
a assinatura para de fato respeitar `spacing`.

### 10.11 Fallback procedural silencioso pode mascarar assets ausentes

`core/texture.py.load_texture()`: quando um caminho de textura não existe,
gera uma textura procedural parecida e só imprime um aviso no console
(`print(f"[Aviso] ...")`), sem falhar. Combinado com `load_texture_set` (que
também cai para procedural se não achar `_diff_`/`_nor_gl_`/`_rough_`), é
fácil um pacote PBR inteiro estar ausente ou mal referenciado e o jogo
simplesmente rodar com aparência degradada, sem nenhum sinal visual óbvio de
que algo está "errado" — só o log de console, que passa despercebido fora de
um terminal aberto. **Sugestão**: manter o fallback (é uma decisão de design
razoável para robustez), mas considerar elevar o nível do aviso (ex.: cor no
terminal, ou um contador de fallbacks usados exibido no HUD/log de saída) já
que hoje é fácil não perceber.

### 10.12 Sem seed determinística para geração procedural

`random` (módulo padrão, não `random.Random` instanciado) é usado sem seed
em `village.py`, `nature.py`, `building.py`, `light_pole.py` e
`mountain.py`. Cada execução gera uma vila/floresta diferente e um padrão de
queda de postes/árvores diferente. Isso provavelmente é intencional
(rejogabilidade — cada sessão parece nova), mas tem um custo: não há como
reproduzir exatamente um bug relatado ("nessa configuração específica de
vila, o prédio X atravessa o poste Y") nem escrever testes determinísticos
sobre o layout gerado. **Sugestão**: se rejogabilidade for o objetivo,
considerar aceitar uma seed opcional (`SEISMICPYGL_SEED` como env var, no
mesmo estilo de `SEISMICPYGL_RESOLUTION`) para permitir reprodução quando
necessário, sem tirar a aleatoriedade por padrão.

### 10.13 Duplicação de shader Voronoi entre três arquivos GLSL

`voronoi_edge`/`random2` (rachaduras sísmicas) estão implementadas de forma
idêntica em `ground.vert`, `ground.frag` e `scene.frag` (renomeada para
`_scene` nesse último). GLSL não tem um sistema de módulos/`#include`
nativo no OpenGL "puro" (só via extensões ou pré-processamento manual do lado
do Python antes de `ShaderProgram.from_files`), então isso é uma limitação
conhecida da linguagem, não necessariamente um erro de design. Mas hoje,
qualquer ajuste na fórmula de rachadura precisa ser replicado manualmente nos
três lugares — já aconteceria uma divergência sutil se alguém alterasse só
um. **Sugestão**: se o projeto crescer, implementar um pré-processador simples
em `shader.py` que faça `#include "voronoi.glsl"` por concatenação de string
antes de compilar, centralizando essa função em um único arquivo fonte.

### 10.14 Assets binários pesados versionados diretamente no Git

O repositório inclui texturas PBR em 4K (JPG/PNG) e HDRIs diretamente no
histórico do Git (não há indício de Git LFS configurado — sem `.gitattributes`
apontando para LFS). Isso é uma observação de higiene de repositório, não um
bug: aumenta consideravelmente o tamanho do clone e do histórico ao longo do
tempo conforme texturas são trocadas/adicionadas. **Sugestão**: considerar
Git LFS ou um passo de download de assets separado do controle de versão,
caso o repositório continue crescendo.

---

## 11. Resumo executivo

O projeto está bem estruturado para o escopo de um simulador didático:
separação clara core/simulation/rendering/world, reaproveitamento de meshes e
materiais via caches globais, uma única fórmula de física sísmica reutilizada
de forma consistente entre CPU e GPU, e efeitos visuais (screen shake por
Perlin, colapso "fatiado" de prédios, shadow mapping, rachaduras por Voronoi)
implementados de forma direta e sem dependências externas além do
essencial (Pygame, PyOpenGL, NumPy, Pillow).

As falhas encontradas são majoritariamente de **manutenibilidade a longo
prazo** (falta de testes, cleanup assimétrico, duplicação de shader) — nada
que afete a experiência de quem já roda o projeto no ambiente para o qual ele
foi construído. O ponto de maior impacto real em uso, o hack de driver
forçado cegamente em todo Linux (antigo item 10.1), foi corrigido; a Seção 12
detalha essa correção e o trabalho de otimização de performance que ela
motivou.

---

## 12. Investigação e correção de performance (2026-09-08)

Motivada por um relato de desempenho ruim mesmo com GPU dedicada disponível.

### 12.1 GPU errada em uso (achado além do escopo original)

Além do hack de WSL (Seção 10.1) já estar incorreto por si só, o diagnóstico
revelou algo mais sutil nesta máquina (notebook híbrido Intel + NVIDIA,
Hyprland/Omarchy sob Wayland): mesmo com `__GLX_VENDOR_LIBRARY_NAME=nvidia`
configurado globalmente pelo compositor (o mecanismo padrão do GLVND/PRIME
render offload para apontar para a GPU dedicada), o SDL2 por padrão criava o
contexto OpenGL pelo caminho **Wayland/EGL nativo**, que não consulta essa
variável — então o jogo rodava silenciosamente na GPU integrada (confirmado
lendo a linha `[Hardware 3D] GPU:` que `main.py` já imprimia no console).
Forçar `SDL_VIDEODRIVER=x11` (caminho XWayland/GLX) fez o SDL respeitar a
seleção de GPU do sistema e escolher a GPU dedicada — mas essa mesma mudança
quebrou a criação do contexto OpenGL de forma reproduzível no ambiente de
teste usado (`OpenGL.error.Error: Attempt to retrieve context when no valid
context`, na primeira chamada de `glVertexAttribPointer`), por uma causa não
totalmente identificada. Por isso a mudança **não é o padrão**: virou opt-in
via `SEISMICPYGL_FORCE_X11=1 python main.py` (Seção 3.1), com o trade-off
documentado no próprio código.

### 12.2 Gargalo estrutural de CPU/Python (causa principal)

O renderizador é inteiramente *immediate-mode*: cada objeto emite sua própria
sequência de `glActiveTexture`/`glBindTexture`/`glUniform*`/`glDrawArrays`
via PyOpenGL/ctypes, cujo overhead por chamada em Python é bem maior que em
C/C++. Dois pontos concentravam a maior parte disso:

- **Escombros de colapso nunca eram instanciados**: cada `BuildingDebris`/
  `RockDebris` (até 60–90 por prédio colapsado) desenhava a si mesmo
  individualmente. Num terremoto forte derrubando vários prédios, isso virava
  milhares de draw calls extras no mesmo frame — o pior caso de desempenho do
  jogo é justamente seu recurso central (o colapso durante um terremoto).
  **Corrigido**: `DebrisRenderer` (Seção 7.8) desenha todo o escombro vivo em
  1–2 draw calls instanciados, seguindo o padrão que `ParticleSystem` já
  usava.
- **Passe de sombra desenhava texturas desnecessariamente**: `shadow.frag` é
  vazio (não lê nenhuma textura), mas todo `draw()` de prédio/rua/poste/
  árvore/montanha sempre executava o bind completo de texturas PBR mesmo
  quando quem desenhava era o shader de sombra. **Corrigido**: todos os
  `draw()` de `src/world/{building,street,light_pole,mountain,nature}.py`
  agora aceitam `shadow_pass: bool`, pulando esse bloco quando `True`.

Complementarmente, foi adicionado um **culling barato por ângulo de visão**
(`_in_view` em `main.py`) para árvores e postes de iluminação (~250 objetos)
no passe principal — objetos claramente fora do campo de visão da câmera não
são desenhados (o passe de sombra continua desenhando tudo, para a sombra
ficar correta).

### 12.3 Verificação

Testado disparando terremotos magnitude 8.5 repetidos até derrubar todos os
prédios/casas (via `wtype` simulando teclas), sem erros/tracebacks, com
screenshots (`grim`) confirmando visualmente que sombras, rachaduras, poeira
e os escombros instanciados renderizam corretamente e com a mesma iluminação
de antes (o shader `debris.frag` replica deliberadamente a fórmula de
Blinn-Phong de `scene.frag`, incluindo o AO de contato por altura, para não
introduzir diferença visual). Uma comparação numérica de FPS antes/depois
limpa não foi possível de automatizar de forma confiável neste ambiente
específico (a IDE aberta às vezes roubava o foco da janela do jogo); a
correção se apoia na redução estrutural de chamadas O(n) → O(1) por
escombro, que é a explicação mais direta para o gargalo relatado.

### 12.4 Não implementado (possível trabalho futuro)

Postes de iluminação (~130) e árvores (~120) ainda desenham cada sub-parte
individualmente (cilindro do poste, cubo do braço/cabeça/lâmpada; tronco +
camadas de copa). O mesmo padrão de instancing do `DebrisRenderer` poderia
ser aplicado a eles — maior ganho estrutural ainda, mas escopo maior porque
cada sub-parte tem um transform dinâmico próprio (queda, balanço).
