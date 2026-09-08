import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# ShaderProgram.from_files/HUD/ShadowMap/ParticleSystem etc. resolve
# assets/... relative to the current working directory.
os.chdir(REPO_ROOT)


@pytest.fixture(scope="session")
def gl_context():
    """
    Janela OpenGL real (escopo de sessão, uma só para todos os testes de
    fumaça) — mesmo padrão (pygame.OPENGL|DOUBLEBUF, driver nativo, sem
    forçar SDL_VIDEODRIVER=x11) usado nos scripts ad hoc desta sessão, que
    se mostrou estável neste ambiente. Se não houver display disponível
    (ex. CI headless futuro), os testes que dependem dela são pulados em
    vez de derrubar a suíte inteira.
    """
    import pygame

    pygame.init()
    try:
        pygame.display.set_mode((320, 240), pygame.OPENGL | pygame.DOUBLEBUF)
    except Exception as exc:
        pygame.quit()
        pytest.skip(f"Sem contexto OpenGL disponível neste ambiente: {exc}")
    yield
    pygame.quit()
