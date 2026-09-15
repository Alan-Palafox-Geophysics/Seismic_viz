"""
exportar_pdf.py
===============

Convierte un documento Markdown a PDF conservando formato y estilo.

La conversión es en dos etapas: Markdown → HTML con una hoja de estilo de
impresión → PDF mediante el motor de maquetación de un navegador basado en
Chromium, que es el que resuelve saltos de página, encabezados de tabla
repetidos y numeración.

Identidad corporativa:
    assets/logo_encabezado.png   logotipo repetido en el margen superior
    assets/marca_agua.png        marca de agua centrada, tenue, bajo el texto

Ambas se incrustan en el HTML como data URI, de modo que el archivo
intermedio es autocontenido y conserva el logotipo aunque se mueva de
carpeta o se imprima desde otro equipo.

Uso::

    python exportar_pdf.py DOCUMENTACION.md
    python exportar_pdf.py DOCUMENTACION.md salida.pdf
    python exportar_pdf.py DOCUMENTACION.md --proteger

Dependencias:
    markdown            pip install markdown
    playwright          pip install playwright && playwright install chromium
                        (necesario para el logotipo repetido en el margen)
    pypdf               pip install pypdf   (solo para --proteger)

Sin Playwright se recurre a Chromium por línea de comandos y el logotipo
aparece como membrete de la portada en lugar de repetirse: los elementos
CSS `position: fixed` solo se anclan dentro del área de contenido, donde
colisionarían con el texto.

Si no encuentra un navegador, deja el HTML intermedio junto al Markdown para
que pueda imprimirse a PDF manualmente desde el navegador (Ctrl+P → Guardar
como PDF), con idéntico resultado.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Identidad corporativa
# ---------------------------------------------------------------------------
DIR_ASSETS = Path(__file__).resolve().parent / "assets"
LOGO_ENCABEZADO = DIR_ASSETS / "logo_encabezado.png"
MARCA_AGUA = DIR_ASSETS / "marca_agua.png"

# --- Ajustes de identidad --------------------------------------------------
# Membrete de portada (logotipo grande, en flujo, primera página).
ALTO_MEMBRETE_MM = 22
# Banda de pie repetida en todas las páginas.
ALTO_PIE_MM = 8
# Marca de agua: a mayor tamaño cubre más superficie y es más difícil de
# recortar; a mayor transparencia estorba menos la lectura, pero también
# protege menos.  Entre 0.04 y 0.10 es el rango utilizable.
ANCHO_MARCA_PCT = 96
OPACIDAD_MARCA = 0.05
ROTACION_MARCA_DEG = -30


def _data_uri(ruta: Path) -> str | None:
    """Incrusta una imagen como data URI para que el HTML sea autocontenido."""
    if not ruta.exists():
        return None
    tipo = mimetypes.guess_type(ruta.name)[0] or "image/png"
    datos = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return f"data:{tipo};base64,{datos}"

# ---------------------------------------------------------------------------
# Hoja de estilo: documento técnico-ejecutivo
# ---------------------------------------------------------------------------
CSS = """
/* Estos márgenes rigen el motor de respaldo; con Playwright los fija la
   llamada de impresión, que además reserva sitio para el encabezado. */
@page {
    size: A4;
    margin: 26mm 18mm 20mm 18mm;
}

:root {
    --tinta:    #1a1d23;
    --suave:    #5b6472;
    --linea:    #d8dce2;
    --acento:   #1f5673;
    --fondo-alt:#f4f6f8;
}

* { box-sizing: border-box; }

body {
    font-family: "Charter", "Georgia", "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: var(--tinta);
    margin: 0;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

/* --- Jerarquía ------------------------------------------------------- */
h1, h2, h3, h4 {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    color: var(--acento);
    line-height: 1.25;
    page-break-after: avoid;
}

h1 {
    font-size: 23pt;
    margin: 0 0 0.2em;
    padding-bottom: 0.35em;
    border-bottom: 3px solid var(--acento);
    letter-spacing: -0.4px;
}

h2 {
    font-size: 15pt;
    margin: 1.9em 0 0.7em;
    padding-bottom: 0.25em;
    border-bottom: 1px solid var(--linea);
    page-break-before: auto;
}

/* Cada capítulo numerado empieza en página nueva */
h2[id^="1-"], h2[id^="2-"], h2[id^="3-"], h2[id^="4-"], h2[id^="5-"],
h2[id^="6-"], h2[id^="7-"], h2[id^="8-"], h2[id^="9-"], h2[id^="10-"],
h2[id^="11-"], h2[id^="12-"], h2[id^="apendice"] {
    page-break-before: always;
}

h3 { font-size: 12pt; margin: 1.5em 0 0.5em; color: var(--tinta); }
h4 { font-size: 10.5pt; margin: 1.2em 0 0.4em; color: var(--suave);
     text-transform: uppercase; letter-spacing: 0.6px; }

p { margin: 0 0 0.75em; text-align: justify; hyphens: auto; }

/* Subtítulo de portada */
h1 + h2 {
    border: none;
    color: var(--suave);
    font-size: 13pt;
    font-weight: 500;
    margin-top: 0.4em;
    page-break-before: avoid;
}
h1 + h2 + p { color: var(--suave); font-size: 9.5pt; font-style: italic; }

/* --- Tablas ---------------------------------------------------------- */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0 1.3em;
    font-size: 9pt;
    page-break-inside: avoid;
}
thead { display: table-header-group; }
th {
    background: var(--acento);
    color: #fff;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-weight: 600;
    text-align: left;
    padding: 6px 9px;
    font-size: 8.5pt;
    letter-spacing: 0.2px;
}
td { padding: 5px 9px; border-bottom: 1px solid var(--linea); vertical-align: top; }
tbody tr:nth-child(even) { background: var(--fondo-alt); }

/* --- Código y ecuaciones --------------------------------------------- */
code {
    font-family: "SF Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 8.8pt;
    background: var(--fondo-alt);
    padding: 1px 4px;
    border-radius: 3px;
    color: #9a3412;
}
pre {
    background: var(--fondo-alt);
    border-left: 3px solid var(--acento);
    padding: 11px 14px;
    margin: 1em 0 1.2em;
    overflow-x: auto;
    page-break-inside: avoid;
    border-radius: 0 4px 4px 0;
}
pre code {
    background: none;
    padding: 0;
    color: var(--tinta);
    font-size: 9pt;
    line-height: 1.5;
}

/* --- Citas destacadas ------------------------------------------------ */
blockquote {
    margin: 1.2em 0;
    padding: 0.8em 1.1em;
    background: #eef4f8;
    border-left: 4px solid var(--acento);
    page-break-inside: avoid;
    border-radius: 0 4px 4px 0;
}
blockquote p { margin: 0 0 0.5em; }
blockquote p:last-child { margin-bottom: 0; }

/* --- Listas y separadores -------------------------------------------- */
ul, ol { margin: 0 0 0.9em; padding-left: 1.5em; }
li { margin-bottom: 0.3em; }

hr { border: none; border-top: 1px solid var(--linea); margin: 2em 0; }

strong { color: #000; font-weight: 600; }
em { color: var(--suave); }

/* --- Identidad corporativa ------------------------------------------- */
/* En la impresión de Chromium los elementos `position: fixed` se repiten en
   todas las páginas, pero solo admiten anclaje dentro del área de contenido:
   cualquier desplazamiento negativo hacia el margen los reubica al pie, y ahí
   chocan con el texto.  Por eso el logotipo repetido no se resuelve con CSS
   sino con la plantilla de encabezado nativa del motor de impresión, que sí
   dibuja dentro del margen.  Aquí solo queda el membrete de portada. */

.membrete {
    text-align: right;
    margin: 0 0 6mm;
    page-break-after: avoid;
}
.membrete img { height: MEMBRETE_MMmm; width: auto; }

.marca-agua {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(ROTACION_DEGdeg);
    width: MARK_PCT%;
    opacity: MARK_OPACITY;
    z-index: 0;
    pointer-events: none;
}

/* El contenido va por encima de la marca de agua */
body > *:not(.marca-agua) {
    position: relative;
    z-index: 1;
}
"""

EXTENSIONES = ["tables", "fenced_code", "toc", "attr_list", "sane_lists"]


def markdown_a_html(texto: str, titulo: str, con_membrete: bool = True) -> str:
    try:
        import markdown
    except ImportError:
        print("Falta el paquete 'markdown'.  Instálelo con:\n\n    pip install markdown\n")
        raise SystemExit(1)

    cuerpo = markdown.markdown(texto, extensions=EXTENSIONES)

    logo = _data_uri(LOGO_ENCABEZADO)
    marca = _data_uri(MARCA_AGUA)

    hoja = (
        CSS.replace("MEMBRETE_MM", str(ALTO_MEMBRETE_MM))
        .replace("PIE_MM", str(ALTO_PIE_MM))
        .replace("MARK_PCT", str(ANCHO_MARCA_PCT))
        .replace("MARK_OPACITY", str(OPACIDAD_MARCA))
        .replace("ROTACION_DEG", str(ROTACION_MARCA_DEG))
    )

    capas = ""
    if marca:
        capas += f"<img class='marca-agua' src='{marca}' alt=''>\n"
    if logo and con_membrete:
        capas += f"<div class='membrete'><img src='{logo}' alt='TLALLI'></div>\n"

    return (
        "<!DOCTYPE html>\n<html lang='es'>\n<head>\n<meta charset='utf-8'>\n"
        f"<title>{titulo}</title>\n<style>{hoja}</style>\n</head>\n<body>\n"
        f"{capas}{cuerpo}\n</body>\n</html>\n"
    )


def _buscar_navegador() -> str | None:
    candidatos = [
        os.environ.get("CHROME_BIN"),
        "/opt/pw-browsers/chromium",
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "msedge",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidatos:
        if not c:
            continue
        ruta = shutil.which(c) or (c if Path(c).exists() else None)
        if ruta:
            return ruta
    return None


PLANTILLA_ENCABEZADO = """
<div style="width:100%; box-sizing:border-box; padding:0 14mm;
            text-align:right; -webkit-print-color-adjust:exact;">
  <img src="{logo}" style="height:34px;">
</div>
"""

PLANTILLA_PIE = """
<div style="width:100%; box-sizing:border-box; padding:0 18mm;
            font-family:Arial,Helvetica,sans-serif; font-size:8pt; color:#5b6472;
            display:flex; justify-content:space-between; align-items:center;">
  <span>{titulo}</span>
  <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
</div>
"""


def _pdf_con_playwright(ruta_html: Path, ruta_pdf: Path, titulo: str) -> bool:
    """
    Motor preferido.  Usa las plantillas nativas de encabezado y pie del motor
    de impresión, que son las únicas que dibujan **dentro del margen** y por
    tanto no colisionan nunca con el texto.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False

    logo = _data_uri(LOGO_ENCABEZADO)
    if logo is None:
        return False

    try:
        with sync_playwright() as pw:
            navegador = pw.chromium.launch()
            pagina = navegador.new_page()
            pagina.goto(ruta_html.resolve().as_uri(), wait_until="load")
            pagina.pdf(
                path=str(ruta_pdf),
                format="A4",
                print_background=True,
                display_header_footer=True,
                header_template=PLANTILLA_ENCABEZADO.format(logo=logo),
                footer_template=PLANTILLA_PIE.format(titulo=titulo),
                margin={
                    "top": f"{ALTO_MEMBRETE_MM + 6}mm",
                    "bottom": "18mm",
                    "left": "18mm",
                    "right": "18mm",
                },
            )
            navegador.close()
    except Exception as exc:
        print(f"   (Playwright no pudo generar el PDF: {exc})")
        return False

    return ruta_pdf.exists() and ruta_pdf.stat().st_size > 0


def _buscar_navegador() -> str | None:
    candidatos = [
        os.environ.get("CHROME_BIN"),
        "/opt/pw-browsers/chromium",
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "msedge",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidatos:
        if not c:
            continue
        ruta = shutil.which(c) or (c if Path(c).exists() else None)
        if ruta:
            return ruta
    return None


def _pdf_con_navegador(ruta_html: Path, ruta_pdf: Path) -> bool:
    """
    Motor de respaldo.  Sin plantillas nativas: el logotipo aparece sólo como
    membrete de la portada, y la marca de agua en todas las páginas.
    """
    navegador = _buscar_navegador()
    if navegador is None:
        return False

    with tempfile.TemporaryDirectory() as perfil:
        comando = [
            navegador,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--user-data-dir={perfil}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={ruta_pdf}",
            ruta_html.resolve().as_uri(),
        ]
        resultado = subprocess.run(comando, capture_output=True, timeout=180)

    return ruta_pdf.exists() and ruta_pdf.stat().st_size > 0 and resultado.returncode == 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    ruta_md = Path(sys.argv[1])
    if not ruta_md.exists():
        print(f"No se encontró el archivo: {ruta_md}")
        return 1

    argumentos = [a for a in sys.argv[2:] if not a.startswith("-")]
    proteger = "--proteger" in sys.argv

    ruta_pdf = Path(argumentos[0]) if argumentos else ruta_md.with_suffix(".pdf")
    ruta_html = ruta_md.with_suffix(".html")

    texto = ruta_md.read_text(encoding="utf-8")
    titulo = next(
        (ln.lstrip("# ").strip() for ln in texto.splitlines() if ln.startswith("# ")),
        ruta_md.stem,
    )

    # Con plantillas nativas el logotipo ya se repite arriba; el membrete de
    # portada sólo se añade cuando hay que recurrir al motor de respaldo.
    hay_playwright = True
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        hay_playwright = False

    ruta_html.write_text(
        markdown_a_html(texto, titulo, con_membrete=not hay_playwright),
        encoding="utf-8",
    )
    print(f"HTML intermedio: {ruta_html}")

    generado = False
    if hay_playwright:
        generado = _pdf_con_playwright(ruta_html, ruta_pdf, titulo)
        if generado:
            print("Motor          : Playwright (encabezado y pie en el margen)")

    if not generado:
        # Reconstruir el HTML con membrete, ya que no habrá encabezado nativo.
        ruta_html.write_text(
            markdown_a_html(texto, titulo, con_membrete=True), encoding="utf-8"
        )
        generado = _pdf_con_navegador(ruta_html, ruta_pdf)
        if generado:
            print("Motor          : Chromium (membrete en portada)")
            print("                 Para el logotipo repetido en todas las páginas:")
            print("                 pip install playwright && playwright install chromium")

    if not generado:
        print(
            "\nNo se encontró Chrome/Chromium para la conversión automática.\n"
            f"Abra {ruta_html} en su navegador y use Ctrl+P → «Guardar como PDF».\n"
            "El estilo va incrustado en el HTML; active «Gráficos de fondo»."
        )
        return 2

    if proteger:
        _proteger(ruta_pdf)

    print(f"PDF generado   : {ruta_pdf}  ({ruta_pdf.stat().st_size // 1024} KB)")
    return 0


def _proteger(ruta_pdf: Path) -> None:
    """
    Cifra el PDF dejándolo de lectura libre pero sin permiso de extracción de
    texto ni de modificación.

    Conviene entender su alcance: impide el copiar-pegar y la edición en los
    lectores que respetan los permisos, pero **no impide capturas de pantalla**
    ni resiste a herramientas que ignoran deliberadamente esos indicadores.  Es
    una barrera de fricción y una declaración de intención, no un cifrado real
    del contenido.
    """
    try:
        import pypdf
        from pypdf.constants import UserAccessPermissions as Permisos
    except ImportError:
        print("   (--proteger requiere: pip install pypdf)")
        return

    try:
        lector = pypdf.PdfReader(str(ruta_pdf))
        escritor = pypdf.PdfWriter()
        for pagina in lector.pages:
            escritor.add_page(pagina)
        escritor.encrypt(
            user_password="",
            owner_password=base64.b64encode(os.urandom(12)).decode("ascii"),
            permissions_flag=Permisos.PRINT | Permisos.PRINT_TO_REPRESENTATION,
        )
        with open(ruta_pdf, "wb") as fh:
            escritor.write(fh)
        print("Protección     : copia de texto y edición deshabilitadas")
    except Exception as exc:
        print(f"   (no se pudo aplicar la protección: {exc})")


if __name__ == "__main__":
    raise SystemExit(main())
