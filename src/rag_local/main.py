"""CLI 인터페이스: rag-local ingest / query / serve."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="rag-local",
    help="로컬 RAG 시스템 - 문서 기반 Q&A",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def ingest(
    directory: Optional[Path] = typer.Option(None, "--dir", "-d", help="문서 디렉토리 경로"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """문서를 처리하여 인덱스에 저장한다."""
    _setup_logging(verbose)
    from rag_local.llm.provider import init_llama_settings
    from rag_local.ingest.pipeline import ingest as run_ingest

    console.print("[bold]문서 처리 시작...[/bold]")
    init_llama_settings()
    result = run_ingest(directory)

    console.print(f"\n[green]처리 완료:[/green] {result['ingested']}개")
    console.print(f"[dim]스킵:[/dim] {result['skipped']}개")
    if result["errors"]:
        console.print(f"[red]에러:[/red] {len(result['errors'])}개")
        for err in result["errors"]:
            console.print(f"  - {err}")


@app.command()
def query(
    question: str = typer.Argument(..., help="질문"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """문서 기반으로 질문에 답한다."""
    _setup_logging(verbose)
    from rag_local.rag.engine import query as run_query

    console.print(f"\n[bold]질문:[/bold] {question}\n")

    with console.status("검색 및 답변 생성 중..."):
        result = run_query(question)

    console.print(f"[green]답변:[/green]\n{result.answer}\n")

    if result.sources:
        table = Table(title="출처")
        table.add_column("파일", style="cyan")
        table.add_column("페이지", justify="center")
        table.add_column("점수", justify="right")
        for src in result.sources:
            table.add_row(
                src["file_name"],
                str(src.get("page_number", "-")),
                str(src.get("score", "-")),
            )
        console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """FastAPI REST API 서버를 시작한다."""
    _setup_logging(verbose)
    import uvicorn

    console.print(f"[bold]API 서버 시작:[/bold] http://{host}:{port}")
    console.print("[dim]Ctrl+C로 종료[/dim]")
    uvicorn.run("rag_local.api.app:app", host=host, port=port, reload=False)


@app.command()
def discord(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Discord 봇을 시작한다."""
    _setup_logging(verbose)
    from rag_local.bot.discord_bot import run_bot

    console.print("[bold]Discord 봇 시작...[/bold]")
    console.print("[dim]Ctrl+C로 종료[/dim]")
    run_bot()


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """인증 및 인덱스 상태를 확인한다."""
    _setup_logging(verbose)
    from rag_local.auth.token import is_logged_in, get_auth_mode
    from rag_local.config import settings

    console.print("\n[bold]=== RAG Local 상태 ===[/bold]\n")

    # 인증
    mode = "unknown"
    try:
        mode = get_auth_mode()
    except Exception:
        pass
    logged_in = is_logged_in()
    console.print(f"인증 모드: {mode}")
    console.print(f"로그인: {'[green]✔[/green]' if logged_in else '[red]✘[/red]'}")
    console.print(f"모델: {settings.llm_model}")
    console.print(f"임베딩: {settings.embed_model}")

    # 문서
    doc_dir = settings.documents_dir
    if doc_dir.exists():
        files = list(doc_dir.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        console.print(f"문서 디렉토리: {doc_dir} ({file_count}개 파일)")
    else:
        console.print(f"문서 디렉토리: [red]없음[/red] ({doc_dir})")

    # ChromaDB
    chroma_dir = settings.chroma_persist_dir
    console.print(f"ChromaDB: {'[green]있음[/green]' if chroma_dir.exists() else '[dim]없음[/dim]'}")
    console.print()


if __name__ == "__main__":
    app()
