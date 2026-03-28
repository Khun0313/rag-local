"""Discord 봇: RAG 시스템과 연동하여 문서 기반 Q&A 제공.

사용법:
  그냥 메시지 입력  — 문서 기반 Q&A (기본 동작)
  !ask <질문>       — 문서 기반 Q&A (명시적)
  !upload + 첨부    — 문서 업로드 + 자동 ingest
  !delete <파일명>  — 문서 삭제
  !docs             — 문서 목록
  !ingest           — 문서 재처리
  !status           — 시스템 상태
  !help_rag         — 도움말

참고: auto-trade/auto_trader/notifications/discord_bot.py
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from functools import partial
from pathlib import Path

import discord
from discord.ext import commands

from rag_local.config import settings
from rag_local.ingest.parsers import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 1900


class RAGBot(commands.Bot):
    """RAG Local Discord 봇."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents, help_command=None)
        self._rag_initialized = False
        self._show_sources = True
        self._setup_commands()

    def _ensure_rag(self):
        """RAG 엔진을 지연 초기화한다."""
        if not self._rag_initialized:
            from rag_local.llm.provider import init_llama_settings
            init_llama_settings()
            self._rag_initialized = True

    async def _do_ask(self, ctx_or_message, question: str):
        """질문에 답변하는 공통 로직."""
        try:
            self._ensure_rag()
            from rag_local.rag.engine import query

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, partial(query, question)
            )

            answer = result.answer
            if self._show_sources and result.sources:
                # 중복 제거 + 파일명/페이지별 그룹핑
                seen = []
                for s in result.sources:
                    entry = s['file_name']
                    if s.get('page_number'):
                        entry += f" > p.{s['page_number']}"
                    if entry not in seen:
                        seen.append(entry)
                sources_str = "\n".join(f"  - {e}" for e in seen)
                answer += f"\n\n---\n📎 **참고 문서:**\n{sources_str}"

            if len(answer) > MAX_MESSAGE_LENGTH:
                answer = answer[:MAX_MESSAGE_LENGTH] + "\n...(생략)"

            await ctx_or_message.reply(answer)

        except Exception as e:
            logger.error("ask 에러: %s", e)
            await ctx_or_message.reply(f"❌ 오류가 발생했습니다: {e}")

    def _setup_commands(self):
        @self.command(name="ask")
        async def ask(ctx: commands.Context, *, question: str):
            """문서 기반으로 질문에 답변한다."""
            async with ctx.typing():
                await self._do_ask(ctx, question)

        @self.command(name="upload")
        async def upload(ctx: commands.Context):
            """첨부된 문서를 저장하고 자동 ingest한다."""
            if not ctx.message.attachments:
                await ctx.reply("❓ 파일을 첨부해주세요. (PDF, TXT, DOCX)")
                return

            async with ctx.typing():
                saved = []
                skipped = []
                for att in ctx.message.attachments:
                    # Discord는 한글 파일명을 title에 보존, filename은 ASCII 변환
                    original_name = att.title or att.filename
                    # title에 확장자가 없으면 filename에서 가져옴
                    if not Path(original_name).suffix and Path(att.filename).suffix:
                        original_name += Path(att.filename).suffix
                    ext = Path(original_name).suffix.lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        skipped.append(f"{original_name} (미지원 형식)")
                        continue

                    save_path = settings.documents_dir / original_name
                    settings.documents_dir.mkdir(parents=True, exist_ok=True)
                    await att.save(save_path)
                    saved.append(original_name)
                    logger.info("파일 저장: %s", save_path)

                if not saved:
                    await ctx.reply(
                        f"❌ 지원하는 파일이 없습니다.\n"
                        f"지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}\n"
                        f"스킵: {', '.join(skipped)}"
                    )
                    return

                # 자동 ingest
                self._ensure_rag()
                from rag_local.ingest.pipeline import ingest as run_ingest

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, run_ingest)

                msg = (
                    f"✅ **파일 업로드 완료**\n"
                    f"  저장: {', '.join(saved)}\n"
                    f"  처리: {result['ingested']}개 | 스킵: {result['skipped']}개"
                )
                if skipped:
                    msg += f"\n  미지원: {', '.join(skipped)}"
                if result["errors"]:
                    msg += f"\n  에러: {len(result['errors'])}개"
                await ctx.reply(msg)

        @self.command(name="delete")
        async def delete(ctx: commands.Context, *, filename: str):
            """문서를 삭제한다."""
            file_path = settings.documents_dir / filename

            if not file_path.exists():
                # 부분 매칭 시도
                matches = [
                    f for f in settings.documents_dir.iterdir()
                    if f.is_file() and filename.lower() in f.name.lower()
                ]
                if len(matches) == 1:
                    file_path = matches[0]
                elif len(matches) > 1:
                    names = "\n".join(f"  - {f.name}" for f in matches)
                    await ctx.reply(f"❓ 여러 파일이 매칭됩니다. 정확한 파일명을 입력해주세요:\n{names}")
                    return
                else:
                    await ctx.reply(f"❌ 파일을 찾을 수 없습니다: `{filename}`")
                    return

            file_path.unlink()
            logger.info("파일 삭제: %s", file_path)

            # 인덱스 재구성 안내
            await ctx.reply(
                f"🗑️ **삭제 완료:** `{file_path.name}`\n"
                f"인덱스 반영을 위해 `!ingest`를 실행해주세요."
            )

        @self.command(name="docs")
        async def docs(ctx: commands.Context):
            """현재 문서 목록을 표시한다."""
            if not settings.documents_dir.exists():
                await ctx.reply("📂 문서 디렉토리가 없습니다.")
                return

            files = sorted(
                f for f in settings.documents_dir.rglob("*")
                if f.is_file() and not f.name.startswith(".")
            )

            if not files:
                await ctx.reply("📂 등록된 문서가 없습니다.")
                return

            lines = []
            for f in files:
                size_kb = f.stat().st_size / 1024
                ext = f.suffix.lower()
                supported = "✅" if ext in SUPPORTED_EXTENSIONS else "⚠️"
                lines.append(f"  {supported} `{f.name}` ({size_kb:.1f}KB)")

            msg = f"📂 **문서 목록** ({len(files)}개)\n" + "\n".join(lines)
            if len(msg) > MAX_MESSAGE_LENGTH:
                msg = msg[:MAX_MESSAGE_LENGTH] + "\n...(생략)"
            await ctx.reply(msg)

        @self.command(name="ingest")
        async def ingest(ctx: commands.Context):
            """문서를 재처리한다."""
            async with ctx.typing():
                try:
                    self._ensure_rag()
                    from rag_local.ingest.pipeline import ingest as run_ingest

                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, run_ingest)

                    msg = (
                        f"✅ **문서 처리 완료**\n"
                        f"  처리: {result['ingested']}개\n"
                        f"  스킵: {result['skipped']}개"
                    )
                    if result["errors"]:
                        msg += f"\n  에러: {len(result['errors'])}개"
                    await ctx.reply(msg)

                except Exception as e:
                    logger.error("!ingest 에러: %s", e)
                    await ctx.reply(f"❌ 오류: {e}")

        @self.command(name="status")
        async def status(ctx: commands.Context):
            """시스템 상태를 확인한다."""
            from rag_local.auth.token import is_logged_in, get_auth_mode

            mode = "unknown"
            try:
                mode = get_auth_mode()
            except Exception:
                pass
            logged_in = is_logged_in()

            doc_count = 0
            if settings.documents_dir.exists():
                doc_count = sum(
                    1 for f in settings.documents_dir.rglob("*") if f.is_file()
                )

            chroma_exists = settings.chroma_persist_dir.exists()

            msg = (
                f"📊 **RAG Local 상태**\n"
                f"  인증: {'✅' if logged_in else '❌'} ({mode})\n"
                f"  모델: `{settings.llm_model}`\n"
                f"  임베딩: `{settings.embed_model}`\n"
                f"  문서: {doc_count}개\n"
                f"  인덱스: {'✅' if chroma_exists else '❌'}"
            )
            await ctx.reply(msg)

        @self.command(name="sources")
        async def sources(ctx: commands.Context):
            """출처 표시를 켜거나 끈다."""
            self._show_sources = not self._show_sources
            state = "ON ✅" if self._show_sources else "OFF ❌"
            await ctx.reply(f"📎 출처 표시: **{state}**")

        @self.command(name="help_rag")
        async def help_rag(ctx: commands.Context):
            """도움말을 표시한다."""
            sources_state = "ON" if self._show_sources else "OFF"
            msg = (
                "📖 **RAG Local 봇 명령어**\n\n"
                "그냥 메시지 입력 → 문서 기반 Q&A\n"
                "`!ask <질문>` — 문서 기반 Q&A (명시적)\n"
                "`!upload` + 파일 첨부 — 문서 업로드 + 자동 처리\n"
                "`!delete <파일명>` — 문서 삭제\n"
                "`!docs` — 문서 목록\n"
                "`!ingest` — 문서 재처리\n"
                "`!sources` — 출처 표시 토글 (현재: " + sources_state + ")\n"
                "`!status` — 시스템 상태\n"
                "`!help_rag` — 이 도움말"
            )
            await ctx.reply(msg)

    async def on_ready(self):
        logger.info("Discord 봇 로그인: %s (ID: %s)", self.user.name, self.user.id)
        if settings.discord_channel_id:
            channel = self.get_channel(settings.discord_channel_id)
            if channel:
                await channel.send("🟢 **RAG Local 봇이 시작되었습니다.** `!help_rag`로 명령어를 확인하세요.")

    async def on_message(self, message: discord.Message):
        """! 없이 메시지를 보내면 자동으로 ask로 처리한다."""
        # 봇 자신의 메시지 무시
        if message.author == self.user:
            return

        # 봇이 아닌 다른 봇 메시지 무시
        if message.author.bot:
            return

        # ! 명령어는 기존 명령어 처리로 위임
        if message.content.startswith("!"):
            await self.process_commands(message)
            return

        # 첨부 파일만 있고 텍스트 없으면 무시 (upload 명령 안내)
        if not message.content.strip() and message.attachments:
            await message.reply("💡 파일을 등록하려면 `!upload`와 함께 첨부해주세요.")
            return

        # 텍스트가 있으면 자동으로 ask 처리
        if message.content.strip():
            async with message.channel.typing():
                await self._do_ask(message, message.content.strip())

    async def on_command_error(self, ctx: commands.Context, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.reply("❓ 질문을 입력해주세요. 예: `!ask 연차휴가는 며칠인가요?`")
        elif isinstance(error, commands.CommandNotFound):
            pass  # 무시
        else:
            logger.error("명령어 에러: %s", error)
            await ctx.reply(f"❌ 오류: {error}")


def run_bot():
    """Discord 봇을 실행한다."""
    token = settings.discord_bot_token
    if not token:
        raise RuntimeError(
            "DISCORD_BOT_TOKEN이 설정되지 않았습니다.\n"
            ".env 파일에 RAG_DISCORD_BOT_TOKEN=<토큰>을 추가하세요."
        )
    bot = RAGBot()
    bot.run(token, log_handler=None)
