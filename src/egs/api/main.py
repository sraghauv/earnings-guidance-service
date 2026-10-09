"""App factory. Opens the DB connection and LLM provider in the lifespan
handler so importing this module never touches disk or the network."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from egs.api.deps import build_av_client, build_provider
from egs.api.routes import router
from egs.config import Settings, get_settings
from egs.db import connect


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.conn = connect(settings.db_path)

        try:
            app.state.provider = build_provider(settings)
            app.state.provider_error = None
        except Exception as exc:  # backend not installed, bad config, etc.
            app.state.provider = None
            app.state.provider_error = str(exc)

        try:
            app.state.av_client = build_av_client(settings)
            app.state.av_client_error = None
        except Exception as exc:
            app.state.av_client = None
            app.state.av_client_error = str(exc)

        yield

        app.state.conn.close()

    app = FastAPI(title="earnings-guidance-service", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
