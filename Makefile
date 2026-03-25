UV ?= uv
STAMPIFY_WEB_HOST ?= 127.0.0.1
STAMPIFY_WEB_PORT ?= 8000
STAMPIFY_WEB_LOG_LEVEL ?= info

.PHONY: dev test

dev:
	STAMPIFY_WEB_HOST=$(STAMPIFY_WEB_HOST) STAMPIFY_WEB_PORT=$(STAMPIFY_WEB_PORT) STAMPIFY_WEB_LOG_LEVEL=$(STAMPIFY_WEB_LOG_LEVEL) $(UV) run --extra web python -m ink_print.webapp

test:
	$(UV) run --extra test pytest -q
