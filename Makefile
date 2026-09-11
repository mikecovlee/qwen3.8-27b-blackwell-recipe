SHELL := /bin/bash

VARIANT ?= fp8v
ifeq ($(VARIANT),fp8)
COMPOSE := inference/kv-fp8-text-only.yml
else ifeq ($(VARIANT),fp8v)
COMPOSE := inference/kv-fp8-text-image.yml
else
COMPOSE := inference/kv-nvfp4-text-image.yml
endif

.PHONY: help online offline export up down logs health bench eval check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

online: ## One-click online install (pull images, download model, start) VARIANT=nvfp4|fp8|fp8v
	VARIANT=$(VARIANT) bash scripts/online/setup.sh

export: ## On a networked machine: build an offline bundle/ directory
	bash scripts/offline/export.sh

offline: ## On an air-gapped machine: install from bundle/ VARIANT=nvfp4|fp8|fp8v
	VARIANT=$(VARIANT) bash scripts/offline/setup.sh

up: ## Start services (VARIANT=nvfp4|fp8|fp8v)
	docker compose --env-file .env -f $(COMPOSE) up -d
	docker compose --env-file .env -f gateway/docker-compose.yml up -d

down: ## Stop all services
	docker compose -f $(COMPOSE) down
	docker compose -f gateway/docker-compose.yml down

logs: ## Tail inference server logs
	docker logs -f llm-infer

health: ## Check service health
	bash scripts/check.sh --health

bench: ## Run the throughput benchmark
	python3 tools/benchmark.py

eval: ## Print how to run quality / long-context evaluations
	@echo "Quality:      python3 tools/gsm8k.py"
	@echo "Long context: python3 tools/ruler-niah.py | ruler-niah-multi.py | ruler-fwe.py"
	@echo "Spot check:   python3 tools/quality-spotcheck.py"

check: ## Compose consistency + secret scan
	bash scripts/check.sh
