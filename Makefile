ENV_NAME := eeg-music
ENV_FILE := env/env.yml

.PHONY: bootstrap create update rebuild kernel

bootstrap: update  ## create or update env fromcond env.yml
	@echo "Env ready: $(ENV_NAME)"

create:
	conda env create -n $(ENV_NAME) -f $(ENV_FILE)

update:
	# Try update; if env doesn't exist, create it
	conda env update -n $(ENV_NAME) -f $(ENV_FILE) --prune \
	|| conda env create -n $(ENV_NAME) -f $(ENV_FILE)

rebuild:          ## drop and recreate from scratch
	- conda env remove -n $(ENV_NAME)
	$(MAKE) create

kernel:           ## (optional) install Jupyter kernel for VS Code
	python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python ($(ENV_NAME))"