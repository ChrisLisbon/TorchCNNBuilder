default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile \
		| while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: lint
lint: # Lint the project with flake8 lib.
	bash .build/lint_and_check.sh

.PHONY: doc
doc: # Build and run the doc locally.
	bash .build/doc.sh