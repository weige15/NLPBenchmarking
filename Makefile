
.PHONY: all
all: run

.PHONY: install
install:
	poetry install

install_pip:
	pip install -r requirements.txt

run:
	poetry run python ./ragas_entry_point.py

clean:
	# TODO: 
	echo "TODO: not yet implemented"
