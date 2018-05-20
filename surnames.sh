#!/usr/bin/env bash

MODE="$1"
shift

case "$MODE" in
	train)
		python -m surnames.train $@
		;;
	eval)
		python -m surnames.eval $@
		;;
	predict)
		python -m surnames.predict $@
		;;
	*)
		echo "Usage: ./surnames.sh [train|eval|predict]"
esac
