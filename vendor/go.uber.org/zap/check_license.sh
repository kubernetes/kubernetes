#!/bin/bash -e

ERROR_COUNT=0
while read -r file
do
	case "$(head -1 "${file}")" in
		*"Copyright (c) "*" Uber Technologies, Inc.")
			# everything's cool
			;;
		*)
			echo "$file is missing license header."
			(( ERROR_COUNT++ ))
			;;
	esac
done < <(git ls-files "*\.go")

exit $ERROR_COUNT
