#!/bin/bash
set -e

# this list should match roughly what's in the Dockerfile (minus the explicit image IDs, of course)
images=(
	busybox:latest
	hello-world:frozen
	jess/unshare:latest
)

if ! docker inspect "${images[@]}" &> /dev/null; then
	hardCodedDir='/docker-frozen-images'
	if [ -d "$hardCodedDir" ]; then
		( set -x; tar -cC "$hardCodedDir" . | docker load )
	else
		dir="$DEST/frozen-images"
		# extract the exact "RUN download-frozen-image.sh" line from the Dockerfile itself for consistency
		# NOTE: this will fail if either "curl" is not installed or if the Dockerfile is not available/readable
		awk '
			$1 == "RUN" && $2 == "./contrib/download-frozen-image.sh" {
				for (i = 2; i < NF; i++)
					printf ( $i == "'"$hardCodedDir"'" ? "'"$dir"'" : $i ) " ";
				print $NF;
				if (/\\$/) {
					inCont = 1;
					next;
				}
			}
			inCont {
				print;
				if (!/\\$/) {
					inCont = 0;
				}
			}
		' Dockerfile | sh -x
		( set -x; tar -cC "$dir" . | docker load )
	fi
fi
