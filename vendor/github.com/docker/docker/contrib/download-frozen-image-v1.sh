#!/usr/bin/env bash
set -e

# hello-world                      latest              ef872312fe1b        3 months ago        910 B
# hello-world                      latest              ef872312fe1bbc5e05aae626791a47ee9b032efa8f3bda39cc0be7b56bfe59b9   3 months ago        910 B

# debian                           latest              f6fab3b798be        10 weeks ago        85.1 MB
# debian                           latest              f6fab3b798be3174f45aa1eb731f8182705555f89c9026d8c1ef230cbf8301dd   10 weeks ago        85.1 MB

if ! command -v curl &> /dev/null; then
	echo >&2 'error: "curl" not found!'
	exit 1
fi

usage() {
	echo "usage: $0 dir image[:tag][@image-id] ..."
	echo "   ie: $0 /tmp/hello-world hello-world"
	echo "       $0 /tmp/debian-jessie debian:jessie"
	echo "       $0 /tmp/old-hello-world hello-world@ef872312fe1bbc5e05aae626791a47ee9b032efa8f3bda39cc0be7b56bfe59b9"
	echo "       $0 /tmp/old-debian debian:latest@f6fab3b798be3174f45aa1eb731f8182705555f89c9026d8c1ef230cbf8301dd"
	[ -z "$1" ] || exit "$1"
}

dir="$1" # dir for building tar in
shift || usage 1 >&2

[ $# -gt 0 -a "$dir" ] || usage 2 >&2
mkdir -p "$dir"

# hacky workarounds for Bash 3 support (no associative arrays)
images=()
rm -f "$dir"/tags-*.tmp
# repositories[busybox]='"latest": "...", "ubuntu-14.04": "..."'

while [ $# -gt 0 ]; do
	imageTag="$1"
	shift
	image="${imageTag%%[:@]*}"
	tag="${imageTag#*:}"
	imageId="${tag##*@}"
	[ "$imageId" != "$tag" ] || imageId=
	[ "$tag" != "$imageTag" ] || tag='latest'
	tag="${tag%@*}"

	imageFile="${image//\//_}" # "/" can't be in filenames :)

	token="$(curl -sSL -o /dev/null -D- -H 'X-Docker-Token: true' "https://index.docker.io/v1/repositories/$image/images" | tr -d '\r' | awk -F ': *' '$1 == "X-Docker-Token" { print $2 }')"

	if [ -z "$imageId" ]; then
		imageId="$(curl -sSL -H "Authorization: Token $token" "https://registry-1.docker.io/v1/repositories/$image/tags/$tag")"
		imageId="${imageId//\"/}"
	fi

	ancestryJson="$(curl -sSL -H "Authorization: Token $token" "https://registry-1.docker.io/v1/images/$imageId/ancestry")"
	if [ "${ancestryJson:0:1}" != '[' ]; then
		echo >&2 "error: /v1/images/$imageId/ancestry returned something unexpected:"
		echo >&2 "  $ancestryJson"
		exit 1
	fi

	IFS=','
	ancestry=( ${ancestryJson//[\[\] \"]/} )
	unset IFS

	if [ -s "$dir/tags-$imageFile.tmp" ]; then
		echo -n ', ' >> "$dir/tags-$imageFile.tmp"
	else
		images=( "${images[@]}" "$image" )
	fi
	echo -n '"'"$tag"'": "'"$imageId"'"' >> "$dir/tags-$imageFile.tmp"

	echo "Downloading '$imageTag' (${#ancestry[@]} layers)..."
	for imageId in "${ancestry[@]}"; do
		mkdir -p "$dir/$imageId"
		echo '1.0' > "$dir/$imageId/VERSION"

		curl -sSL -H "Authorization: Token $token" "https://registry-1.docker.io/v1/images/$imageId/json" -o "$dir/$imageId/json"

		# TODO figure out why "-C -" doesn't work here
		# "curl: (33) HTTP server doesn't seem to support byte ranges. Cannot resume."
		# "HTTP/1.1 416 Requested Range Not Satisfiable"
		if [ -f "$dir/$imageId/layer.tar" ]; then
			# TODO hackpatch for no -C support :'(
			echo "skipping existing ${imageId:0:12}"
			continue
		fi
		curl -SL --progress -H "Authorization: Token $token" "https://registry-1.docker.io/v1/images/$imageId/layer" -o "$dir/$imageId/layer.tar" # -C -
	done
	echo
done

echo -n '{' > "$dir/repositories"
firstImage=1
for image in "${images[@]}"; do
	imageFile="${image//\//_}" # "/" can't be in filenames :)

	[ "$firstImage" ] || echo -n ',' >> "$dir/repositories"
	firstImage=
	echo -n $'\n\t' >> "$dir/repositories"
	echo -n '"'"$image"'": { '"$(cat "$dir/tags-$imageFile.tmp")"' }' >> "$dir/repositories"
done
echo -n $'\n}\n' >> "$dir/repositories"

rm -f "$dir"/tags-*.tmp

echo "Download of images into '$dir' complete."
echo "Use something like the following to load the result into a Docker daemon:"
echo "  tar -cC '$dir' . | docker load"
