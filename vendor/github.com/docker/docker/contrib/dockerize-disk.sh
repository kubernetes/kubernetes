#!/usr/bin/env bash
set -e

if ! command -v qemu-nbd &> /dev/null; then
  echo >&2 'error: "qemu-nbd" not found!'
  exit 1
fi

usage() {
  echo "Convert disk image to docker image"
  echo ""
  echo "usage: $0 image-name disk-image-file [ base-image ]"
  echo "   ie: $0 cirros:0.3.3 cirros-0.3.3-x86_64-disk.img"
  echo "       $0 ubuntu:cloud ubuntu-14.04-server-cloudimg-amd64-disk1.img ubuntu:14.04"
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

CURDIR=$(pwd)

image_name="${1%:*}"
image_tag="${1#*:}"
if [ "$image_tag" == "$1" ]; then
  image_tag="latest"
fi

disk_image_file="$2"
docker_base_image="$3"

block_device=/dev/nbd0

builddir=$(mktemp -d)

cleanup() {
  umount "$builddir/disk_image" || true
  umount "$builddir/workdir" || true
  qemu-nbd -d $block_device &> /dev/null || true
  rm -rf $builddir
}
trap cleanup EXIT

# Mount disk image
modprobe nbd max_part=63
qemu-nbd -rc ${block_device} -P 1 "$disk_image_file"
mkdir "$builddir/disk_image"
mount -o ro ${block_device} "$builddir/disk_image"

mkdir "$builddir/workdir"
mkdir "$builddir/diff"

base_image_mounts=""

# Unpack base image
if [ -n "$docker_base_image" ]; then
  mkdir -p "$builddir/base"
  docker pull "$docker_base_image"
  docker save "$docker_base_image" | tar -xC "$builddir/base"

  image_id=$(docker inspect -f "{{.Id}}" "$docker_base_image")
  while [ -n "$image_id" ]; do
    mkdir -p "$builddir/base/$image_id/layer"
    tar -xf "$builddir/base/$image_id/layer.tar" -C "$builddir/base/$image_id/layer"

    base_image_mounts="${base_image_mounts}:$builddir/base/$image_id/layer=ro+wh"
    image_id=$(docker inspect -f "{{.Parent}}" "$image_id")
  done
fi

# Mount work directory
mount -t aufs -o "br=$builddir/diff=rw${base_image_mounts},dio,xino=/dev/shm/aufs.xino" none "$builddir/workdir"

# Update files
cd $builddir
LC_ALL=C diff -rq disk_image workdir \
  | sed -re "s|Only in workdir(.*?): |DEL \1/|g;s|Only in disk_image(.*?): |ADD \1/|g;s|Files disk_image/(.+) and workdir/(.+) differ|UPDATE /\1|g" \
  | while read action entry; do
      case "$action" in
        ADD|UPDATE)
          cp -a "disk_image$entry" "workdir$entry"
          ;;
        DEL)
          rm -rf "workdir$entry"
          ;;
        *)
          echo "Error: unknown diff line: $action $entry" >&2
          ;;
      esac
    done

# Pack new image
new_image_id="$(for i in $(seq 1 32); do printf "%02x" $(($RANDOM % 256)); done)"
mkdir -p $builddir/result/$new_image_id
cd diff
tar -cf $builddir/result/$new_image_id/layer.tar *
echo "1.0" > $builddir/result/$new_image_id/VERSION
cat > $builddir/result/$new_image_id/json <<-EOS
{ "docker_version": "1.4.1"
, "id": "$new_image_id"
, "created": "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
EOS

if [ -n "$docker_base_image" ]; then
  image_id=$(docker inspect -f "{{.Id}}" "$docker_base_image")
  echo ", \"parent\": \"$image_id\"" >> $builddir/result/$new_image_id/json
fi

echo "}" >> $builddir/result/$new_image_id/json

echo "{\"$image_name\":{\"$image_tag\":\"$new_image_id\"}}" > $builddir/result/repositories

cd $builddir/result

# mkdir -p $CURDIR/$image_name
# cp -r * $CURDIR/$image_name
tar -c * | docker load
