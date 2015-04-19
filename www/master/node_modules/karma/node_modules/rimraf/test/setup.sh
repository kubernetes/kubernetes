#!/bin/bash

set -e

files=10
folders=2
depth=4
target="$PWD/target"

rm -rf target

fill () {
  local depth=$1
  local files=$2
  local folders=$3
  local target=$4

  if ! [ -d $target ]; then
    mkdir -p $target
  fi

  local f

  f=$files
  while [ $f -gt 0 ]; do
    touch "$target/f-$depth-$f"
    let f--
  done

  let depth--

  if [ $depth -le 0 ]; then
    return 0
  fi

  f=$folders
  while [ $f -gt 0 ]; do
    mkdir "$target/folder-$depth-$f"
    fill $depth $files $folders "$target/d-$depth-$f"
    let f--
  done
}

fill $depth $files $folders $target

# sanity assert
[ -d $target ]
