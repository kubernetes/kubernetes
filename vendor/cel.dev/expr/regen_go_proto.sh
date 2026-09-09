#!/bin/sh
bazel build //proto/cel/expr/conformance/...
files=($(bazel aquery 'kind(proto, //proto/cel/expr/conformance/...)' | grep Outputs | grep "[.]pb[.]go" | sed 's/Outputs: \[//' | sed 's/\]//' | tr "," "\n"))
for src in ${files[@]};
do
  dst=$(echo $src | sed 's/\(.*\/cel.dev\/expr\/\(.*\)\)/\2/')
  echo "copying $dst"
  $(cp $src $dst)
done
