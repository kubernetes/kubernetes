#!/usr/bin/env bash
bazel build //proto/cel/expr:all

rm -vf ./*.pb.go

files=( $(bazel cquery //proto/cel/expr:expr_go_proto --output=starlark --starlark:expr="'\n'.join([f.path for f in target.output_groups.go_generated_srcs.to_list()])") )
for src in "${files[@]}";
do
  cp -v "${src}" ./
done
