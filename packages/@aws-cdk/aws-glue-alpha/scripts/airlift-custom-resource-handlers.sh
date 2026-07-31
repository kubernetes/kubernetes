#!/bin/bash

# This is a pre-build script that is run prior to building the aws-glue-alpha module.
# aws-glue-alpha has a dev dependency on custom-resource-handlers which triggers that
# module to build first. Once complete, this script will run to move all generated custom
# resource code into a custom-resource-handlers directory directly in aws-glue-alpha.

scriptdir=$(cd $(dirname $0) && pwd)
customresourcedir=$(node -p "path.dirname(require.resolve('@aws-cdk/custom-resource-handlers/package.json'))")
awscdklibdir=${scriptdir}/..

# copy code generated custom resource files into aws-glue-alpha/custom-resource-handlers
# from custom-resource-handlers/dist/aws-glue-alpha
function airlift() {
  mkdir -p $awscdklibdir/custom-resource-handlers/$1
  cp $customresourcedir/$2 $awscdklibdir/custom-resource-handlers/$1
}

# recursively iterates over directories and files in custom-resource-handlers/dist/aws-glue-alpha
# and copies them into aws-glue-alpha/custom-resource-handlers
function recurse() {
  local dir=$1

  for file in $dir/*; do
    if [ -f $file ]; then
      case $file in
        $customresourcedir/dist/aws-glue-alpha/*.generated.ts)
          cr=$(echo $file | rev | cut -d "/" -f 2-3 | rev)
          airlift $cr $cr/*.generated.ts
          ;;
        $customresourcedir/dist/aws-glue-alpha/*/index.*)
          cr=$(echo $file | rev | cut -d "/" -f 2-4 | rev)
          airlift $cr $cr/index.*
          ;;
      esac
    fi

    if [ -d $file ]; then
      recurse $file
    fi
  done
}

recurse $customresourcedir/dist/aws-glue-alpha
