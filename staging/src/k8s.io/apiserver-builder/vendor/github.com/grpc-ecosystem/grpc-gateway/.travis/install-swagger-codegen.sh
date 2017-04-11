#!/bin/sh -eu
codegen_version=$1
if test -z "${codegen_version}"; then
	echo "Usage: .travis/install-swagger-codegen.sh codegen-version"
	exit 1
fi

wget http://repo1.maven.org/maven2/io/swagger/swagger-codegen-cli/${codegen_version}/swagger-codegen-cli-${codegen_version}.jar \
  -O $HOME/local/swagger-codegen-cli.jar
