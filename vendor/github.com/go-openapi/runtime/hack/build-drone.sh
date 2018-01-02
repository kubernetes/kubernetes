#!/bin/bash
set -e -o pipefail

 mkdir -p /drone/{testresults,coverage,dist}
 go test -race -timeout 20m -v ./... | go-junit-report -dir /drone/testresults

# Run test coverage on each subdirectories and merge the coverage profile.
echo "mode: ${GOCOVMODE-count}" > profile.cov

# Standard go tooling behavior is to ignore dirs with leading underscores
# skip generator for race detection and coverage
for dir in $(go list ./...)
do
  pth="$GOPATH/src/$dir"
  go test -covermode=${GOCOVMODE-count} -coverprofile=${pth}/profile.out $dir
  if [ -f $pth/profile.out ]
  then
      cat $pth/profile.out | tail -n +2 >> profile.cov
      # rm $pth/profile.out
  fi
done

go tool cover -func profile.cov
gocov convert profile.cov | gocov report
gocov convert profile.cov | gocov-html > /drone/coverage/coverage-${CI_BUILD_NUM-"0"}.html