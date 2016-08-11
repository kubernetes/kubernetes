#!/bin/bash

result_dir=$HOME/test_logs

if [ ! -d $result_dir ]; then 
  mkdir $result_dir
fi

if [ -f "$result_dir/latest-build.txt" ]; then
  latest=$(cat "$result_dir/latest-build.txt")
else
  latest=0
fi

((latest++))

./test/e2e_node/jenkins/benchmark/e2e-node-benchmark-jenkins.sh \
  ./test/e2e_node/jenkins/benchmark/jenkins-benchmark.properties \
  > $result_dir/"$latest".log 2>&1

echo $latest > "$result_dir/latest-build.txt"