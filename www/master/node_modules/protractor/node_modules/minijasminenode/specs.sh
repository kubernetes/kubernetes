#!/usr/bin/env bash

entry="node lib/cli.js"

#command="${entry} spec/because2.js"
#echo $command
#time $command
#exit

echo "All these tests should pass"
command="${entry} spec/*_spec.js"
echo $command
time $command #/nested/uber-nested
echo -e "\033[1;35m--- Should have 44, 74 assertions, 0 failures. ---\033[0m"
echo ""

echo "These should be examples of failing tests"
command="${entry} spec/because.js spec/failure_egs.js spec/syntax_error.js"
echo $command
time $command #/nested/uber-nested
echo -e "\033[1;35m--- Should have 12 tests, 18 assertions, 11 failures ---\033[0m"
echo ""
