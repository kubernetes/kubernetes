#!/usr/bin/env bats

load helpers

@test "runc version" {
  runc -v
  [ "$status" -eq 0 ]
  [[ ${lines[0]} =~ runc\ version\ [0-9]+\.[0-9]+\.[0-9]+ ]]
  [[ ${lines[1]} =~ commit:+ ]]
  [[ ${lines[2]} =~ spec:\ [0-9]+\.[0-9]+\.[0-9]+ ]]
}
