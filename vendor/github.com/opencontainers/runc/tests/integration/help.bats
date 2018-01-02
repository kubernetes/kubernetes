#!/usr/bin/env bats

load helpers

@test "runc -h" {
  runc -h
  [ "$status" -eq 0 ]
  [[ ${lines[0]} =~ NAME:+ ]]
  [[ ${lines[1]} =~ runc\ '-'\ Open\ Container\ Initiative\ runtime+ ]]

  runc --help
  [ "$status" -eq 0 ]
  [[ ${lines[0]} =~ NAME:+ ]]
  [[ ${lines[1]} =~ runc\ '-'\ Open\ Container\ Initiative\ runtime+ ]]
}

@test "runc command -h" {
  runc checkpoint -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ checkpoint+ ]]

  runc delete -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ delete+ ]]

  runc events -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ events+ ]]

  runc exec -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ exec+ ]]

  runc kill -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ kill+ ]]

  runc list -h
  [ "$status" -eq 0 ]
  [[ ${lines[0]} =~ NAME:+ ]]
  [[ ${lines[1]} =~ runc\ list+ ]]

  runc list --help
  [ "$status" -eq 0 ]
  [[ ${lines[0]} =~ NAME:+ ]]
  [[ ${lines[1]} =~ runc\ list+ ]]

  runc pause -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ pause+ ]]

  runc restore -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ restore+ ]]

  runc resume -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ resume+ ]]

  # We don't use runc_spec here, because we're just testing the help page.
  runc spec -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ spec+ ]]

  runc start -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ start+ ]]

  runc run -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ run+ ]]

  runc state -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ state+ ]]

  runc update -h
  [ "$status" -eq 0 ]
  [[ ${lines[1]} =~ runc\ update+ ]]

}

@test "runc foo -h" {
  runc foo -h
  [ "$status" -ne 0 ]
  [[ "${output}" == *"No help topic for 'foo'"* ]]
}
