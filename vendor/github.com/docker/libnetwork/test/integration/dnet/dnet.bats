#!/usr/bin/env bats

load helpers

@test "Test dnet custom port" {
    start_dnet 1 a 4567
    dnet_cmd 4567 network ls
    stop_dnet 1 a
}

@test "Test dnet invalid custom port" {
    start_dnet 1 b 4567
    run dnet_cmd 4568 network ls
    echo ${output}
    [ "$status" -ne 0 ]
    stop_dnet 1 b
}

@test "Test dnet invalid params" {
    start_dnet 1 c
    run dnet_cmd 8080 network ls
    echo ${output}
    [ "$status" -ne 0 ]
    run ./bin/dnet -H=unix://var/run/dnet.sock network ls
    echo ${output}
    [ "$status" -ne 0 ]
    run ./bin/dnet -H= -l=invalid network ls
    echo ${output}
    [ "$status" -ne 0 ]
    stop_dnet 1 c
}
