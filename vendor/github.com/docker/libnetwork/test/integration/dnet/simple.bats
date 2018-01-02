#!/usr/bin/env bats

load helpers

@test "Test network create" {
    echo $(docker ps)
    run dnet_cmd $(inst_id2port 1) network create -d test mh1
    echo ${output}
    [ "$status" -eq 0 ]
    run dnet_cmd $(inst_id2port 1) network ls
    echo ${output}
    line=$(dnet_cmd $(inst_id2port 1) network ls | grep mh1)
    echo ${line}
    name=$(echo ${line} | cut -d" " -f2)
    driver=$(echo ${line} | cut -d" " -f3)
    echo ${name} ${driver}
    [ "$name" = "mh1" ]
    [ "$driver" = "test" ]
    dnet_cmd $(inst_id2port 1) network rm mh1
}

@test "Test network delete with id" {
    echo $(docker ps)
    run dnet_cmd $(inst_id2port 1) network create -d test mh1
    [ "$status" -eq 0 ]
    echo ${output}
    dnet_cmd $(inst_id2port 1) network rm ${output}
}

@test "Test service create" {
    echo $(docker ps)
    dnet_cmd $(inst_id2port 1) network create -d test multihost
    run dnet_cmd $(inst_id2port 1) service publish svc1.multihost
    echo ${output}
    [ "$status" -eq 0 ]
    run dnet_cmd $(inst_id2port 1) service ls
    echo ${output}
    echo ${lines[1]}
    [ "$status" -eq 0 ]
    svc=$(echo ${lines[1]} | cut -d" " -f2)
    network=$(echo ${lines[1]} | cut -d" " -f3)
    echo ${svc} ${network}
    [ "$network" = "multihost" ]
    [ "$svc" = "svc1" ]
    dnet_cmd $(inst_id2port 1) service unpublish svc1.multihost
    dnet_cmd $(inst_id2port 1) network rm multihost
}

@test "Test service delete with id" {
    echo $(docker ps)
    dnet_cmd $(inst_id2port 1) network create -d test multihost
    run dnet_cmd $(inst_id2port 1) service publish svc1.multihost
    [ "$status" -eq 0 ]
    echo ${output}
    run dnet_cmd $(inst_id2port 1) service ls
    [ "$status" -eq 0 ]
    echo ${output}
    echo ${lines[1]}
    id=$(echo ${lines[1]} | cut -d" " -f1)
    dnet_cmd $(inst_id2port 1) service unpublish ${id}.multihost
    dnet_cmd $(inst_id2port 1) network rm multihost
}

@test "Test service attach" {
    echo $(docker ps)
    dnet_cmd $(inst_id2port 1) network create -d test multihost
    dnet_cmd $(inst_id2port 1) service publish svc1.multihost
    dnet_cmd $(inst_id2port 1) container create container_1
    dnet_cmd $(inst_id2port 1) service attach container_1 svc1.multihost
    run dnet_cmd $(inst_id2port 1) service ls
    [ "$status" -eq 0 ]
    echo ${output}
    echo ${lines[1]}
    container=$(echo ${lines[1]} | cut -d" " -f4)
    [ "$container" = "container_1" ]
    dnet_cmd $(inst_id2port 1) service detach container_1 svc1.multihost
    dnet_cmd $(inst_id2port 1) container rm container_1
    dnet_cmd $(inst_id2port 1) service unpublish svc1.multihost
    dnet_cmd $(inst_id2port 1) network rm multihost
}
