# -*- mode: sh -*-
#!/usr/bin/env bats

load helpers

@test "Test overlay network with consul" {
    skip_for_circleci
    test_overlay consul
}

@test "Test overlay network singlehost with consul" {
    skip_for_circleci
    test_overlay_singlehost consul
}

@test "Test overlay network with dnet restart" {
    skip_for_circleci
    test_overlay consul skip_rm
    docker restart dnet-1-consul
    wait_for_dnet $(inst_id2port 1) dnet-1-consul
    docker restart dnet-2-consul
    wait_for_dnet $(inst_id2port 2) dnet-2-consul
    docker restart dnet-3-consul
    wait_for_dnet $(inst_id2port 3) dnet-3-consul
    test_overlay consul skip_add
}

@test "Test overlay network internal network with consul" {
    skip_for_circleci
    test_overlay consul internal
}

@test "Test overlay network with dnet ungraceful shutdown" {
    skip_for_circleci
    dnet_cmd $(inst_id2port 1) network create -d overlay multihost
    start=1
    end=3
    for i in `seq ${start} ${end}`;
    do
	dnet_cmd $(inst_id2port $i) container create container_${i}
	net_connect ${i} container_${i} multihost
    done

    hrun runc $(dnet_container_name 1 consul) $(get_sbox_id 1 container_1) "ifconfig eth0"
    container_1_ip=$(echo ${output} | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}')

    # ungracefully kill dnet-1-consul container
    docker rm -f dnet-1-consul

    # forcefully unpublish the service from dnet2 when dnet1 is dead.
    dnet_cmd $(inst_id2port 2) service unpublish -f container_1.multihost
    dnet_cmd $(inst_id2port 2) container create container_1
    net_connect 2 container_1 multihost

    hrun runc $(dnet_container_name 2 consul) $(get_sbox_id 2 container_1) "ifconfig eth0"
    container_1_new_ip=$(echo ${output} | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}')

    if [ "$container_1_ip" != "$container_1_new_ip" ]; then
	exit 1
    fi
}
