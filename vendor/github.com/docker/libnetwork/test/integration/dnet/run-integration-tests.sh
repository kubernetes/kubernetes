#!/usr/bin/env bash

set -e

export INTEGRATION_ROOT=./integration-tmp
export TMPC_ROOT=./integration-tmp/tmpc

declare -A cmap

trap "cleanup_containers" EXIT SIGINT

function cleanup_containers() {
    for c in "${!cmap[@]}";
    do
	docker stop $c 1>>${INTEGRATION_ROOT}/test.log 2>&1 || true
	if [ -z "$CIRCLECI" ]; then
	    docker rm -f $c 1>>${INTEGRATION_ROOT}/test.log 2>&1 || true
	fi
    done

    unset cmap
}

function run_bridge_tests() {
    ## Setup
    start_dnet 1 bridge 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-bridge]=dnet-1-bridge

    ## Run the test cases
    ./integration-tmp/bin/bats ./test/integration/dnet/bridge.bats

    ## Teardown
    stop_dnet 1 bridge 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-bridge]
}

function run_overlay_local_tests() {
    ## Test overlay network in local scope
    ## Setup
    start_dnet 1 local 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-local]=dnet-1-local
    start_dnet 2 local:$(dnet_container_ip 1 local) 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-2-local]=dnet-2-local
    start_dnet 3 local:$(dnet_container_ip 1 local) 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-3-local]=dnet-3-local

    ## Run the test cases
    ./integration-tmp/bin/bats ./test/integration/dnet/overlay-local.bats

    ## Teardown
    stop_dnet 1 local 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-local]
    stop_dnet 2 local 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-2-local]
    stop_dnet 3 local 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-3-local]
}

function run_overlay_consul_tests() {
    ## Test overlay network with consul
    ## Setup
    start_dnet 1 consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-consul]=dnet-1-consul
    start_dnet 2 consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-2-consul]=dnet-2-consul
    start_dnet 3 consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-3-consul]=dnet-3-consul

    ## Run the test cases
    ./integration-tmp/bin/bats ./test/integration/dnet/overlay-consul.bats

    ## Teardown
    stop_dnet 1 consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-consul]
    stop_dnet 2 consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-2-consul]
    stop_dnet 3 consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-3-consul]
}

function run_overlay_consul_host_tests() {
    export _OVERLAY_HOST_MODE="true"
    ## Setup
    start_dnet 1 consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-consul]=dnet-1-consul

    ## Run the test cases
    ./integration-tmp/bin/bats ./test/integration/dnet/overlay-consul-host.bats

    ## Teardown
    stop_dnet 1 consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-consul]
    unset _OVERLAY_HOST_MODE
}

function run_overlay_zk_tests() {
    ## Test overlay network with zookeeper
    start_dnet 1 zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-zookeeper]=dnet-1-zookeeper
    start_dnet 2 zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-2-zookeeper]=dnet-2-zookeeper
    start_dnet 3 zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-3-zookeeper]=dnet-3-zookeeper

    ./integration-tmp/bin/bats ./test/integration/dnet/overlay-zookeeper.bats

    stop_dnet 1 zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-zookeeper]
    stop_dnet 2 zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-2-zookeeper]
    stop_dnet 3 zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-3-zookeeper]
}

function run_overlay_etcd_tests() {
    ## Test overlay network with etcd
    start_dnet 1 etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-etcd]=dnet-1-etcd
    start_dnet 2 etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-2-etcd]=dnet-2-etcd
    start_dnet 3 etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-3-etcd]=dnet-3-etcd

    ./integration-tmp/bin/bats ./test/integration/dnet/overlay-etcd.bats

    stop_dnet 1 etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-etcd]
    stop_dnet 2 etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-2-etcd]
    stop_dnet 3 etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-3-etcd]
}

function run_dnet_tests() {
    # Test dnet configuration options
    ./integration-tmp/bin/bats ./test/integration/dnet/dnet.bats
}

function run_simple_consul_tests() {
    # Test a single node configuration with a global scope test driver
    ## Setup
    start_dnet 1 simple 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-simple]=dnet-1-simple

    ## Run the test cases
    ./integration-tmp/bin/bats ./test/integration/dnet/simple.bats

    ## Teardown
    stop_dnet 1 simple 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-simple]
}

function run_multi_consul_tests() {
    # Test multi node configuration with a global scope test driver backed by consul

    ## Setup
    start_dnet 1 multi_consul consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-multi_consul]=dnet-1-multi_consul
    start_dnet 2 multi_consul consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-2-multi_consul]=dnet-2-multi_consul
    start_dnet 3 multi_consul consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-3-multi_consul]=dnet-3-multi_consul

    ## Run the test cases
    ./integration-tmp/bin/bats ./test/integration/dnet/multi.bats

    ## Teardown
    stop_dnet 1 multi_consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-multi_consul]
    stop_dnet 2 multi_consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-2-multi_consul]
    stop_dnet 3 multi_consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-3-multi_consul]
}

function run_multi_zk_tests() {
    # Test multi node configuration with a global scope test driver backed by zookeeper

    ## Setup
    start_dnet 1 multi_zk zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-multi_zk]=dnet-1-multi_zk
    start_dnet 2 multi_zk zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-2-multi_zk]=dnet-2-multi_zk
    start_dnet 3 multi_zk zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-3-multi_zk]=dnet-3-multi_zk

    ## Run the test cases
    ./integration-tmp/bin/bats ./test/integration/dnet/multi.bats

    ## Teardown
    stop_dnet 1 multi_zk 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-multi_zk]
    stop_dnet 2 multi_zk 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-2-multi_zk]
    stop_dnet 3 multi_zk 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-3-multi_zk]
}

function run_multi_etcd_tests() {
    # Test multi node configuration with a global scope test driver backed by etcd

    ## Setup
    start_dnet 1 multi_etcd etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-1-multi_etcd]=dnet-1-multi_etcd
    start_dnet 2 multi_etcd etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-2-multi_etcd]=dnet-2-multi_etcd
    start_dnet 3 multi_etcd etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dnet-3-multi_etcd]=dnet-3-multi_etcd

    ## Run the test cases
    ./integration-tmp/bin/bats ./test/integration/dnet/multi.bats

    ## Teardown
    stop_dnet 1 multi_etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-1-multi_etcd]
    stop_dnet 2 multi_etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-2-multi_etcd]
    stop_dnet 3 multi_etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    unset cmap[dnet-3-multi_etcd]
}

source ./test/integration/dnet/helpers.bash

if [ ! -d ${INTEGRATION_ROOT} ]; then
    mkdir -p ${INTEGRATION_ROOT}
    git clone https://github.com/sstephenson/bats.git ${INTEGRATION_ROOT}/bats
    ./integration-tmp/bats/install.sh ./integration-tmp
fi

if [ ! -d ${TMPC_ROOT} ]; then
    mkdir -p ${TMPC_ROOT}
    docker pull busybox:ubuntu
    docker export $(docker create busybox:ubuntu) > ${TMPC_ROOT}/busybox.tar
    mkdir -p ${TMPC_ROOT}/rootfs
    tar -C ${TMPC_ROOT}/rootfs -xf ${TMPC_ROOT}/busybox.tar
fi

# Suite setup

if [ -z "$SUITES" ]; then
    if [ -n "$CIRCLECI" ]
    then
	# We can only run a limited list of suites in circleci because of the
	# old kernel and limited docker environment.
	suites="dnet multi_consul multi_zk multi_etcd"
    else
	suites="dnet multi_consul multi_zk multi_etcd  bridge overlay_consul overlay_consul_host overlay_zk overlay_etcd"
    fi
else
    suites="$SUITES"
fi

if [[ ( "$suites" =~ .*consul.* ) ||  ( "$suites" =~ .*bridge.* ) ]]; then
    echo "Starting consul ..."
    start_consul 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[pr_consul]=pr_consul
fi

if [[ "$suites" =~ .*zk.* ]]; then
    echo "Starting zookeeper ..."
    start_zookeeper 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[zookeeper_server]=zookeeper_server
fi

if [[ "$suites" =~ .*etcd.* ]]; then
    echo "Starting etcd ..."
    start_etcd 1>>${INTEGRATION_ROOT}/test.log 2>&1
    cmap[dn_etcd]=dn_etcd
fi

echo ""

for suite in ${suites};
do
    suite_func=run_${suite}_tests
    echo "Running ${suite}_tests ..."
    declare -F $suite_func >/dev/null && $suite_func
    echo ""
done
