#!/usr/bin/env bash

# Required tools
DOCKER="${DOCKER:-docker}"
NSENTER="${NSENTER:-nsenter}"
BRIDGE="${BRIDGE:-bridge}"
BRCTL="${BRCTL:-brctl}"
IPTABLES="${IPTABLES:-iptables}"

NSDIR=/var/run/docker/netns
BRIDGEIF=br0

function die {
    echo $*
    exit 1
}

type -P ${DOCKER} > /dev/null || die "This tool requires the docker binary"
type -P ${NSENTER} > /dev/null || die "This tool requires nsenter"
type -P ${BRIDGE} > /dev/null || die "This tool requires bridge"
type -P ${BRCTL} > /dev/null || die "This tool requires brctl"
type -P ${IPTABLES} > /dev/null || die "This tool requires iptables"

echo "iptables configuration"
${IPTABLES} -n -v -L -t filter
${IPTABLES} -n -v -L -t nat
echo ""

echo "Overlay network configuration"
for networkID in $(${DOCKER} network ls --filter driver=overlay -q) ; do
    echo "Network ${networkID}"
    nspath=(${NSDIR}/*-$(echo ${networkID}| cut -c1-10))
    ${DOCKER} network inspect -v ${networkID}
    ${NSENTER} --net=${nspath[0]} ${BRIDGE} fdb show ${BRIDGEIF}
    ${NSENTER} --net=${nspath[0]} ${BRCTL} showmacs ${BRIDGEIF}
    echo ""
done
