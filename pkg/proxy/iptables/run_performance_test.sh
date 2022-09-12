#!/bin/sh

# Create test namespace
sudo ip netns del test-iptables
sudo ip netns add test-iptables
sudo ip netns exec test-iptables ip link set up dev lo

sudo ip link add name vethIn type veth peer name vethOut
sudo ip link set netns test-iptables dev vethIn
sudo ip link set up dev vethOut
sudo ip netns exec test-iptables ip link set up dev vethIn
sudo ip netns exec test-iptables ip addr add 1.1.1.1/32 dev vethIn
sudo ip netns exec test-iptables ip route add default dev vethIn

# compile test
go test -run ^TestIPTablesRestore$ k8s.io/kubernetes/pkg/proxy/iptables -c -v
sudo ip netns exec test-iptables ./iptables.test -test.run ^TestIPTablesRestore$
