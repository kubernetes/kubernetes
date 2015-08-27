#!/bin/sh
# goproxy IP
GOPROXY_SERVER="10.10.10.1"
# goproxy port
GOPROXY_PORT="3129"
GOPROXY_PORT_TLS="3128"
# DO NOT MODIFY BELOW
# Load IPTABLES modules for NAT and IP conntrack support
modprobe ip_conntrack
modprobe ip_conntrack_ftp
echo 1 > /proc/sys/net/ipv4/ip_forward
echo 2 > /proc/sys/net/ipv4/conf/all/rp_filter

# Clean old firewall
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Write new rules
iptables -t nat -A PREROUTING -s $GOPROXY_SERVER -p tcp --dport $GOPROXY_PORT -j ACCEPT
iptables -t nat -A PREROUTING -s $GOPROXY_SERVER -p tcp --dport $GOPROXY_PORT_TLS -j ACCEPT
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination $GOPROXY_SERVER:$GOPROXY_PORT
iptables -t nat -A PREROUTING -p tcp --dport 443 -j DNAT --to-destination $GOPROXY_SERVER:$GOPROXY_PORT_TLS
# The following line supports using goproxy as an explicit proxy in addition
iptables -t nat -A PREROUTING -p tcp --dport 8080 -j DNAT --to-destination $GOPROXY_SERVER:$GOPROXY_PORT
iptables -t nat -A POSTROUTING -j MASQUERADE
iptables -t mangle -A PREROUTING -p tcp --dport $GOPROXY_PORT -j DROP
iptables -t mangle -A PREROUTING -p tcp --dport $GOPROXY_PORT_TLS -j DROP
