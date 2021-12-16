package utils

import "net"

func IsIPv4(ip net.IP) bool {
	// If ip is not an IPv4 address, To4 returns nil.
	// Note that there might be some corner cases, where this is not correct.
	// See https://stackoverflow.com/questions/22751035/golang-distinguish-ipv4-ipv6.
	return ip.To4() != nil
}
