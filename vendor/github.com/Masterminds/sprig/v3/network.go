package sprig

import (
	"math/rand"
	"net"
)

func getHostByName(name string) string {
	addrs, _ := net.LookupHost(name)
	//TODO: add error handing when release v3 cames out
	return addrs[rand.Intn(len(addrs))]
}
