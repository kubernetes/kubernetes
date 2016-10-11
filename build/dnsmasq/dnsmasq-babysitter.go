package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"

	flag "github.com/spf13/pflag"
	"github.com/vishvananda/netlink"
)

const (
	defaultBridgeName = "dbr0"
	// Beware of overlap with docker0.
	defaultIPAddr = "169.254.120.1/30"
)

var (
	// TODO: Get this from a configmap
	flags      = flag.NewFlagSet("", flag.ContinueOnError)
	bridgeName = flags.String("bridge-name", defaultBridgeName, `dnsmasq bridge name. This bridge is created if it doesn't exist.`)
	bip        = flags.String("bridge-ip", defaultIPAddr, `IP address to assign to the bridge.`)
)

func deleteBridge(name string) error {
	link := netlink.NewLinkAttrs()
	link.Name = name
	br := &netlink.Bridge{link}
	return netlink.LinkDel(br)
}

func createBridge(name, ipAddr string) error {
	_, err := net.InterfaceByName(name)
	if err == nil {
		log.Printf("Bridge already exists, deleting")
		deleteBridge(name)
	}
	la := netlink.NewLinkAttrs()
	la.Name = name
	br := &netlink.Bridge{la}

	if err := netlink.LinkAdd(br); err != nil {
		return fmt.Errorf("Failed to create bridge %v", err)
	}
	log.Printf("Created bridge %+v", br.Name)

	addr, err := netlink.ParseAddr(ipAddr)
	if err != nil {
		return fmt.Errorf("parse address %s: %v", ipAddr, err)
	}
	if err := netlink.AddrAdd(br, addr); err != nil {
		fmt.Errorf("Failed to add address %+v to bridge", addr)
	}
	log.Printf("Assigned ip address %v to bridge %v", addr, br.Name)

	if err := netlink.LinkSetUp(br); err != nil {
		return fmt.Errorf("Failed to up bridge: %v", err)
	}
	log.Printf("Bridge %v is up: %v", br.Name, addr)
	return nil
}

func shellOut(cmd string) {
	out, err := exec.Command("sh", "-c", cmd).CombinedOutput()
	if err != nil {
		log.Fatalf("Failed to execute %v: %v, err: %v", cmd, string(out), err)
	}
}

func main() {
	flags.Parse(os.Args)
	log.Printf("Creating bridge %v", *bridgeName)
	if err := createBridge(*bridgeName, *bip); err != nil {
		log.Fatalf("failed to create bridge %v", err)
	}
	log.Printf("Starting dnsmasq")
	shellOut("dnsmasq -k -x /var/run/dnsmasq.pid")
	// TODO: Reload dnsmasq on config map update.

}
