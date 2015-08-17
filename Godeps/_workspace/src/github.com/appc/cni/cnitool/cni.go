package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/appc/cni"
)

const (
	EnvCNIPath = "CNI_PATH"
	EnvNetDir  = "NETCONFPATH"

	DefaultNetDir = "/etc/cni/net.d"

	CmdAdd = "add"
	CmdDel = "del"
)

func main() {
	if len(os.Args) < 3 {
		usage()
		return
	}

	netdir := os.Getenv(EnvNetDir)
	if netdir == "" {
		netdir = DefaultNetDir
	}
	netconf, err := cni.LoadNetConf(netdir, os.Args[2])
	if err != nil {
		exit(err)
	}

	netns := os.Args[3]

	cninet := &cni.CNIConfig{
		Path: strings.Split(os.Getenv(EnvCNIPath), ":"),
	}

	rt := &cni.RuntimeConf{
		ContainerID: "cni",
		NetNS:       netns,
		IfName:      "eth0",
		Args:        "",
	}

	switch os.Args[1] {
	case CmdAdd:
		_, err := cninet.AddNetwork(netconf, rt)
		exit(err)
	case CmdDel:
		exit(cninet.DelNetwork(netconf, rt))
	}
}

func usage() {
	exe := filepath.Base(os.Args[0])

	fmt.Fprintf(os.Stderr, "%s: Add or remove network interfaces from a network namespace\n", exe)
	fmt.Fprintf(os.Stderr, "  %s %s <net> <netns>\n", exe, CmdAdd)
	fmt.Fprintf(os.Stderr, "  %s %s <net> <netns>\n", exe, CmdDel)
	os.Exit(1)
}

func exit(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
	os.Exit(0)
}
