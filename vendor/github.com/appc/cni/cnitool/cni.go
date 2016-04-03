// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/appc/cni/libcni"
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
	netconf, err := libcni.LoadConf(netdir, os.Args[2])
	if err != nil {
		exit(err)
	}

	netns := os.Args[3]

	cninet := &libcni.CNIConfig{
		Path: strings.Split(os.Getenv(EnvCNIPath), ":"),
	}

	rt := &libcni.RuntimeConf{
		ContainerID: "cni",
		NetNS:       netns,
		IfName:      "eth0",
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
