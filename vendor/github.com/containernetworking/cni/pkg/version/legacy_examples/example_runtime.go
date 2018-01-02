// Copyright 2016 CNI authors
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

package legacy_examples

import (
	"fmt"
	"io/ioutil"
	"os"

	noop_debug "github.com/containernetworking/cni/plugins/test/noop/debug"
)

// An ExampleRuntime is a small program that uses libcni to invoke a network plugin.
// It should call ADD and DELETE, verifying all intermediate steps
// and data structures.
type ExampleRuntime struct {
	Example
	NetConfs []string // The network configuration names to pass
}

type exampleNetConfTemplate struct {
	conf   string
	result string
}

// NetConfs are various versioned network configuration files. Examples should
// specify which version they expect
var netConfTemplates = map[string]exampleNetConfTemplate{
	"unversioned": {
		conf: `{
	"name": "default",
	"type": "noop",
	"debugFile": "%s"
}`,
		result: `{
	"ip4": {
		"ip": "1.2.3.30/24",
		"gateway": "1.2.3.1",
		"routes": [
			{
				"dst": "15.5.6.0/24",
				"gw": "15.5.6.8"
			}
		]
	},
	"ip6": {
		"ip": "abcd:1234:ffff::cdde/64",
		"gateway": "abcd:1234:ffff::1",
		"routes": [
			{
				"dst": "1111:dddd::/80",
				"gw": "1111:dddd::aaaa"
			}
		]
	},
	"dns":{}
}`,
	},
	"0.1.0": {
		conf: `{
	"cniVersion": "0.1.0",
	"name": "default",
	"type": "noop",
	"debugFile": "%s"
}`,
		result: `{
	"cniVersion": "0.1.0",
	"ip4": {
		"ip": "1.2.3.30/24",
		"gateway": "1.2.3.1",
		"routes": [
			{
				"dst": "15.5.6.0/24",
				"gw": "15.5.6.8"
			}
		]
	},
	"ip6": {
		"ip": "abcd:1234:ffff::cdde/64",
		"gateway": "abcd:1234:ffff::1",
		"routes": [
			{
				"dst": "1111:dddd::/80",
				"gw": "1111:dddd::aaaa"
			}
		]
	},
	"dns":{}
}`,
	},
}

func (e *ExampleRuntime) GenerateNetConf(name string) (*ExampleNetConf, error) {
	template, ok := netConfTemplates[name]
	if !ok {
		return nil, fmt.Errorf("unknown example net config template %q", name)
	}

	debugFile, err := ioutil.TempFile("", "cni_debug")
	if err != nil {
		return nil, fmt.Errorf("failed to create noop plugin debug file: %v", err)
	}
	debugFilePath := debugFile.Name()

	debug := &noop_debug.Debug{
		ReportResult: template.result,
	}
	if err := debug.WriteDebug(debugFilePath); err != nil {
		os.Remove(debugFilePath)
		return nil, fmt.Errorf("failed to write noop plugin debug file %q: %v", debugFilePath, err)
	}
	conf := &ExampleNetConf{
		Config:        fmt.Sprintf(template.conf, debugFilePath),
		debugFilePath: debugFilePath,
	}

	return conf, nil
}

type ExampleNetConf struct {
	Config        string
	debugFilePath string
}

func (c *ExampleNetConf) Cleanup() {
	os.Remove(c.debugFilePath)
}

// V010_Runtime creates a simple noop network configuration, then
// executes libcni against the the noop test plugin.
var V010_Runtime = ExampleRuntime{
	NetConfs: []string{"unversioned", "0.1.0"},
	Example: Example{
		Name:          "example_invoker_v010",
		CNIRepoGitRef: "c0d34c69", //version with ns.Do
		PluginSource: `package main

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"

	"github.com/containernetworking/cni/pkg/ns"
	"github.com/containernetworking/cni/libcni"
)

func main(){
	code :=	exec()
	os.Exit(code)
}

func exec() int {
	confBytes, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		fmt.Printf("could not read netconfig from stdin: %+v", err)
		return 1
	}

	netConf, err := libcni.ConfFromBytes(confBytes)
	if err != nil {
		fmt.Printf("could not parse netconfig: %+v", err)
		return 1
	}
	fmt.Printf("Parsed network configuration: %+v\n", netConf.Network)

	if len(os.Args) == 1 {
		fmt.Printf("Expect CNI plugin paths in argv")
		return 1
	}

	targetNs, err := ns.NewNS()
	if err !=  nil {
		fmt.Printf("Could not create ns: %+v", err)
		return 1
	}
	defer targetNs.Close()

	ifName := "eth0"

	runtimeConf := &libcni.RuntimeConf{
		ContainerID: "some-container-id",
		NetNS:       targetNs.Path(),
		IfName:      ifName,
	}

	cniConfig := &libcni.CNIConfig{Path: os.Args[1:]}

	result, err := cniConfig.AddNetwork(netConf, runtimeConf)
	if err != nil {
		fmt.Printf("AddNetwork failed: %+v", err)
		return 2
	}
	fmt.Printf("AddNetwork result: %+v", result)

	// Validate expected results
	const expectedIP4 string = "1.2.3.30/24"
	if result.IP4.IP.String() != expectedIP4 {
		fmt.Printf("Expected IPv4 address %q, got %q", expectedIP4, result.IP4.IP.String())
		return 3
	}
	const expectedIP6 string = "abcd:1234:ffff::cdde/64"
	if result.IP6.IP.String() != expectedIP6 {
		fmt.Printf("Expected IPv6 address %q, got %q", expectedIP6, result.IP6.IP.String())
		return 4
	}

	err = cniConfig.DelNetwork(netConf, runtimeConf)
	if err != nil {
		fmt.Printf("DelNetwork failed: %v", err)
		return 5
	}

	err = targetNs.Do(func(ns.NetNS) error {
		_, err := net.InterfaceByName(ifName)
		if err == nil {
			return fmt.Errorf("interface was not deleted")
		}
		return nil
	})
	if err != nil {
		fmt.Println(err)
		return 6
	}

	return 0
}
`,
	},
}
