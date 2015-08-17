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
	"crypto/sha512"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"runtime"

	"github.com/vishvananda/netlink"

	"github.com/appc/cni/pkg/ip"
	"github.com/appc/cni/pkg/ipam"
	"github.com/appc/cni/pkg/ns"
	"github.com/appc/cni/pkg/skel"
	"github.com/appc/cni/pkg/types"
)

func init() {
	// this ensures that main runs only on main thread (thread group leader).
	// since namespace ops (unshare, setns) are done for a single thread, we
	// must ensure that the goroutine does not jump from OS thread to thread
	runtime.LockOSThread()
}

type NetConf struct {
	types.NetConf
	IPMasq bool `json:"ipMasq"`
	MTU    int  `json:"mtu"`
}

func setupContainerVeth(netns, ifName string, mtu int, pr *types.Result) (string, error) {
	var hostVethName string
	err := ns.WithNetNSPath(netns, false, func(hostNS *os.File) error {
		hostVeth, _, err := ip.SetupVeth(ifName, mtu, hostNS)
		if err != nil {
			return err
		}

		err = ipam.ConfigureIface(ifName, pr)
		if err != nil {
			return err
		}

		hostVethName = hostVeth.Attrs().Name

		return nil
	})
	return hostVethName, err
}

func setupHostVeth(vethName string, ipConf *types.IPConfig) error {
	// hostVeth moved namespaces and may have a new ifindex
	veth, err := netlink.LinkByName(vethName)
	if err != nil {
		return fmt.Errorf("failed to lookup %q: %v", vethName, err)
	}

	// TODO(eyakubovich): IPv6
	ipn := &net.IPNet{
		IP:   ipConf.Gateway,
		Mask: net.CIDRMask(31, 32),
	}
	addr := &netlink.Addr{IPNet: ipn, Label: ""}
	if err = netlink.AddrAdd(veth, addr); err != nil {
		return fmt.Errorf("failed to add IP addr (%#v) to veth: %v", ipn, err)
	}

	// dst happens to be the same as IP/net of host veth
	if err = ip.AddHostRoute(ipn, nil, veth); err != nil && !os.IsExist(err) {
		return fmt.Errorf("failed to add route on host: %v", err)
	}

	return nil
}

func cmdAdd(args *skel.CmdArgs) error {
	conf := NetConf{}
	if err := json.Unmarshal(args.StdinData, &conf); err != nil {
		return fmt.Errorf("failed to load netconf: %v", err)
	}

	if err := ip.EnableIP4Forward(); err != nil {
		return fmt.Errorf("failed to enable forwarding: %v", err)
	}

	// run the IPAM plugin and get back the config to apply
	result, err := ipam.ExecAdd(conf.IPAM.Type, args.StdinData)
	if err != nil {
		return err
	}
	if result.IP4 == nil {
		return errors.New("IPAM plugin returned missing IPv4 config")
	}

	hostVethName, err := setupContainerVeth(args.Netns, args.IfName, conf.MTU, result)
	if err != nil {
		return err
	}

	if err = setupHostVeth(hostVethName, result.IP4); err != nil {
		return err
	}

	if conf.IPMasq {
		h := sha512.Sum512([]byte(args.ContainerID))
		chain := fmt.Sprintf("CNI-%s-%x", conf.Name, h[:8])
		if err = ip.SetupIPMasq(&result.IP4.IP, chain); err != nil {
			return err
		}
	}

	return result.Print()
}

func cmdDel(args *skel.CmdArgs) error {
	conf := NetConf{}
	if err := json.Unmarshal(args.StdinData, &conf); err != nil {
		return fmt.Errorf("failed to load netconf: %v", err)
	}

	var ipn *net.IPNet
	err := ns.WithNetNSPath(args.Netns, false, func(hostNS *os.File) error {
		var err error
		ipn, err = ip.DelLinkByNameAddr(args.IfName, netlink.FAMILY_V4)
		return err
	})
	if err != nil {
		return err
	}

	if conf.IPMasq {
		h := sha512.Sum512([]byte(args.ContainerID))
		chain := fmt.Sprintf("CNI-%s-%x", conf.Name, h[:8])
		if err = ip.TeardownIPMasq(ipn, chain); err != nil {
			return err
		}
	}

	return ipam.ExecDel(conf.IPAM.Type, args.StdinData)
}

func main() {
	skel.PluginMain(cmdAdd, cmdDel)
}
