// Copyright 2015 The rkt Authors
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

package networking

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"strings"
	"syscall"

	"github.com/appc/cni/pkg/ns"
	cnitypes "github.com/appc/cni/pkg/types"
	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"
	"github.com/vishvananda/netlink"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/networking/netinfo"
	"github.com/coreos/rkt/pkg/log"
)

const (
	IfNamePattern = "eth%d"
	selfNetNS     = "/proc/self/ns/net"
)

// ForwardedPort describes a port that will be
// forwarded (mapped) from the host to the pod
type ForwardedPort struct {
	Protocol string
	HostPort uint
	PodPort  uint
}

// Networking describes the networking details of a pod.
type Networking struct {
	podEnv

	hostNS *os.File
	nets   []activeNet
}

// NetConf local struct extends cnitypes.NetConf with information about masquerading
// similar to CNI plugins
type NetConf struct {
	cnitypes.NetConf
	IPMasq bool `json:"ipMasq"`
	MTU    int  `json:"mtu"`
}

var stderr *log.Logger

// Setup creates a new networking namespace and executes network plugins to
// set up networking. It returns in the new pod namespace
func Setup(podRoot string, podID types.UUID, fps []ForwardedPort, netList common.NetList, localConfig, flavor string, debug bool) (*Networking, error) {

	stderr = log.New(os.Stderr, "networking", debug)

	if flavor == "kvm" {
		return kvmSetup(podRoot, podID, fps, netList, localConfig)
	}

	// TODO(jonboulle): currently podRoot is _always_ ".", and behaviour in other
	// circumstances is untested. This should be cleaned up.
	n := Networking{
		podEnv: podEnv{
			podRoot:      podRoot,
			podID:        podID,
			netsLoadList: netList,
			localConfig:  localConfig,
		},
	}

	hostNS, podNS, err := basicNetNS()
	if err != nil {
		return nil, err
	}
	// we're in podNS!
	n.hostNS = hostNS

	nspath := n.podNSPath()

	if err = bindMountFile(selfNetNS, nspath); err != nil {
		return nil, err
	}

	defer func() {
		if err != nil {
			if err := syscall.Unmount(nspath, 0); err != nil {
				stderr.PrintE(fmt.Sprintf("error unmounting %q", nspath), err)
			}
		}
	}()

	n.nets, err = n.loadNets()
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error loading network definitions"), err)
	}

	err = withNetNS(podNS, hostNS, func() error {
		if err := n.setupNets(n.nets); err != nil {
			return err
		}
		if len(fps) > 0 {
			if err = n.enableDefaultLocalnetRouting(); err != nil {
				return err
			}
			if err := n.forwardPorts(fps, n.GetDefaultIP()); err != nil {
				n.unforwardPorts()
				return err
			}
			return nil
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return &n, nil
}

// enableDefaultLocalnetRouting enables the route_localnet attribute on the supposedly default network interface.
// This allows setting up loopback NAT so the host can access the pod's forwarded ports on the localhost address.
func (n *Networking) enableDefaultLocalnetRouting() error {
	routeLocalnetFormat := ""

	defaultHostIP, err := n.GetDefaultHostIP()
	if err != nil {
		return err
	}

	defaultHostIPstring := defaultHostIP.String()
	switch {
	case strings.Contains(defaultHostIPstring, "."):
		routeLocalnetFormat = "/proc/sys/net/ipv4/conf/%s/route_localnet"
	case strings.Contains(defaultHostIPstring, ":"):
		return fmt.Errorf("unexpected IPv6 Address returned for default host interface: %q", defaultHostIPstring)
	default:
		return fmt.Errorf("unknown type for default Host IP: %q", defaultHostIPstring)
	}

	hostIfaces, err := n.GetIfacesByIP(defaultHostIP)
	if err != nil {
		return err
	}

	for _, hostIface := range hostIfaces {
		routeLocalnetPath := fmt.Sprintf(routeLocalnetFormat, hostIface.Name)
		routeLocalnetValue, err := ioutil.ReadFile(routeLocalnetPath)
		if err != nil {
			return err
		}
		if string(routeLocalnetValue) != "1" {
			routeLocalnetFile, err := os.OpenFile(routeLocalnetPath, os.O_WRONLY, 0)
			if err != nil {
				return err
			}
			defer routeLocalnetFile.Close()

			if _, err = io.WriteString(routeLocalnetFile, "1"); err != nil {
				return err
			}
		}
	}

	return nil
}

// Load creates the Networking object from saved state.
// Assumes the current netns is that of the host.
func Load(podRoot string, podID *types.UUID) (*Networking, error) {
	// the current directory is pod root
	pdirfd, err := syscall.Open(podRoot, syscall.O_RDONLY|syscall.O_DIRECTORY, 0)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("failed to open pod root directory (%v)", podRoot), err)
	}
	defer syscall.Close(pdirfd)

	nis, err := netinfo.LoadAt(pdirfd)
	if err != nil {
		return nil, err
	}

	hostNS, err := os.Open(selfNetNS)
	if err != nil {
		return nil, err
	}

	var nets []activeNet
	for _, ni := range nis {
		n, err := loadNet(ni.ConfPath)
		if err != nil {
			if !os.IsNotExist(err) {
				stderr.PrintE(fmt.Sprintf("error loading %q; ignoring", ni.ConfPath), err)
			}
			continue
		}

		// make a copy of ni to make it a unique object as it's saved via ptr
		rti := ni
		n.runtime = &rti
		nets = append(nets, *n)
	}

	return &Networking{
		podEnv: podEnv{
			podRoot: podRoot,
			podID:   *podID,
		},
		hostNS: hostNS,
		nets:   nets,
	}, nil
}

func (n *Networking) GetDefaultIP() net.IP {
	if len(n.nets) == 0 {
		return nil
	}
	return n.nets[len(n.nets)-1].runtime.IP
}

func (n *Networking) GetDefaultHostIP() (net.IP, error) {
	if len(n.nets) == 0 {
		return nil, fmt.Errorf("no networks found")
	}
	return n.nets[len(n.nets)-1].runtime.HostIP, nil
}

// GetIfacesByIP searches for and returns the interfaces with the given IP
// Disregards the subnet mask since not every net.IP object contains
// On success it will return the list of found interfaces
func (n *Networking) GetIfacesByIP(ifaceIP net.IP) ([]net.Interface, error) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil, err
	}

	searchAddr := strings.Split(ifaceIP.String(), "/")[0]
	resultInterfaces := make([]net.Interface, 0)

	for _, iface := range ifaces {
		if iface.Flags&net.FlagLoopback != 0 {
			continue
		}

		addrs, err := iface.Addrs()
		if err != nil {
			return nil, errwrap.Wrap(fmt.Errorf("cannot get addresses for interface %v", iface.Name), err)
		}

		for _, addr := range addrs {
			currentAddr := strings.Split(addr.String(), "/")[0]
			if searchAddr == currentAddr {
				resultInterfaces = append(resultInterfaces, iface)
				break
			}
		}
	}

	if len(resultInterfaces) == 0 {
		return nil, fmt.Errorf("no interface found with IP %q", ifaceIP)
	}

	return resultInterfaces, nil
}

// Teardown cleans up a produced Networking object.
func (n *Networking) Teardown(flavor string, debug bool) {

	stderr = log.New(os.Stderr, "networking", debug)

	// Teardown everything in reverse order of setup.
	// This should be idempotent -- be tolerant of missing stuff

	if flavor == "kvm" {
		n.kvmTeardown()
		return
	}

	if err := n.enterHostNS(); err != nil {
		stderr.PrintE("error switching to host netns", err)
		return
	}

	if err := n.unforwardPorts(); err != nil {
		stderr.PrintE("error removing forwarded ports", err)
	}

	n.teardownNets(n.nets)

	if err := syscall.Unmount(n.podNSPath(), 0); err != nil {
		// if already unmounted, umount(2) returns EINVAL
		if !os.IsNotExist(err) && err != syscall.EINVAL {
			stderr.PrintE(fmt.Sprintf("error unmounting %q", n.podNSPath()), err)
		}
	}
}

// sets up new netns with just lo
func basicNetNS() (hostNS, podNS *os.File, err error) {
	hostNS, podNS, err = newNetNS()
	if err != nil {
		err = errwrap.Wrap(errors.New("failed to create new netns"), err)
		return
	}
	// we're in podNS!!

	if err = loUp(); err != nil {
		hostNS.Close()
		podNS.Close()
		return nil, nil, err
	}

	return
}

// enterHostNS moves into the host's network namespace.
func (n *Networking) enterHostNS() error {
	return ns.SetNS(n.hostNS, syscall.CLONE_NEWNET)
}

// Save writes out the info about active nets
// for "rkt list" and friends to display
func (e *Networking) Save() error {
	var nis []netinfo.NetInfo
	for _, n := range e.nets {
		nis = append(nis, *n.runtime)
	}

	return netinfo.Save(e.podRoot, nis)
}

func newNetNS() (hostNS, childNS *os.File, err error) {
	defer func() {
		if err != nil {
			if hostNS != nil {
				hostNS.Close()
			}
			if childNS != nil {
				childNS.Close()
			}
		}
	}()

	hostNS, err = os.Open(selfNetNS)
	if err != nil {
		return
	}

	if err = syscall.Unshare(syscall.CLONE_NEWNET); err != nil {
		return
	}

	childNS, err = os.Open(selfNetNS)
	if err != nil {
		ns.SetNS(hostNS, syscall.CLONE_NEWNET)
		return
	}

	return
}

// execute f() in tgtNS
func withNetNS(curNS, tgtNS *os.File, f func() error) error {
	if err := ns.SetNS(tgtNS, syscall.CLONE_NEWNET); err != nil {
		return err
	}

	if err := f(); err != nil {
		// Attempt to revert the net ns in a known state
		if err := ns.SetNS(curNS, syscall.CLONE_NEWNET); err != nil {
			stderr.PrintE("cannot revert the net namespace", err)
		}
		return err
	}

	return ns.SetNS(curNS, syscall.CLONE_NEWNET)
}

func loUp() error {
	lo, err := netlink.LinkByName("lo")
	if err != nil {
		return errwrap.Wrap(errors.New("failed to lookup lo"), err)
	}

	if err := netlink.LinkSetUp(lo); err != nil {
		return errwrap.Wrap(errors.New("failed to set lo up"), err)
	}

	return nil
}

func bindMountFile(src, dst string) error {
	// mount point has to be an existing file
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	f.Close()

	return syscall.Mount(src, dst, "none", syscall.MS_BIND, "")
}
