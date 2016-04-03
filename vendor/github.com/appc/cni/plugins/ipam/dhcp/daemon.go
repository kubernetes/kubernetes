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
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/appc/cni/pkg/skel"
	"github.com/appc/cni/pkg/types"
	"github.com/coreos/go-systemd/activation"
)

const listenFdsStart = 3
const resendCount = 3

var errNoMoreTries = errors.New("no more tries")

type DHCP struct {
	mux    sync.Mutex
	leases map[string]*DHCPLease
}

func newDHCP() *DHCP {
	return &DHCP{
		leases: make(map[string]*DHCPLease),
	}
}

// Allocate acquires an IP from a DHCP server for a specified container.
// The acquired lease will be maintained until Release() is called.
func (d *DHCP) Allocate(args *skel.CmdArgs, result *types.Result) error {
	conf := types.NetConf{}
	if err := json.Unmarshal(args.StdinData, &conf); err != nil {
		return fmt.Errorf("error parsing netconf: %v", err)
	}

	clientID := args.ContainerID + "/" + conf.Name
	l, err := AcquireLease(clientID, args.Netns, args.IfName)
	if err != nil {
		return err
	}

	ipn, err := l.IPNet()
	if err != nil {
		l.Stop()
		return err
	}

	d.setLease(args.ContainerID, conf.Name, l)

	result.IP4 = &types.IPConfig{
		IP:      *ipn,
		Gateway: l.Gateway(),
		Routes:  l.Routes(),
	}

	return nil
}

// Release stops maintenance of the lease acquired in Allocate()
// and sends a release msg to the DHCP server.
func (d *DHCP) Release(args *skel.CmdArgs, reply *struct{}) error {
	conf := types.NetConf{}
	if err := json.Unmarshal(args.StdinData, &conf); err != nil {
		return fmt.Errorf("error parsing netconf: %v", err)
	}

	if l := d.getLease(args.ContainerID, conf.Name); l != nil {
		l.Stop()
		return nil
	}

	return fmt.Errorf("lease not found: %v/%v", args.ContainerID, conf.Name)
}

func (d *DHCP) getLease(contID, netName string) *DHCPLease {
	d.mux.Lock()
	defer d.mux.Unlock()

	// TODO(eyakubovich): hash it to avoid collisions
	l, ok := d.leases[contID+netName]
	if !ok {
		return nil
	}
	return l
}

func (d *DHCP) setLease(contID, netName string, l *DHCPLease) {
	d.mux.Lock()
	defer d.mux.Unlock()

	// TODO(eyakubovich): hash it to avoid collisions
	d.leases[contID+netName] = l
}

func getListener() (net.Listener, error) {
	l, err := activation.Listeners(true)
	if err != nil {
		return nil, err
	}

	switch {
	case len(l) == 0:
		if err := os.MkdirAll(filepath.Dir(socketPath), 0700); err != nil {
			return nil, err
		}
		return net.Listen("unix", socketPath)

	case len(l) == 1:
		if l[0] == nil {
			return nil, fmt.Errorf("LISTEN_FDS=1 but no FD found")
		}
		return l[0], nil

	default:
		return nil, fmt.Errorf("Too many (%v) FDs passed through socket activation", len(l))
	}
}

func runDaemon() {
	// since other goroutines (on separate threads) will change namespaces,
	// ensure the RPC server does not get scheduled onto those
	runtime.LockOSThread()

	l, err := getListener()
	if err != nil {
		log.Printf("Error getting listener: %v", err)
		return
	}

	dhcp := newDHCP()
	rpc.Register(dhcp)
	rpc.HandleHTTP()
	http.Serve(l, nil)
}
