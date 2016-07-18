// Copyright 2015 flannel authors
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

package network

import (
	"errors"
	"fmt"
	"sync"
	"time"

	log "github.com/golang/glog"
	"golang.org/x/net/context"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/subnet"
)

const (
	renewMargin = time.Hour
)

var (
	errInterrupted = errors.New("interrupted")
	errCanceled    = errors.New("canceled")
)

type Network struct {
	Name   string
	Config *subnet.Config

	ctx        context.Context
	cancelFunc context.CancelFunc
	sm         subnet.Manager
	bm         backend.Manager
	ipMasq     bool
	bn         backend.Network
}

func NewNetwork(ctx context.Context, sm subnet.Manager, bm backend.Manager, name string, ipMasq bool) *Network {
	ctx, cf := context.WithCancel(ctx)

	return &Network{
		Name:       name,
		sm:         sm,
		bm:         bm,
		ipMasq:     ipMasq,
		ctx:        ctx,
		cancelFunc: cf,
	}
}

func wrapError(desc string, err error) error {
	if err == context.Canceled {
		return err
	}
	return fmt.Errorf("failed to %v: %v", desc, err)
}

func (n *Network) init() error {
	var err error

	n.Config, err = n.sm.GetNetworkConfig(n.ctx, n.Name)
	if err != nil {
		return wrapError("retrieve network config", err)
	}

	be, err := n.bm.GetBackend(n.Config.BackendType)
	if err != nil {
		return wrapError("create and initialize network", err)
	}

	n.bn, err = be.RegisterNetwork(n.ctx, n.Name, n.Config)
	if err != nil {
		return wrapError("register network", err)
	}

	if n.ipMasq {
		err = setupIPMasq(n.Config.Network)
		if err != nil {
			return wrapError("set up IP Masquerade", err)
		}
	}

	return nil
}

func (n *Network) retryInit() error {
	for {
		err := n.init()
		if err == nil || err == context.Canceled {
			return err
		}

		log.Error(err)

		select {
		case <-n.ctx.Done():
			return n.ctx.Err()
		case <-time.After(time.Second):
		}
	}
}

func (n *Network) runOnce(extIface *backend.ExternalInterface, inited func(bn backend.Network)) error {
	if err := n.retryInit(); err != nil {
		return errCanceled
	}

	inited(n.bn)

	ctx, interruptFunc := context.WithCancel(n.ctx)

	wg := sync.WaitGroup{}

	wg.Add(1)
	go func() {
		n.bn.Run(ctx)
		wg.Done()
	}()

	evts := make(chan subnet.Event)

	wg.Add(1)
	go func() {
		subnet.WatchLease(ctx, n.sm, n.Name, n.bn.Lease().Subnet, evts)
		wg.Done()
	}()

	defer func() {
		if n.ipMasq {
			if err := teardownIPMasq(n.Config.Network); err != nil {
				log.Errorf("Failed to tear down IP Masquerade for network %v: %v", n.Name, err)
			}
		}
	}()

	defer wg.Wait()

	dur := n.bn.Lease().Expiration.Sub(time.Now()) - renewMargin
	for {
		select {
		case <-time.After(dur):
			err := n.sm.RenewLease(n.ctx, n.Name, n.bn.Lease())
			if err != nil {
				log.Error("Error renewing lease (trying again in 1 min): ", err)
				dur = time.Minute
				continue
			}

			log.Info("Lease renewed, new expiration: ", n.bn.Lease().Expiration)
			dur = n.bn.Lease().Expiration.Sub(time.Now()) - renewMargin

		case e := <-evts:
			switch e.Type {
			case subnet.EventAdded:
				n.bn.Lease().Expiration = e.Lease.Expiration
				dur = n.bn.Lease().Expiration.Sub(time.Now()) - renewMargin

			case subnet.EventRemoved:
				log.Warning("Lease has been revoked")
				interruptFunc()
				return errInterrupted
			}

		case <-n.ctx.Done():
			return errCanceled
		}
	}
}

func (n *Network) Run(extIface *backend.ExternalInterface, inited func(bn backend.Network)) {
	for {
		switch n.runOnce(extIface, inited) {
		case errInterrupted:

		case errCanceled:
			return
		default:
			panic("unexpected error returned")
		}
	}
}

func (n *Network) Cancel() {
	n.cancelFunc()
}
