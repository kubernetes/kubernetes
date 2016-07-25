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

package subnet

import (
	"errors"
	"fmt"
	"strconv"
	"time"

	etcd "github.com/coreos/etcd/client"
	"github.com/coreos/flannel/pkg/ip"
	log "github.com/golang/glog"
	"golang.org/x/net/context"
)

const (
	raceRetries = 10
	subnetTTL   = 24 * time.Hour
)

type LocalManager struct {
	registry Registry
}

type watchCursor struct {
	index uint64
}

func isErrEtcdTestFailed(e error) bool {
	if e == nil {
		return false
	}
	etcdErr, ok := e.(etcd.Error)
	return ok && etcdErr.Code == etcd.ErrorCodeTestFailed
}

func isErrEtcdNodeExist(e error) bool {
	if e == nil {
		return false
	}
	etcdErr, ok := e.(etcd.Error)
	return ok || etcdErr.Code == etcd.ErrorCodeNodeExist
}

func isErrEtcdKeyNotFound(e error) bool {
	if e == nil {
		return false
	}
	etcdErr, ok := e.(etcd.Error)
	return ok || etcdErr.Code == etcd.ErrorCodeKeyNotFound
}

func (c watchCursor) String() string {
	return strconv.FormatUint(c.index, 10)
}

func NewLocalManager(config *EtcdConfig) (Manager, error) {
	r, err := newEtcdSubnetRegistry(config, nil)
	if err != nil {
		return nil, err
	}
	return newLocalManager(r), nil
}

func newLocalManager(r Registry) Manager {
	return &LocalManager{
		registry: r,
	}
}

func (m *LocalManager) GetNetworkConfig(ctx context.Context, network string) (*Config, error) {
	cfg, err := m.registry.getNetworkConfig(ctx, network)
	if err != nil {
		return nil, err
	}

	return ParseConfig(cfg)
}

func (m *LocalManager) AcquireLease(ctx context.Context, network string, attrs *LeaseAttrs) (*Lease, error) {
	config, err := m.GetNetworkConfig(ctx, network)
	if err != nil {
		return nil, err
	}

	for i := 0; i < raceRetries; i++ {
		l, err := m.tryAcquireLease(ctx, network, config, attrs.PublicIP, attrs)
		switch err {
		case nil:
			return l, nil
		case errTryAgain:
			continue
		default:
			return nil, err
		}
	}

	return nil, errors.New("Max retries reached trying to acquire a subnet")
}

func findLeaseByIP(leases []Lease, pubIP ip.IP4) *Lease {
	for _, l := range leases {
		if pubIP == l.Attrs.PublicIP {
			return &l
		}
	}

	return nil
}

func (m *LocalManager) tryAcquireLease(ctx context.Context, network string, config *Config, extIaddr ip.IP4, attrs *LeaseAttrs) (*Lease, error) {
	leases, _, err := m.registry.getSubnets(ctx, network)
	if err != nil {
		return nil, err
	}

	// try to reuse a subnet if there's one that matches our IP
	if l := findLeaseByIP(leases, extIaddr); l != nil {
		// make sure the existing subnet is still within the configured network
		if isSubnetConfigCompat(config, l.Subnet) {
			log.Infof("Found lease (%v) for current IP (%v), reusing", l.Subnet, extIaddr)

			ttl := time.Duration(0)
			if !l.Expiration.IsZero() {
				// Not a reservation
				ttl = subnetTTL
			}
			exp, err := m.registry.updateSubnet(ctx, network, l.Subnet, attrs, ttl, 0)
			if err != nil {
				return nil, err
			}

			l.Attrs = *attrs
			l.Expiration = exp
			return l, nil
		} else {
			log.Infof("Found lease (%v) for current IP (%v) but not compatible with current config, deleting", l.Subnet, extIaddr)
			if err := m.registry.deleteSubnet(ctx, network, l.Subnet); err != nil {
				return nil, err
			}
		}
	}

	// no existing match, grab a new one
	sn, err := m.allocateSubnet(config, leases)
	if err != nil {
		return nil, err
	}

	exp, err := m.registry.createSubnet(ctx, network, sn, attrs, subnetTTL)
	switch {
	case err == nil:
		return &Lease{
			Subnet:     sn,
			Attrs:      *attrs,
			Expiration: exp,
		}, nil
	case isErrEtcdNodeExist(err):
		return nil, errTryAgain
	default:
		return nil, err
	}
}

func (m *LocalManager) allocateSubnet(config *Config, leases []Lease) (ip.IP4Net, error) {
	log.Infof("Picking subnet in range %s ... %s", config.SubnetMin, config.SubnetMax)

	var bag []ip.IP4
	sn := ip.IP4Net{IP: config.SubnetMin, PrefixLen: config.SubnetLen}

OuterLoop:
	for ; sn.IP <= config.SubnetMax && len(bag) < 100; sn = sn.Next() {
		for _, l := range leases {
			if sn.Overlaps(l.Subnet) {
				continue OuterLoop
			}
		}
		bag = append(bag, sn.IP)
	}

	if len(bag) == 0 {
		return ip.IP4Net{}, errors.New("out of subnets")
	} else {
		i := randInt(0, len(bag))
		return ip.IP4Net{IP: bag[i], PrefixLen: config.SubnetLen}, nil
	}
}

func (m *LocalManager) RevokeLease(ctx context.Context, network string, sn ip.IP4Net) error {
	return m.registry.deleteSubnet(ctx, network, sn)
}

func (m *LocalManager) RenewLease(ctx context.Context, network string, lease *Lease) error {
	exp, err := m.registry.updateSubnet(ctx, network, lease.Subnet, &lease.Attrs, subnetTTL, 0)
	if err != nil {
		return err
	}

	lease.Expiration = exp
	return nil
}

func getNextIndex(cursor interface{}) (uint64, error) {
	nextIndex := uint64(0)

	if wc, ok := cursor.(watchCursor); ok {
		nextIndex = wc.index
	} else if s, ok := cursor.(string); ok {
		var err error
		nextIndex, err = strconv.ParseUint(s, 10, 64)
		if err != nil {
			return 0, fmt.Errorf("failed to parse cursor: %v", err)
		}
	} else {
		return 0, fmt.Errorf("internal error: watch cursor is of unknown type")
	}

	return nextIndex, nil
}

func (m *LocalManager) leaseWatchReset(ctx context.Context, network string, sn ip.IP4Net) (LeaseWatchResult, error) {
	l, index, err := m.registry.getSubnet(ctx, network, sn)
	if err != nil {
		return LeaseWatchResult{}, err
	}

	return LeaseWatchResult{
		Snapshot: []Lease{*l},
		Cursor:   watchCursor{index},
	}, nil
}

func (m *LocalManager) WatchLease(ctx context.Context, network string, sn ip.IP4Net, cursor interface{}) (LeaseWatchResult, error) {
	if cursor == nil {
		return m.leaseWatchReset(ctx, network, sn)
	}

	nextIndex, err := getNextIndex(cursor)
	if err != nil {
		return LeaseWatchResult{}, err
	}

	evt, index, err := m.registry.watchSubnet(ctx, network, nextIndex, sn)

	switch {
	case err == nil:
		return LeaseWatchResult{
			Events: []Event{evt},
			Cursor: watchCursor{index},
		}, nil

	case isIndexTooSmall(err):
		log.Warning("Watch of subnet leases failed because etcd index outside history window")
		return m.leaseWatchReset(ctx, network, sn)

	default:
		return LeaseWatchResult{}, err
	}
}

func (m *LocalManager) WatchLeases(ctx context.Context, network string, cursor interface{}) (LeaseWatchResult, error) {
	if cursor == nil {
		return m.leasesWatchReset(ctx, network)
	}

	nextIndex, err := getNextIndex(cursor)
	if err != nil {
		return LeaseWatchResult{}, err
	}

	evt, index, err := m.registry.watchSubnets(ctx, network, nextIndex)

	switch {
	case err == nil:
		return LeaseWatchResult{
			Events: []Event{evt},
			Cursor: watchCursor{index},
		}, nil

	case isIndexTooSmall(err):
		log.Warning("Watch of subnet leases failed because etcd index outside history window")
		return m.leasesWatchReset(ctx, network)

	default:
		return LeaseWatchResult{}, err
	}
}

func (m *LocalManager) WatchNetworks(ctx context.Context, cursor interface{}) (NetworkWatchResult, error) {
	if cursor == nil {
		return m.networkWatchReset(ctx)
	}

	nextIndex, err := getNextIndex(cursor)
	if err != nil {
		return NetworkWatchResult{}, err
	}

	for {
		evt, index, err := m.registry.watchNetworks(ctx, nextIndex)

		switch {
		case err == nil:
			return NetworkWatchResult{
				Events: []Event{evt},
				Cursor: watchCursor{index},
			}, nil

		case err == errTryAgain:
			nextIndex = index

		case isIndexTooSmall(err):
			log.Warning("Watch of networks failed because etcd index outside history window")
			return m.networkWatchReset(ctx)

		default:
			return NetworkWatchResult{}, err
		}
	}
}

func isIndexTooSmall(err error) bool {
	etcdErr, ok := err.(etcd.Error)
	return ok && etcdErr.Code == etcd.ErrorCodeEventIndexCleared
}

// leasesWatchReset is called when incremental lease watch failed and we need to grab a snapshot
func (m *LocalManager) leasesWatchReset(ctx context.Context, network string) (LeaseWatchResult, error) {
	wr := LeaseWatchResult{}

	leases, index, err := m.registry.getSubnets(ctx, network)
	if err != nil {
		return wr, fmt.Errorf("failed to retrieve subnet leases: %v", err)
	}

	wr.Cursor = watchCursor{index}
	wr.Snapshot = leases
	return wr, nil
}

// networkWatchReset is called when incremental network watch failed and we need to grab a snapshot
func (m *LocalManager) networkWatchReset(ctx context.Context) (NetworkWatchResult, error) {
	wr := NetworkWatchResult{}

	networks, index, err := m.registry.getNetworks(ctx)
	if err != nil {
		return wr, fmt.Errorf("failed to retrieve networks: %v", err)
	}

	wr.Cursor = watchCursor{index}
	wr.Snapshot = networks
	return wr, nil
}

func isSubnetConfigCompat(config *Config, sn ip.IP4Net) bool {
	if sn.IP < config.SubnetMin || sn.IP > config.SubnetMax {
		return false
	}

	return sn.PrefixLen == config.SubnetLen
}

func (m *LocalManager) tryAddReservation(ctx context.Context, network string, r *Reservation) error {
	attrs := &LeaseAttrs{
		PublicIP: r.PublicIP,
	}

	_, err := m.registry.createSubnet(ctx, network, r.Subnet, attrs, 0)
	switch {
	case err == nil:
		return nil

	case !isErrEtcdNodeExist(err):
		return err
	}

	// This subnet or its reservation already exists.
	// Get what's there and
	// - if PublicIP matches, remove the TTL make it a reservation
	// - otherwise, error out
	sub, asof, err := m.registry.getSubnet(ctx, network, r.Subnet)
	switch {
	case err == nil:
	case isErrEtcdKeyNotFound(err):
		// Subnet just got expired or was deleted
		return errTryAgain
	default:
		return err
	}

	if sub.Attrs.PublicIP != r.PublicIP {
		// Subnet already taken
		return ErrLeaseTaken
	}

	// remove TTL
	_, err = m.registry.updateSubnet(ctx, network, r.Subnet, &sub.Attrs, 0, asof)
	if isErrEtcdTestFailed(err) {
		return errTryAgain
	}
	return err
}

func (m *LocalManager) AddReservation(ctx context.Context, network string, r *Reservation) error {
	config, err := m.GetNetworkConfig(ctx, network)
	if err != nil {
		return err
	}

	if config.SubnetLen != r.Subnet.PrefixLen {
		return fmt.Errorf("reservation subnet has mask incompatible with network config")
	}

	if !config.Network.Overlaps(r.Subnet) {
		return fmt.Errorf("reservation subnet is outside of flannel network")
	}

	for i := 0; i < raceRetries; i++ {
		err := m.tryAddReservation(ctx, network, r)
		switch {
		case err == nil:
			return nil
		case err == errTryAgain:
			continue
		default:
			return err
		}
	}

	return ErrNoMoreTries
}

func (m *LocalManager) tryRemoveReservation(ctx context.Context, network string, subnet ip.IP4Net) error {
	sub, asof, err := m.registry.getSubnet(ctx, network, subnet)
	if err != nil {
		return err
	}

	// add back the TTL
	_, err = m.registry.updateSubnet(ctx, network, subnet, &sub.Attrs, subnetTTL, asof)
	if isErrEtcdTestFailed(err) {
		return errTryAgain
	}
	return err
}

//RemoveReservation removes the subnet by setting TTL back to subnetTTL (24hours)
func (m *LocalManager) RemoveReservation(ctx context.Context, network string, subnet ip.IP4Net) error {
	for i := 0; i < raceRetries; i++ {
		err := m.tryRemoveReservation(ctx, network, subnet)
		switch {
		case err == nil:
			return nil
		case err == errTryAgain:
			continue
		default:
			return err
		}
	}

	return ErrNoMoreTries
}

func (m *LocalManager) ListReservations(ctx context.Context, network string) ([]Reservation, error) {
	subnets, _, err := m.registry.getSubnets(ctx, network)
	if err != nil {
		return nil, err
	}

	rsvs := []Reservation{}
	for _, sub := range subnets {
		// Reservations don't have TTL and so no expiration
		if !sub.Expiration.IsZero() {
			continue
		}

		r := Reservation{
			Subnet:   sub.Subnet,
			PublicIP: sub.Attrs.PublicIP,
		}
		rsvs = append(rsvs, r)
	}

	return rsvs, nil
}
