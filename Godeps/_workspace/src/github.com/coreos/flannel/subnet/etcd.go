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

package subnet

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"path"
	"regexp"
	"strconv"
	"time"

	etcd "github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/etcd/client"
	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"
	"github.com/coreos/flannel/pkg/ip"
	"log"
)

const (
	registerRetries = 10
	subnetTTL       = 24 * time.Hour
)

type EtcdManager struct {
	registry     Registry
	networkRegex *regexp.Regexp
}

var (
	subnetRegex *regexp.Regexp = regexp.MustCompile(`(\d+\.\d+.\d+.\d+)-(\d+)`)
)

type watchCursor struct {
	index uint64
}

func (c watchCursor) String() string {
	return strconv.FormatUint(c.index, 10)
}

func NewEtcdManager(config *EtcdConfig) (Manager, error) {
	r, err := newEtcdSubnetRegistry(config)
	if err != nil {
		return nil, err
	}
	return &EtcdManager{
		registry:     r,
		networkRegex: regexp.MustCompile(config.Prefix + `/([^/]*)/config`),
	}, nil
}

func newEtcdManager(r Registry) Manager {
	return &EtcdManager{
		registry:     r,
		networkRegex: regexp.MustCompile(`/coreos.com/network/([^/]*)/config`),
	}
}

func (m *EtcdManager) GetNetworkConfig(ctx context.Context, network string) (*Config, error) {
	cfgResp, err := m.registry.getNetworkConfig(ctx, network)
	if err != nil {
		return nil, err
	}

	return ParseConfig(cfgResp.Node.Value)
}

func (m *EtcdManager) AcquireLease(ctx context.Context, network string, attrs *LeaseAttrs) (*Lease, error) {
	config, err := m.GetNetworkConfig(ctx, network)
	if err != nil {
		return nil, err
	}

	for {
		l, err := m.acquireLeaseOnce(ctx, network, config, attrs)
		switch {
		case err == nil:
			log.Printf("Subnet lease acquired: ", l.Subnet)
			return l, nil

		case err == context.Canceled, err == context.DeadlineExceeded:
			return nil, err

		default:
			log.Printf("Failed to acquire subnet: ", err)
		}

		select {
		case <-time.After(time.Second):

		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func findLeaseByIP(leases []Lease, pubIP ip.IP4) *Lease {
	for _, l := range leases {
		if pubIP == l.Attrs.PublicIP {
			return &l
		}
	}

	return nil
}

func (m *EtcdManager) tryAcquireLease(ctx context.Context, network string, config *Config, extIaddr ip.IP4, attrs *LeaseAttrs) (*Lease, error) {
	var err error
	leases, _, err := m.getLeases(ctx, network)
	if err != nil {
		return nil, err
	}

	attrBytes, err := json.Marshal(attrs)
	if err != nil {
		return nil, err
	}

	// try to reuse a subnet if there's one that matches our IP
	if l := findLeaseByIP(leases, extIaddr); l != nil {
		// make sure the existing subnet is still within the configured network
		if isSubnetConfigCompat(config, l.Subnet) {
			log.Printf("Found lease (%v) for current IP (%v), reusing", l.Subnet, extIaddr)
			resp, err := m.registry.updateSubnet(ctx, network, l.Key(), string(attrBytes), subnetTTL)
			if err != nil {
				return nil, err
			}

			l.Attrs = attrs
			l.Expiration = *resp.Node.Expiration
			return l, nil
		} else {
			log.Printf("Found lease (%v) for current IP (%v) but not compatible with current config, deleting", l.Subnet, extIaddr)
			if _, err := m.registry.deleteSubnet(ctx, network, l.Key()); err != nil {
				return nil, err
			}
		}
	}

	// no existing match, grab a new one
	sn, err := m.allocateSubnet(config, leases)
	if err != nil {
		return nil, err
	}

	resp, err := m.registry.createSubnet(ctx, network, sn.StringSep(".", "-"), string(attrBytes), subnetTTL)
	if err == nil {
		return &Lease{
			Subnet:     sn,
			Attrs:      attrs,
			Expiration: *resp.Node.Expiration,
		}, nil
	}

	if etcdErr, ok := err.(etcd.Error); ok && etcdErr.Code == etcd.ErrorCodeNodeExist {
		// if etcd returned Key Already Exists, try again.
		return nil, nil
	}

	return nil, err
}

func (m *EtcdManager) acquireLeaseOnce(ctx context.Context, network string, config *Config, attrs *LeaseAttrs) (*Lease, error) {
	for i := 0; i < registerRetries; i++ {
		l, err := m.tryAcquireLease(ctx, network, config, attrs.PublicIP, attrs)
		switch {
		case err != nil:
			return nil, err
		case l != nil:
			return l, nil
		}

		// before moving on, check for cancel
		// TODO(eyakubovich): propogate ctx deeper into registry
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}

	return nil, errors.New("Max retries reached trying to acquire a subnet")
}

func parseSubnetKey(s string) (ip.IP4Net, error) {
	if parts := subnetRegex.FindStringSubmatch(s); len(parts) == 3 {
		snIp := net.ParseIP(parts[1]).To4()
		prefixLen, err := strconv.ParseUint(parts[2], 10, 5)
		if snIp != nil && err == nil {
			return ip.IP4Net{IP: ip.FromIP(snIp), PrefixLen: uint(prefixLen)}, nil
		}
	}

	return ip.IP4Net{}, errors.New("Error parsing IP Subnet")
}

func (m *EtcdManager) allocateSubnet(config *Config, leases []Lease) (ip.IP4Net, error) {
	log.Printf("Picking subnet in range %s ... %s", config.SubnetMin, config.SubnetMax)

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

// getLeases queries etcd to get a list of currently allocated leases for a given network.
// It returns the leases along with the "as-of" etcd-index that can be used as the starting
// point for etcd watch.
func (m *EtcdManager) getLeases(ctx context.Context, network string) ([]Lease, uint64, error) {
	resp, err := m.registry.getSubnets(ctx, network)

	leases := []Lease{}

	if err == nil {
		for _, node := range resp.Node.Nodes {
			sn, err := parseSubnetKey(node.Key)
			if err == nil {
				attrs := &LeaseAttrs{}
				if err = json.Unmarshal([]byte(node.Value), attrs); err == nil {
					exp := time.Time{}
					if node.Expiration != nil {
						exp = *node.Expiration
					}

					lease := Lease{
						Subnet:     sn,
						Attrs:      attrs,
						Expiration: exp,
					}
					leases = append(leases, lease)
				}
			}
		}

		return leases, resp.Index, nil
	}

	if etcdErr, ok := err.(etcd.Error); ok && etcdErr.Code == etcd.ErrorCodeKeyNotFound {
		// key not found: treat it as empty set
		return leases, etcdErr.Index, nil
	}

	return nil, 0, err
}

func (m *EtcdManager) RenewLease(ctx context.Context, network string, lease *Lease) error {
	attrBytes, err := json.Marshal(lease.Attrs)
	if err != nil {
		return err
	}

	// TODO(eyakubovich): propogate ctx into registry
	resp, err := m.registry.updateSubnet(ctx, network, lease.Key(), string(attrBytes), subnetTTL)
	if err != nil {
		return err
	}

	lease.Expiration = *resp.Node.Expiration
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

func (m *EtcdManager) WatchLeases(ctx context.Context, network string, cursor interface{}) (LeaseWatchResult, error) {
	if cursor == nil {
		return m.leaseWatchReset(ctx, network)
	}

	nextIndex, err := getNextIndex(cursor)
	if err != nil {
		return LeaseWatchResult{}, err
	}

	resp, err := m.registry.watch(ctx, path.Join(network, "subnets"), nextIndex)

	switch {
	case err == nil:
		return parseSubnetWatchResponse(resp)

	case isIndexTooSmall(err):
		log.Printf("Watch of subnet leases failed because etcd index outside history window")
		return m.leaseWatchReset(ctx, network)

	default:
		return LeaseWatchResult{}, err
	}
}

func (m *EtcdManager) WatchNetworks(ctx context.Context, cursor interface{}) (NetworkWatchResult, error) {
	if cursor == nil {
		return m.networkWatchReset(ctx)
	}

	nextIndex, err := getNextIndex(cursor)
	if err != nil {
		return NetworkWatchResult{}, err
	}

	resp, err := m.registry.watch(ctx, "", nextIndex)

	switch {
	case err == nil:
		return m.parseNetworkWatchResponse(resp)

	case isIndexTooSmall(err):
		log.Printf("Watch of subnet leases failed because etcd index outside history window")
		return m.networkWatchReset(ctx)

	default:
		return NetworkWatchResult{}, err
	}
}

func isIndexTooSmall(err error) bool {
	etcdErr, ok := err.(etcd.Error)
	return ok && etcdErr.Code == etcd.ErrorCodeEventIndexCleared
}

func parseSubnetWatchResponse(resp *etcd.Response) (LeaseWatchResult, error) {
	sn, err := parseSubnetKey(resp.Node.Key)
	if err != nil {
		return LeaseWatchResult{}, fmt.Errorf("error parsing subnet IP: %s", resp.Node.Key)
	}

	evt := Event{}

	switch resp.Action {
	case "delete", "expire":
		evt = Event{
			EventRemoved,
			Lease{Subnet: sn},
			"",
		}

	default:
		attrs := &LeaseAttrs{}
		err := json.Unmarshal([]byte(resp.Node.Value), attrs)
		if err != nil {
			return LeaseWatchResult{}, err
		}

		exp := time.Time{}
		if resp.Node.Expiration != nil {
			exp = *resp.Node.Expiration
		}

		evt = Event{
			EventAdded,
			Lease{
				Subnet:     sn,
				Attrs:      attrs,
				Expiration: exp,
			},
			"",
		}
	}

	return LeaseWatchResult{
		Cursor: watchCursor{resp.Node.ModifiedIndex},
		Events: []Event{evt},
	}, nil
}

// Returns network name from config key (eg, /coreos.com/network/foobar/config),
// if the 'config' key isn't present we don't consider the network valid
func (m *EtcdManager) parseNetworkKey(s string) (string, error) {
	if parts := m.networkRegex.FindStringSubmatch(s); len(parts) == 2 {
		return parts[1], nil
	}

	return "", errors.New("Error parsing Network key")
}

func (m *EtcdManager) parseNetworkWatchResponse(resp *etcd.Response) (NetworkWatchResult, error) {
	netname, err := m.parseNetworkKey(resp.Node.Key)
	if err != nil {
		// Ignore non .../<netname>/config keys
		return NetworkWatchResult{}, nil
	}

	evt := Event{}

	switch resp.Action {
	case "delete":
		evt = Event{
			EventRemoved,
			Lease{},
			netname,
		}

	default:
		_, err := ParseConfig(resp.Node.Value)
		if err != nil {
			return NetworkWatchResult{}, err
		}

		evt = Event{
			EventAdded,
			Lease{},
			netname,
		}
	}

	return NetworkWatchResult{
		Cursor: watchCursor{resp.Node.ModifiedIndex},
		Events: []Event{evt},
	}, nil
}

// getNetworks queries etcd to get a list of network names.  It returns the
// networks along with the 'as-of' etcd-index that can be used as the starting
// point for etcd watch.
func (m *EtcdManager) getNetworks(ctx context.Context) ([]string, uint64, error) {
	resp, err := m.registry.getNetworks(ctx)

	networks := []string{}

	if err == nil {
		for _, node := range resp.Node.Nodes {
			netname, err := m.parseNetworkKey(node.Key)
			if err == nil {
				networks = append(networks, netname)
			}
		}

		return networks, resp.Index, nil
	}

	if etcdErr, ok := err.(etcd.Error); ok && etcdErr.Code == etcd.ErrorCodeKeyNotFound {
		// key not found: treat it as empty set
		return networks, etcdErr.Index, nil
	}

	return nil, 0, err
}

// leaseWatchReset is called when incremental lease watch failed and we need to grab a snapshot
func (m *EtcdManager) leaseWatchReset(ctx context.Context, network string) (LeaseWatchResult, error) {
	wr := LeaseWatchResult{}

	leases, index, err := m.getLeases(ctx, network)
	if err != nil {
		return wr, fmt.Errorf("failed to retrieve subnet leases: %v", err)
	}

	wr.Cursor = watchCursor{index}
	wr.Snapshot = leases
	return wr, nil
}

// networkWatchReset is called when incremental network watch failed and we need to grab a snapshot
func (m *EtcdManager) networkWatchReset(ctx context.Context) (NetworkWatchResult, error) {
	wr := NetworkWatchResult{}

	networks, index, err := m.getNetworks(ctx)
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
