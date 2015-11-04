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
	"fmt"
	"path"
	"sort"
	"strings"
	"time"

	etcd "github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/etcd/client"
	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"
)

type MockSubnetRegistry struct {
	networks map[string]*etcd.Node
	events   chan *etcd.Response
	index    uint64
	ttl      time.Duration
}

const networkKeyPrefix = "/coreos.com/network"

func NewMockRegistry(ttlOverride time.Duration, network, config string, initialSubnets []*etcd.Node) *MockSubnetRegistry {
	index := uint64(0)

	node := &etcd.Node{Key: normalizeNetKey(network), Value: config, ModifiedIndex: 0, Nodes: make([]*etcd.Node, 0, 20)}
	for _, n := range initialSubnets {
		if n.ModifiedIndex > index {
			index = n.ModifiedIndex
		}
		node.Nodes = append(node.Nodes, n)
	}

	msr := &MockSubnetRegistry{
		events: make(chan *etcd.Response, 1000),
		index:  index + 1,
		ttl:    ttlOverride,
	}

	msr.networks = make(map[string]*etcd.Node)
	msr.networks[network] = node
	return msr
}

func (msr *MockSubnetRegistry) getNetworkConfig(ctx context.Context, network string) (*etcd.Response, error) {
	return &etcd.Response{
		Index: msr.index,
		Node:  msr.networks[network],
	}, nil
}

func (msr *MockSubnetRegistry) setConfig(network, config string) error {
	n, ok := msr.networks[network]
	if !ok {
		return fmt.Errorf("Network %s not found", network)
	}
	n.Value = config
	return nil
}

func (msr *MockSubnetRegistry) getSubnets(ctx context.Context, network string) (*etcd.Response, error) {
	n, ok := msr.networks[network]
	if !ok {
		return nil, fmt.Errorf("Network %s not found", network)
	}
	return &etcd.Response{
		Node:  n,
		Index: msr.index,
	}, nil
}

func (msr *MockSubnetRegistry) createSubnet(ctx context.Context, network, sn, data string, ttl time.Duration) (*etcd.Response, error) {
	n, ok := msr.networks[network]
	if !ok {
		return nil, fmt.Errorf("Network %s not found", network)
	}

	msr.index += 1

	if msr.ttl > 0 {
		ttl = msr.ttl
	}

	exp := time.Now().Add(ttl)

	node := &etcd.Node{
		Key:           sn,
		Value:         data,
		ModifiedIndex: msr.index,
		Expiration:    &exp,
	}

	n.Nodes = append(n.Nodes, node)
	msr.events <- &etcd.Response{
		Action: "add",
		Node:   node,
	}

	return &etcd.Response{
		Node:  node,
		Index: msr.index,
	}, nil
}

func (msr *MockSubnetRegistry) updateSubnet(ctx context.Context, network, sn, data string, ttl time.Duration) (*etcd.Response, error) {
	n, ok := msr.networks[network]
	if !ok {
		return nil, fmt.Errorf("Network %s not found", network)
	}

	msr.index += 1

	exp := time.Now().Add(ttl)

	for _, sub := range n.Nodes {
		if sub.Key == sn {
			sub.Value = data
			sub.ModifiedIndex = msr.index
			sub.Expiration = &exp
			msr.events <- &etcd.Response{
				Action: "add",
				Node:   sub,
			}

			return &etcd.Response{
				Node:  sub,
				Index: msr.index,
			}, nil
		}
	}

	return nil, fmt.Errorf("Subnet not found")
}

func (msr *MockSubnetRegistry) deleteSubnet(ctx context.Context, network, sn string) (*etcd.Response, error) {
	n, ok := msr.networks[network]
	if !ok {
		return nil, fmt.Errorf("Network %s not found", network)
	}

	msr.index += 1

	for i, sub := range n.Nodes {
		if sub.Key == sn {
			n.Nodes[i] = n.Nodes[len(n.Nodes)-1]
			n.Nodes = n.Nodes[:len(n.Nodes)-1]
			sub.ModifiedIndex = msr.index
			msr.events <- &etcd.Response{
				Action: "delete",
				Node:   sub,
			}

			return &etcd.Response{
				Node:  sub,
				Index: msr.index,
			}, nil
		}
	}

	return nil, fmt.Errorf("Subnet not found")

}

func (msr *MockSubnetRegistry) watch(ctx context.Context, network string, since uint64) (*etcd.Response, error) {
	for {
		if since < msr.index {
			return nil, etcd.Error{
				Code:    etcd.ErrorCodeEventIndexCleared,
				Cause:   "out of date",
				Message: "cursor is out of date",
				Index:   msr.index,
			}
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()

		case r := <-msr.events:
			if r.Node.ModifiedIndex <= since {
				continue
			}
			return r, nil
		}
	}
}

func (msr *MockSubnetRegistry) hasSubnet(network, sn string) bool {
	n, ok := msr.networks[network]
	if !ok {
		return false
	}

	for _, sub := range n.Nodes {
		if sub.Key == sn {
			return true
		}
	}
	return false
}

func (msr *MockSubnetRegistry) expireSubnet(network, sn string) {
	n, ok := msr.networks[network]
	if !ok {
		return
	}

	for i, sub := range n.Nodes {
		if sub.Key == sn {
			msr.index += 1
			n.Nodes[i] = n.Nodes[len(n.Nodes)-1]
			n.Nodes = n.Nodes[:len(n.Nodes)-2]
			sub.ModifiedIndex = msr.index
			msr.events <- &etcd.Response{
				Action: "expire",
				Node:   sub,
			}
			return
		}
	}
}

func (msr *MockSubnetRegistry) getNetworks(ctx context.Context) (*etcd.Response, error) {
	var keys []string
	for k := range msr.networks {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	networks := &etcd.Node{Key: networkKeyPrefix, Value: "", ModifiedIndex: msr.index, Nodes: make([]*etcd.Node, 0, len(keys))}
	for _, k := range keys {
		networks.Nodes = append(networks.Nodes, msr.networks[k])
	}

	return &etcd.Response{
		Node:  networks,
		Index: msr.index,
	}, nil
}

func (msr *MockSubnetRegistry) getNetwork(ctx context.Context, network string) (*etcd.Response, error) {
	n, ok := msr.networks[network]
	if !ok {
		return nil, fmt.Errorf("Network %s not found", network)
	}

	return &etcd.Response{
		Node:  n,
		Index: msr.index,
	}, nil
}

func (msr *MockSubnetRegistry) CreateNetwork(ctx context.Context, network, config string) (*etcd.Response, error) {
	_, ok := msr.networks[network]
	if ok {
		return nil, fmt.Errorf("Network %s already exists", network)
	}

	msr.index += 1

	node := &etcd.Node{
		Key:           normalizeNetKey(network),
		Value:         config,
		ModifiedIndex: msr.index,
	}

	msr.networks[network] = node
	msr.events <- &etcd.Response{
		Action: "add",
		Node:   node,
	}

	return &etcd.Response{
		Node:  node,
		Index: msr.index,
	}, nil
}

func (msr *MockSubnetRegistry) DeleteNetwork(ctx context.Context, network string) (*etcd.Response, error) {
	n, ok := msr.networks[network]
	if !ok {
		return nil, fmt.Errorf("Network %s not found", network)
	}

	msr.index += 1

	n.ModifiedIndex = msr.index

	delete(msr.networks, network)
	msr.events <- &etcd.Response{
		Action: "delete",
		Node:   n,
	}

	return &etcd.Response{
		Node:  n,
		Index: msr.index,
	}, nil
}

func normalizeNetKey(key string) string {
	match := networkKeyPrefix
	newKey := key
	if !strings.HasPrefix(newKey, match+"/") {
		newKey = path.Join(match, key)
	}
	if !strings.HasSuffix(newKey, "/config") {
		newKey = path.Join(newKey, "config")
	}
	return newKey
}
