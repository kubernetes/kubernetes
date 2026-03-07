/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package knftables

import (
	"context"
	"fmt"

	gnftables "github.com/google/nftables"
)

var nftableFamilies = map[Family]gnftables.TableFamily{
	IPv4Family:   gnftables.TableFamilyIPv4,
	IPv6Family:   gnftables.TableFamilyIPv6,
	InetFamily:   gnftables.TableFamilyINet,
	ARPFamily:    gnftables.TableFamilyARP,
	BridgeFamily: gnftables.TableFamilyBridge,
	NetDevFamily: gnftables.TableFamilyNetdev,
}

type netlink interface {
	List(ctx context.Context, objectType string) ([]string, error)
}

type netlinkAdapter struct {
	conn   *gnftables.Conn
	family gnftables.TableFamily
	table  string
}

func newNetlinkAdapter(family Family, table string) (netlink, error) {
	c, err := gnftables.New()
	if err != nil {
		return nil, err
	}
	gnfamily, ok := nftableFamilies[family]
	if !ok {
		return nil, fmt.Errorf("unsupported family: %s", family)
	}
	return &netlinkAdapter{
		conn:   c,
		family: gnfamily,
		table:  table,
	}, nil
}

// List implements Interface.List
func (n *netlinkAdapter) List(_ context.Context, objectType string) ([]string, error) {
	if n.table == "" {
		return nil, fmt.Errorf("no table specified")
	}

	objectType = canonicalObjectType(objectType)

	switch objectType {
	case "chain":
		return n.listChains()
	case "set", "map":
		return n.listSets(objectType)
	case "counter":
		return n.listCounters()
	case "flowtable":
		return n.listFlowtables()
	default:
		return nil, fmt.Errorf("unsupported object type for netlink listing: %s", objectType)
	}
}

func (n *netlinkAdapter) listChains() ([]string, error) {
	chains, err := n.conn.ListChainsOfTableFamily(n.family)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(chains))
	for _, c := range chains {
		if c.Table.Name == n.table {
			names = append(names, c.Name)
		}
	}
	return names, nil
}

func (n *netlinkAdapter) listSets(objectType string) ([]string, error) {
	if objectType != "map" && objectType != "set" {
		return nil, fmt.Errorf("unsupported object type: %s", objectType)
	}
	// google/nftables uses GetSets
	sets, err := n.conn.GetSets(&gnftables.Table{Name: n.table, Family: n.family})
	if err != nil {
		return nil, err
	}
	var names []string
	for _, s := range sets {
		if s.IsMap == (objectType == "map") {
			names = append(names, s.Name)
		}
	}
	return names, nil
}

func (n *netlinkAdapter) listCounters() ([]string, error) {
	// nftables.Conn doesn't have explicit ListCounters high level method that returns counters
	// It has GetObj looking for ObjTypeCounter.
	objs, err := n.conn.GetObjects(&gnftables.Table{Name: n.table, Family: n.family})
	if err != nil {
		return nil, err
	}
	var names []string
	for _, o := range objs {
		if c, ok := o.(*gnftables.CounterObj); ok {
			names = append(names, c.Name)
		}
	}
	return names, nil
}

func (n *netlinkAdapter) listFlowtables() ([]string, error) {
	flowtables, err := n.conn.ListFlowtables(&gnftables.Table{Name: n.table, Family: n.family})
	if err != nil {
		return nil, err
	}
	var names []string
	for _, ft := range flowtables {
		names = append(names, ft.Name)
	}
	return names, nil
}
