package cluster

import (
	"math/rand"

	"github.com/influxdb/influxdb/meta"
)

// Balancer represents a load-balancing algorithm for a set of nodes
type Balancer interface {
	// Next returns the next Node according to the balancing method
	// or nil if there are no nodes available
	Next() *meta.NodeInfo
}

type nodeBalancer struct {
	nodes []meta.NodeInfo // data nodes to balance between
	p     int             // current node index
}

// NewNodeBalancer create a shuffled, round-robin balancer so that
// multiple instances will return nodes in randomized order and each
// each returned node will be repeated in a cycle
func NewNodeBalancer(nodes []meta.NodeInfo) Balancer {
	// make a copy of the node slice so we can randomize it
	// without affecting the original instance as well as ensure
	// that each Balancer returns nodes in a different order
	b := &nodeBalancer{}

	b.nodes = make([]meta.NodeInfo, len(nodes))
	copy(b.nodes, nodes)

	b.shuffle()
	return b
}

// shuffle randomizes the ordering the balancers available nodes
func (b *nodeBalancer) shuffle() {
	for i := range b.nodes {
		j := rand.Intn(i + 1)
		b.nodes[i], b.nodes[j] = b.nodes[j], b.nodes[i]
	}
}

// online returns a slice of the nodes that are online
func (b *nodeBalancer) online() []meta.NodeInfo {
	return b.nodes
	// now := time.Now().UTC()
	// up := []meta.NodeInfo{}
	// for _, n := range b.nodes {
	// 	if n.OfflineUntil.After(now) {
	// 		continue
	// 	}
	// 	up = append(up, n)
	// }
	// return up
}

// Next returns the next available nodes
func (b *nodeBalancer) Next() *meta.NodeInfo {
	// only use online nodes
	up := b.online()

	// no nodes online
	if len(up) == 0 {
		return nil
	}

	// rollover back to the beginning
	if b.p >= len(up) {
		b.p = 0
	}

	d := &up[b.p]
	b.p++

	return d
}
