package consul

import (
	"fmt"

	"github.com/hashicorp/consul/consul/agent"
	"github.com/hashicorp/serf/serf"
)

// lanMergeDelegate is used to handle a cluster merge on the LAN gossip
// ring. We check that the peers are in the same datacenter and abort the
// merge if there is a mis-match.
type lanMergeDelegate struct {
	dc string
}

func (md *lanMergeDelegate) NotifyMerge(members []*serf.Member) error {
	for _, m := range members {
		ok, dc := isConsulNode(*m)
		if ok {
			if dc != md.dc {
				return fmt.Errorf("Member '%s' part of wrong datacenter '%s'",
					m.Name, dc)
			}
			continue
		}

		ok, parts := agent.IsConsulServer(*m)
		if ok && parts.Datacenter != md.dc {
			return fmt.Errorf("Member '%s' part of wrong datacenter '%s'",
				m.Name, parts.Datacenter)
		}
	}
	return nil
}

// wanMergeDelegate is used to handle a cluster merge on the WAN gossip
// ring. We check that the peers are server nodes and abort the merge
// otherwise.
type wanMergeDelegate struct {
}

func (md *wanMergeDelegate) NotifyMerge(members []*serf.Member) error {
	for _, m := range members {
		ok, _ := agent.IsConsulServer(*m)
		if !ok {
			return fmt.Errorf("Member '%s' is not a server", m.Name)
		}
	}
	return nil
}
