//go:build linux

/*
Copyright 2024 The Kubernetes Authors.

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

package conntrack

import (
	"fmt"

	"github.com/vishvananda/netlink"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
)

type flowFilter struct {
	flows sets.Set[string]
}

var _ netlink.CustomConntrackFilter = (*flowFilter)(nil)

func newFlowFilter(flows []*netlink.ConntrackFlow) *flowFilter {
	f := &flowFilter{
		flows: sets.New[string](),
	}
	for _, flow := range flows {
		f.flows.Insert(flowKey(flow))
	}
	return f
}

// flowKey returns a unique key for the given flow.
func flowKey(flow *netlink.ConntrackFlow) string {
	return fmt.Sprintf("%d#%s#%d#%s#%d#%s#%d#%s#%d",
		flow.Forward.Protocol,
		flow.Forward.SrcIP.String(),
		flow.Forward.SrcPort,
		flow.Forward.DstIP.String(),
		flow.Forward.DstPort,
		flow.Reverse.SrcIP.String(),
		flow.Reverse.SrcPort,
		flow.Reverse.DstIP.String(),
		flow.Reverse.DstPort)
}

// MatchConntrackFlow applies the filter to the flow and returns true if the flow matches the filter
func (f *flowFilter) MatchConntrackFlow(flow *netlink.ConntrackFlow) bool {
	if f.flows.Has(flowKey(flow)) {
		// for debugging conntrack issues and understanding what entries are deleted
		klog.V(6).InfoS("Deleting conntrack entry", "flow", flow)
		return true
	}
	return false
}
