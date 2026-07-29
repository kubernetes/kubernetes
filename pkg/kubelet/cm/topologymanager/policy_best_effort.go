/*
Copyright 2019 The Kubernetes Authors.

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

package topologymanager

import "k8s.io/klog/v2"

type bestEffortPolicy struct {
	// numaInfo represents list of NUMA Nodes available on the underlying machine and distances between them
	numaInfo *NUMAInfo
	opts     PolicyOptions
}

var _ Policy = &bestEffortPolicy{}

// PolicyBestEffort policy name.
const PolicyBestEffort string = "best-effort"

// NewBestEffortPolicy returns best-effort policy.
func NewBestEffortPolicy(numaInfo *NUMAInfo, opts PolicyOptions) Policy {
	return &bestEffortPolicy{numaInfo: numaInfo, opts: opts}
}

func (p *bestEffortPolicy) Name() string {
	return PolicyBestEffort
}

func (p *bestEffortPolicy) canAdmitPodResult(hint *TopologyHint) bool {
	return true
}

func (p *bestEffortPolicy) Merge(logger klog.Logger, providersHints []map[string][]TopologyHint) (TopologyHint, bool) {
	filteredHints := filterProvidersHints(logger, providersHints)
	merger := NewHintMerger(p.numaInfo, filteredHints, p.Name(), p.opts)
	bestHint := merger.Merge()
	admit := p.canAdmitPodResult(&bestHint)
	return bestHint, admit
}
