/*
Copyright 2026 The Kubernetes Authors.

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

// Package watchexperiment turns a single multi-master scale run into a
// controlled experiment.
//
// THIS PACKAGE IS A TEMPORARY TESTING SCAFFOLD. It exists because a 5000-node
// run is expensive and the three apiservers in an HA run see statistically
// identical workloads, which makes them three arms of one experiment rather
// than three copies of one measurement. It must not merge.
//
// The cluster provisioner sets one apiserver spec for every master, so the arm
// cannot come from a flag; it is derived from the hostname instead. Which arm a
// given apiserver landed in is published as apiserver_watch_experiment_variant.
package watchexperiment

import (
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/http2"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
)

// Variant is one arm of the experiment.
type Variant string

const (
	// VariantGraceBudget raises the dispatch time budget ceiling. If this arm
	// stops killing watchers while the others keep killing, the watchers were
	// momentarily behind and recovered, which points at a server-side transient
	// rather than a client that is gone for seconds.
	VariantGraceBudget Variant = "A_grace_budget"

	// VariantChunkWriteSize raises HTTP/2's handler write buffer past the
	// observed event size, so events stop pushing to the wire from inside
	// Write. Expect write down, flush up, and one socket push per event.
	VariantChunkWriteSize Variant = "B_chunk_write_size"

	// VariantControl changes nothing.
	VariantControl Variant = "C_control"
)

const (
	// defaultMaxDispatchBudget matches the upstream maxBudget.
	defaultMaxDispatchBudget = 100 * time.Millisecond
	// raisedMaxDispatchBudget only moves the burst ceiling. The refresh rate is
	// deliberately untouched, so sustained dispatch cost is identical to
	// control and this arm cannot spend more time dispatching on average.
	raisedMaxDispatchBudget = 1 * time.Second

	defaultChunkWriteSize = 4 << 10
	raisedChunkWriteSize  = 16 << 10
)

var experimentVariant = compbasemetrics.NewGaugeVec(
	&compbasemetrics.GaugeOpts{
		Namespace:      "apiserver",
		Name:           "watch_experiment_variant",
		Help:           "Always 1, labelled with the watch-chain experiment arm this apiserver is running. Join on this to attribute a difference to an arm rather than to the instance.",
		StabilityLevel: compbasemetrics.ALPHA,
	},
	[]string{"variant"},
)

var (
	once     sync.Once
	resolved Variant

	// applyOnce guards Apply, which several servers in one process (tests, the
	// aggregator and apiextensions delegates) would otherwise run repeatedly.
	applyOnce sync.Once
)

// variantForHostname maps a master to an arm.
//
// kops names masters <role>-<region>-<zone>-<random>, e.g.
// control-plane-us-east1-b-crf7, and a 3-master scale cluster puts one master
// in each of three zones. The zone letter is therefore both stable across runs
// and guaranteed distinct between masters, which hashing the whole hostname
// would not be: with three hosts and three arms a hash collides into a
// degenerate assignment more than three quarters of the time.
func variantForHostname(hostname string) Variant {
	segments := strings.Split(hostname, "-")
	if len(segments) >= 2 {
		zone := segments[len(segments)-2]
		switch zone {
		case "b":
			return VariantGraceBudget
		case "c":
			return VariantChunkWriteSize
		case "d":
			return VariantControl
		}
	}
	// Anything not matching the expected shape runs as control, so an
	// unexpected topology degrades to a normal run rather than to a silently
	// mixed one.
	return VariantControl
}

// Current returns this apiserver's arm, resolving it once.
func Current() Variant {
	once.Do(func() {
		hostname, err := os.Hostname()
		if err != nil {
			klog.Errorf("watch experiment: cannot read hostname, running as control: %v", err)
			resolved = VariantControl
			return
		}
		resolved = variantForHostname(hostname)
		klog.Infof("watch experiment: hostname %q assigned to variant %q", hostname, resolved)
	})
	return resolved
}

// MaxDispatchBudget is the ceiling for the cacher's dispatch time budget.
func MaxDispatchBudget() time.Duration {
	if Current() == VariantGraceBudget {
		return raisedMaxDispatchBudget
	}
	return defaultMaxDispatchBudget
}

// Apply installs the settings that cannot be read lazily, and publishes which
// arm is active. It must be called once during server startup, before any
// connection is served.
func Apply() {
	applyOnce.Do(func() {
		legacyregistry.MustRegister(experimentVariant)
		variant := Current()
		experimentVariant.WithLabelValues(string(variant)).Set(1)

		chunkWriteSize := defaultChunkWriteSize
		if variant == VariantChunkWriteSize {
			chunkWriteSize = raisedChunkWriteSize
		}
		// Always set it, so control and the raised arm go through identical code.
		http2.SetHandlerChunkWriteSize(chunkWriteSize)
		klog.Infof("watch experiment: variant %q, http2 handler chunk write size %d", variant, chunkWriteSize)
	})
}
