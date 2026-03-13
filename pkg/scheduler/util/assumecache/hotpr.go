/*
Copyright 2017 The Kubernetes Authors.

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

package assumecache

import (
	"k8s.io/klog/v2"
)

// ConflictTier represents the severity of a conflict between
// assumed state and informer updates.
type ConflictTier int

const (
	// TierAutoMerge: informer update doesn't overlap with assumed fields.
	// Keep both — assumed placement + new metadata from informer.
	TierAutoMerge ConflictTier = iota + 1
	// TierEtcdWins: informer update conflicts with assume.
	// Etcd is truth. Scheduler gets notified (not silent drop).
	TierEtcdWins
	// TierBound: pod was already bound. Binding is truth.
	// Stale informer updates and assumes are discarded.
	TierBound
)

// ConflictResult is returned by ResolveConflict.
type ConflictResult struct {
	Tier       ConflictTier
	Resolution string // "merged", "etcd_wins", "bound"
	AssumeGen  int64
	EtcdGen    int64
	// MergedObj is the result object after resolution (nil if discarded)
	MergedObj interface{}
}

// ResolveConflict determines how to handle a conflict between
// an assumed object and an informer update.
//
// Resolution tiers:
//   - TierBound: pod already bound — binding is truth, return assumed object.
//   - TierAutoMerge: assume is newer than informer — keep assumed state.
//   - TierEtcdWins: informer is newer — etcd wins, log the conflict.
func ResolveConflict(assumed, fromInformer interface{}, assumeVersion, informerVersion int64, isBound bool) ConflictResult {
	// Tier 3: Bound objects are truth. The binding decision has been
	// committed and stale informer updates must not override it.
	if isBound {
		return ConflictResult{
			Tier:       TierBound,
			Resolution: "bound",
			AssumeGen:  assumeVersion,
			EtcdGen:    informerVersion,
			MergedObj:  assumed,
		}
	}

	// Tier 1: Informer version is not newer than the assume version.
	// The assume is at least as fresh — keep assumed state (auto-merge).
	if informerVersion <= assumeVersion {
		return ConflictResult{
			Tier:       TierAutoMerge,
			Resolution: "merged",
			AssumeGen:  assumeVersion,
			EtcdGen:    informerVersion,
			MergedObj:  assumed,
		}
	}

	// Tier 2: Informer version is strictly newer. Etcd is truth.
	// Log the conflict so the scheduler is aware — never silently drop.
	klog.V(4).InfoS("Hot PR conflict: etcd wins",
		"assumeGen", assumeVersion,
		"informerGen", informerVersion,
	)
	return ConflictResult{
		Tier:       TierEtcdWins,
		Resolution: "etcd_wins",
		AssumeGen:  assumeVersion,
		EtcdGen:    informerVersion,
		MergedObj:  fromInformer,
	}
}
