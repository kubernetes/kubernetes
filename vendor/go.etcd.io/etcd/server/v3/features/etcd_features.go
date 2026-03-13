// Copyright 2024 The etcd Authors
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

package features

import (
	"fmt"

	"go.uber.org/zap"

	"go.etcd.io/etcd/pkg/v3/featuregate"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // kep: https://kep.k8s.io/NNN (or issue: https://github.com/etcd-io/etcd/issues/NNN, or main PR: https://github.com/etcd-io/etcd/pull/NNN)
	// // alpha: v3.X
	// MyFeature featuregate.Feature = "MyFeature"
	//
	// Feature gates should be listed in alphabetical, case-sensitive
	// (upper before any lower case character) order. This reduces the risk
	// of code conflicts because changes are more likely to be scattered
	// across the file.

	// StopGRPCServiceOnDefrag enables etcd gRPC service to stop serving client requests on defragmentation.
	// owner: @chaochn47
	// alpha: v3.6
	// main PR: https://github.com/etcd-io/etcd/pull/18279
	StopGRPCServiceOnDefrag featuregate.Feature = "StopGRPCServiceOnDefrag"
	// TxnModeWriteWithSharedBuffer enables the write transaction to use a shared buffer in its readonly check operations.
	// owner: @wilsonwang371
	// beta: v3.5
	// main PR: https://github.com/etcd-io/etcd/pull/12896
	TxnModeWriteWithSharedBuffer featuregate.Feature = "TxnModeWriteWithSharedBuffer"
	// InitialCorruptCheck enable to check data corruption before serving any client/peer traffic.
	// owner: @serathius
	// alpha: v3.6
	// main PR: https://github.com/etcd-io/etcd/pull/10524
	InitialCorruptCheck featuregate.Feature = "InitialCorruptCheck"
	// CompactHashCheck enables leader to periodically check followers compaction hashes.
	// owner: @serathius
	// alpha: v3.6
	// main PR: https://github.com/etcd-io/etcd/pull/14120
	CompactHashCheck featuregate.Feature = "CompactHashCheck"
	// LeaseCheckpoint enables leader to send regular checkpoints to other members to prevent reset of remaining TTL on leader change.
	// owner: @serathius
	// alpha: v3.6
	// main PR: https://github.com/etcd-io/etcd/pull/13508
	LeaseCheckpoint featuregate.Feature = "LeaseCheckpoint"
	// LeaseCheckpointPersist enables persisting remainingTTL to prevent indefinite auto-renewal of long lived leases. Always enabled in v3.6. Should be used to ensure smooth upgrade from v3.5 clusters with this feature enabled.
	// Requires EnableLeaseCheckpoint featuragate to be enabled.
	// TODO: Delete in v3.7
	// owner: @serathius
	// alpha: v3.6
	// main PR: https://github.com/etcd-io/etcd/pull/13508
	// Deprecated: Enabled by default in v3.6, to be removed in v3.7.
	LeaseCheckpointPersist featuregate.Feature = "LeaseCheckpointPersist"
	// SetMemberLocalAddr enables using the first specified and non-loopback local address from initial-advertise-peer-urls as the local address when communicating with a peer.
	// Requires SetMemberLocalAddr featuragate to be enabled.
	// owner: @flawedmatrix
	// alpha: v3.6
	// main PR: https://github.com/etcd-io/etcd/pull/17661
	SetMemberLocalAddr featuregate.Feature = "SetMemberLocalAddr"
)

var (
	DefaultEtcdServerFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
		StopGRPCServiceOnDefrag:      {Default: false, PreRelease: featuregate.Alpha},
		InitialCorruptCheck:          {Default: false, PreRelease: featuregate.Alpha},
		CompactHashCheck:             {Default: false, PreRelease: featuregate.Alpha},
		TxnModeWriteWithSharedBuffer: {Default: true, PreRelease: featuregate.Beta},
		LeaseCheckpoint:              {Default: false, PreRelease: featuregate.Alpha},
		LeaseCheckpointPersist:       {Default: false, PreRelease: featuregate.Alpha},
		SetMemberLocalAddr:           {Default: false, PreRelease: featuregate.Alpha},
	}
	// ExperimentalFlagToFeatureMap is the map from the cmd line flags of experimental features
	// to their corresponding feature gates.
	// Deprecated: Only add existing experimental features here. DO NOT use for new features.
	ExperimentalFlagToFeatureMap = map[string]featuregate.Feature{
		"experimental-stop-grpc-service-on-defrag":       StopGRPCServiceOnDefrag,
		"experimental-initial-corrupt-check":             InitialCorruptCheck,
		"experimental-compact-hash-check-enabled":        CompactHashCheck,
		"experimental-txn-mode-write-with-shared-buffer": TxnModeWriteWithSharedBuffer,
		"experimental-enable-lease-checkpoint":           LeaseCheckpoint,
		"experimental-enable-lease-checkpoint-persist":   LeaseCheckpointPersist,
	}
)

func NewDefaultServerFeatureGate(name string, lg *zap.Logger) featuregate.FeatureGate {
	fg := featuregate.New(fmt.Sprintf("%sServerFeatureGate", name), lg)
	if err := fg.Add(DefaultEtcdServerFeatureGates); err != nil {
		panic(err)
	}
	return fg
}
