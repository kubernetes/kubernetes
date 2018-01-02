package convert

import (
	"fmt"
	"strings"
	"time"

	types "github.com/docker/docker/api/types/swarm"
	swarmapi "github.com/docker/swarmkit/api"
	"github.com/docker/swarmkit/ca"
	gogotypes "github.com/gogo/protobuf/types"
)

// SwarmFromGRPC converts a grpc Cluster to a Swarm.
func SwarmFromGRPC(c swarmapi.Cluster) types.Swarm {
	swarm := types.Swarm{
		ClusterInfo: types.ClusterInfo{
			ID: c.ID,
			Spec: types.Spec{
				Orchestration: types.OrchestrationConfig{
					TaskHistoryRetentionLimit: &c.Spec.Orchestration.TaskHistoryRetentionLimit,
				},
				Raft: types.RaftConfig{
					SnapshotInterval:           c.Spec.Raft.SnapshotInterval,
					KeepOldSnapshots:           &c.Spec.Raft.KeepOldSnapshots,
					LogEntriesForSlowFollowers: c.Spec.Raft.LogEntriesForSlowFollowers,
					HeartbeatTick:              int(c.Spec.Raft.HeartbeatTick),
					ElectionTick:               int(c.Spec.Raft.ElectionTick),
				},
				EncryptionConfig: types.EncryptionConfig{
					AutoLockManagers: c.Spec.EncryptionConfig.AutoLockManagers,
				},
				CAConfig: types.CAConfig{
					// do not include the signing CA cert or key (it should already be redacted via the swarm APIs) -
					// the key because it's secret, and the cert because otherwise doing a get + update on the spec
					// can cause issues because the key would be missing and the cert wouldn't
					ForceRotate: c.Spec.CAConfig.ForceRotate,
				},
			},
			TLSInfo: types.TLSInfo{
				TrustRoot: string(c.RootCA.CACert),
			},
			RootRotationInProgress: c.RootCA.RootRotation != nil,
		},
		JoinTokens: types.JoinTokens{
			Worker:  c.RootCA.JoinTokens.Worker,
			Manager: c.RootCA.JoinTokens.Manager,
		},
	}

	issuerInfo, err := ca.IssuerFromAPIRootCA(&c.RootCA)
	if err == nil && issuerInfo != nil {
		swarm.TLSInfo.CertIssuerSubject = issuerInfo.Subject
		swarm.TLSInfo.CertIssuerPublicKey = issuerInfo.PublicKey
	}

	heartbeatPeriod, _ := gogotypes.DurationFromProto(c.Spec.Dispatcher.HeartbeatPeriod)
	swarm.Spec.Dispatcher.HeartbeatPeriod = heartbeatPeriod

	swarm.Spec.CAConfig.NodeCertExpiry, _ = gogotypes.DurationFromProto(c.Spec.CAConfig.NodeCertExpiry)

	for _, ca := range c.Spec.CAConfig.ExternalCAs {
		swarm.Spec.CAConfig.ExternalCAs = append(swarm.Spec.CAConfig.ExternalCAs, &types.ExternalCA{
			Protocol: types.ExternalCAProtocol(strings.ToLower(ca.Protocol.String())),
			URL:      ca.URL,
			Options:  ca.Options,
			CACert:   string(ca.CACert),
		})
	}

	// Meta
	swarm.Version.Index = c.Meta.Version.Index
	swarm.CreatedAt, _ = gogotypes.TimestampFromProto(c.Meta.CreatedAt)
	swarm.UpdatedAt, _ = gogotypes.TimestampFromProto(c.Meta.UpdatedAt)

	// Annotations
	swarm.Spec.Annotations = annotationsFromGRPC(c.Spec.Annotations)

	return swarm
}

// SwarmSpecToGRPC converts a Spec to a grpc ClusterSpec.
func SwarmSpecToGRPC(s types.Spec) (swarmapi.ClusterSpec, error) {
	return MergeSwarmSpecToGRPC(s, swarmapi.ClusterSpec{})
}

// MergeSwarmSpecToGRPC merges a Spec with an initial grpc ClusterSpec
func MergeSwarmSpecToGRPC(s types.Spec, spec swarmapi.ClusterSpec) (swarmapi.ClusterSpec, error) {
	// We take the initSpec (either created from scratch, or returned by swarmkit),
	// and will only change the value if the one taken from types.Spec is not nil or 0.
	// In other words, if the value taken from types.Spec is nil or 0, we will maintain the status quo.
	if s.Annotations.Name != "" {
		spec.Annotations.Name = s.Annotations.Name
	}
	if len(s.Annotations.Labels) != 0 {
		spec.Annotations.Labels = s.Annotations.Labels
	}

	if s.Orchestration.TaskHistoryRetentionLimit != nil {
		spec.Orchestration.TaskHistoryRetentionLimit = *s.Orchestration.TaskHistoryRetentionLimit
	}
	if s.Raft.SnapshotInterval != 0 {
		spec.Raft.SnapshotInterval = s.Raft.SnapshotInterval
	}
	if s.Raft.KeepOldSnapshots != nil {
		spec.Raft.KeepOldSnapshots = *s.Raft.KeepOldSnapshots
	}
	if s.Raft.LogEntriesForSlowFollowers != 0 {
		spec.Raft.LogEntriesForSlowFollowers = s.Raft.LogEntriesForSlowFollowers
	}
	if s.Raft.HeartbeatTick != 0 {
		spec.Raft.HeartbeatTick = uint32(s.Raft.HeartbeatTick)
	}
	if s.Raft.ElectionTick != 0 {
		spec.Raft.ElectionTick = uint32(s.Raft.ElectionTick)
	}
	if s.Dispatcher.HeartbeatPeriod != 0 {
		spec.Dispatcher.HeartbeatPeriod = gogotypes.DurationProto(time.Duration(s.Dispatcher.HeartbeatPeriod))
	}
	if s.CAConfig.NodeCertExpiry != 0 {
		spec.CAConfig.NodeCertExpiry = gogotypes.DurationProto(s.CAConfig.NodeCertExpiry)
	}
	if s.CAConfig.SigningCACert != "" {
		spec.CAConfig.SigningCACert = []byte(s.CAConfig.SigningCACert)
	}
	if s.CAConfig.SigningCAKey != "" {
		// do propagate the signing CA key here because we want to provide it TO the swarm APIs
		spec.CAConfig.SigningCAKey = []byte(s.CAConfig.SigningCAKey)
	}
	spec.CAConfig.ForceRotate = s.CAConfig.ForceRotate

	for _, ca := range s.CAConfig.ExternalCAs {
		protocol, ok := swarmapi.ExternalCA_CAProtocol_value[strings.ToUpper(string(ca.Protocol))]
		if !ok {
			return swarmapi.ClusterSpec{}, fmt.Errorf("invalid protocol: %q", ca.Protocol)
		}
		spec.CAConfig.ExternalCAs = append(spec.CAConfig.ExternalCAs, &swarmapi.ExternalCA{
			Protocol: swarmapi.ExternalCA_CAProtocol(protocol),
			URL:      ca.URL,
			Options:  ca.Options,
			CACert:   []byte(ca.CACert),
		})
	}

	spec.EncryptionConfig.AutoLockManagers = s.EncryptionConfig.AutoLockManagers

	return spec, nil
}
