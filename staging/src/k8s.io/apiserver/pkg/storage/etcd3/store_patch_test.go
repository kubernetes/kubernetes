/*
Copyright 2022 The KCP Authors.

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

package etcd3

import (
	"testing"

	"github.com/kcp-dev/logicalcluster/v3"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

func TestAdjustClusterNameIfWildcard(t *testing.T) {
	tests := map[string]struct {
		wildcard        bool
		partialMetadata bool
		prefix          string
		builtInType     bool
	}{
		"not clusterWildcard": {
			prefix: "/registry/group/resource/identity/",
		},
		"clusterWildcard, not partial": {
			wildcard: true,
			prefix:   "/registry/group/resource/identity/",
		},
		"clusterWildcard, partial": {
			wildcard:        true,
			partialMetadata: true,
			prefix:          "/registry/group/resource/",
		},
		"clusterWildcard, partial, built-in type": {
			wildcard:        true,
			partialMetadata: true,
			prefix:          "/registry/core/configmaps/",
			builtInType:     true,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			cluster := &genericapirequest.Cluster{
				Name:                   logicalcluster.New("root:org:ws"),
				PartialMetadataRequest: tc.partialMetadata,
			}

			if tc.wildcard {
				cluster.Name = logicalcluster.Wildcard
			}

			key := "/registry/group/resource/identity/root:org:ws/somename"
			if tc.builtInType {
				key = "/registry/core/configmaps/root:org:ws/somename"
			}
			expected := "root:org:ws"

			clusterName := adjustClusterNameIfWildcard(genericapirequest.Shard(""), cluster, !tc.builtInType, tc.prefix, key)
			if e, a := expected, clusterName.String(); e != a {
				t.Errorf("expected: %q, actual %q", e, a)
			}
		})
	}
}

func TestAdjustClusterNameIfWildcardWithShardSupport(t *testing.T) {
	tests := map[string]struct {
		cluster             genericapirequest.Cluster
		shard               genericapirequest.Shard
		key                 string
		keyPrefix           string
		expectedClusterName string
	}{
		"not wildcard": {
			cluster:             genericapirequest.Cluster{Name: logicalcluster.New("root:org:ws")},
			shard:               "amber",
			key:                 "/registry/group/resource:identity/amber/root:org:ws/somename",
			keyPrefix:           "/registry/group/resource:identity/",
			expectedClusterName: "root:org:ws",
		},
		"both wildcard": {
			cluster:             genericapirequest.Cluster{Name: logicalcluster.Wildcard},
			shard:               "*",
			key:                 "/registry/group/resource:identity/amber/root:org:ws/somename",
			keyPrefix:           "/registry/group/resource:identity/",
			expectedClusterName: "root:org:ws",
		},
		"both wildcard, built-in type": {
			cluster:             genericapirequest.Cluster{Name: logicalcluster.Wildcard},
			shard:               "*",
			key:                 "/registry/core/configmaps/amber/root:org:ws/somename",
			keyPrefix:           "/registry/core/configmaps/",
			expectedClusterName: "root:org:ws",
		},
		"only cluster wildcard": {
			cluster:             genericapirequest.Cluster{Name: logicalcluster.Wildcard},
			shard:               "amber",
			key:                 "/registry/group/resource:identity/amber/root:org:ws/somename",
			keyPrefix:           "/registry/group/resource:identity/amber/",
			expectedClusterName: "root:org:ws",
		},
		"only shard wildcard": {
			cluster:             genericapirequest.Cluster{Name: logicalcluster.New("root:org:ws")},
			shard:               "*",
			key:                 "/registry/core/configmaps/amber/root:org:ws/somename",
			keyPrefix:           "/registry/group/resource:identity/",
			expectedClusterName: "root:org:ws",
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			clusterName := adjustClusterNameIfWildcard(tc.shard, &tc.cluster, false, tc.keyPrefix, tc.key)
			if tc.expectedClusterName != clusterName.String() {
				t.Errorf("expected: %q, actual %q", tc.expectedClusterName, clusterName)
			}
		})
	}
}
