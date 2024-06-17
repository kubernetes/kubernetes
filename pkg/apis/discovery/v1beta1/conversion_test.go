/*
Copyright 2021 The Kubernetes Authors.

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

package v1beta1

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/api/discovery/v1beta1"
	"k8s.io/kubernetes/pkg/apis/discovery"
	utilpointer "k8s.io/utils/pointer"
)

func TestEndpointZoneConverstion(t *testing.T) {
	testcases := []struct {
		desc     string
		external v1beta1.Endpoint
		internal discovery.Endpoint
	}{
		{
			desc:     "no topology field",
			external: v1beta1.Endpoint{},
			internal: discovery.Endpoint{},
		},
		{
			desc: "non empty topology map, but no zone",
			external: v1beta1.Endpoint{
				Topology: map[string]string{
					"key1": "val1",
				},
			},
			internal: discovery.Endpoint{
				DeprecatedTopology: map[string]string{
					"key1": "val1",
				},
			},
		},
		{
			desc: "non empty topology map, with zone",
			external: v1beta1.Endpoint{
				Topology: map[string]string{
					"key1":                   "val1",
					corev1.LabelTopologyZone: "zone1",
				},
			},
			internal: discovery.Endpoint{
				DeprecatedTopology: map[string]string{
					"key1": "val1",
				},
				Zone: utilpointer.String("zone1"),
			},
		},
		{
			desc: "only zone in topology map",
			external: v1beta1.Endpoint{
				Topology: map[string]string{
					corev1.LabelTopologyZone: "zone1",
				},
			},
			internal: discovery.Endpoint{
				Zone: utilpointer.String("zone1"),
			},
		},
		{
			desc: "nodeName and topology[hostname] are populated with different values",
			external: v1beta1.Endpoint{
				NodeName: utilpointer.String("node-1"),
				Topology: map[string]string{
					corev1.LabelHostname: "node-2",
				},
			},
			internal: discovery.Endpoint{
				NodeName: utilpointer.String("node-1"),
				DeprecatedTopology: map[string]string{
					corev1.LabelHostname: "node-2",
				},
			},
		},
		{
			desc: "nodeName and topology[hostname] are populated with same values",
			external: v1beta1.Endpoint{
				NodeName: utilpointer.String("node-1"),
				Topology: map[string]string{
					corev1.LabelHostname: "node-1",
				},
			},
			internal: discovery.Endpoint{
				NodeName: utilpointer.String("node-1"),
			},
		},
		{
			desc: "only topology[hostname] is populated",
			external: v1beta1.Endpoint{
				Topology: map[string]string{
					corev1.LabelHostname: "node-1",
				},
			},
			internal: discovery.Endpoint{
				DeprecatedTopology: map[string]string{
					corev1.LabelHostname: "node-1",
				},
			},
		},
		{
			desc: "only nodeName is populated",
			external: v1beta1.Endpoint{
				NodeName: utilpointer.String("node-1"),
				Topology: map[string]string{
					corev1.LabelHostname: "node-1",
				},
			},
			internal: discovery.Endpoint{
				NodeName: utilpointer.String("node-1"),
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			convertedInternal := discovery.Endpoint{}
			require.NoError(t, Convert_v1beta1_Endpoint_To_discovery_Endpoint(&tc.external, &convertedInternal, nil))
			assert.Equal(t, tc.internal, convertedInternal, "v1beta1.Endpoint -> discovery.Endpoint")

			convertedV1beta1 := v1beta1.Endpoint{}
			require.NoError(t, Convert_discovery_Endpoint_To_v1beta1_Endpoint(&tc.internal, &convertedV1beta1, nil))
			assert.Equal(t, tc.external, convertedV1beta1, "discovery.Endpoint -> v1beta1.Endpoint")
		})
	}
}
