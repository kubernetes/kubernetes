/*
Copyright 2020 The Kubernetes Authors.

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

package topology

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"
)

func Test_GetZoneKey(t *testing.T) {
	tests := []struct {
		name string
		node *v1.Node
		zone string
	}{
		{
			name: "has no zone or region keys",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{},
				},
			},
			zone: "",
		},
		{
			name: "has beta zone and region keys",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						v1.LabelFailureDomainBetaZone:   "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
					},
				},
			},
			zone: "region1:\x00:zone1",
		},
		{
			name: "has GA zone and region keys",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						v1.LabelTopologyZone:   "zone1",
						v1.LabelTopologyRegion: "region1",
					},
				},
			},
			zone: "region1:\x00:zone1",
		},
		{
			name: "has both beta and GA zone and region keys",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						v1.LabelTopologyZone:            "zone1",
						v1.LabelTopologyRegion:          "region1",
						v1.LabelFailureDomainBetaZone:   "zone1",
						v1.LabelFailureDomainBetaRegion: "region1",
					},
				},
			},
			zone: "region1:\x00:zone1",
		},
		{
			name: "has both beta and GA zone and region keys, beta labels take precedent",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						v1.LabelTopologyZone:            "zone1",
						v1.LabelTopologyRegion:          "region1",
						v1.LabelFailureDomainBetaZone:   "zone2",
						v1.LabelFailureDomainBetaRegion: "region2",
					},
				},
			},
			zone: "region2:\x00:zone2",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			zone := GetZoneKey(test.node)
			if zone != test.zone {
				t.Logf("actual zone key: %q", zone)
				t.Logf("expected zone key: %q", test.zone)
				t.Errorf("unexpected zone key")
			}
		})
	}
}
