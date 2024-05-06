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

package leadermigration

import (
	"testing"

	internal "k8s.io/controller-manager/config"
)

func TestLeaderMigratorFilterFunc(t *testing.T) {
	fromConfig := &internal.LeaderMigrationConfiguration{
		ResourceLock: "leases",
		LeaderName:   "cloud-provider-extraction-migration",
		ControllerLeaders: []internal.ControllerLeaderConfiguration{
			{
				Name:      "route-controller",
				Component: "kube-controller-manager",
			}, {
				Name:      "service-controller",
				Component: "kube-controller-manager",
			}, {
				Name:      "cloud-node-lifecycle-controller",
				Component: "kube-controller-manager",
			},
		},
	}
	toConfig := &internal.LeaderMigrationConfiguration{
		ResourceLock: "leases",
		LeaderName:   "cloud-provider-extraction-migration",
		ControllerLeaders: []internal.ControllerLeaderConfiguration{
			{
				Name:      "route-controller",
				Component: "cloud-controller-manager",
			}, {
				Name:      "service-controller",
				Component: "cloud-controller-manager",
			}, {
				Name:      "cloud-node-lifecycle-controller",
				Component: "cloud-controller-manager",
			},
		},
	}
	wildcardConfig := &internal.LeaderMigrationConfiguration{
		ResourceLock: "leases",
		LeaderName:   "cloud-provider-extraction-migration",
		ControllerLeaders: []internal.ControllerLeaderConfiguration{
			{
				Name:      "route-controller",
				Component: "*",
			}, {
				Name:      "service-controller",
				Component: "*",
			}, {
				Name:      "cloud-node-lifecycle-controller",
				Component: "*",
			},
		},
	}
	for _, tc := range []struct {
		name         string
		config       *internal.LeaderMigrationConfiguration
		component    string
		migrated     bool
		expectResult map[string]FilterResult
	}{
		{
			name:      "from config, kcm",
			config:    fromConfig,
			component: "kube-controller-manager",
			expectResult: map[string]FilterResult{
				"deployment-controller":           ControllerNonMigrated,
				"route-controller":                ControllerMigrated,
				"service-controller":              ControllerMigrated,
				"cloud-node-lifecycle-controller": ControllerMigrated,
			},
		},
		{
			name:      "from config, ccm",
			config:    fromConfig,
			component: "cloud-controller-manager",
			expectResult: map[string]FilterResult{
				"cloud-node":                      ControllerNonMigrated,
				"route-controller":                ControllerUnowned,
				"service-controller":              ControllerUnowned,
				"cloud-node-lifecycle-controller": ControllerUnowned,
			},
		},
		{
			name:      "to config, kcm",
			config:    toConfig,
			component: "kube-controller-manager",
			expectResult: map[string]FilterResult{
				"deployment-controller":           ControllerNonMigrated,
				"route-controller":                ControllerUnowned,
				"service-controller":              ControllerUnowned,
				"cloud-node-lifecycle-controller": ControllerUnowned,
			},
		},
		{
			name:      "to config, ccm",
			config:    toConfig,
			component: "cloud-controller-manager",
			expectResult: map[string]FilterResult{
				"cloud-node-controller":           ControllerNonMigrated,
				"route-controller":                ControllerMigrated,
				"service-controller":              ControllerMigrated,
				"cloud-node-lifecycle-controller": ControllerMigrated,
			},
		},
		{
			name:      "wildcard config, kcm",
			config:    wildcardConfig,
			component: "kube-controller-manager",
			expectResult: map[string]FilterResult{
				"deployment-controller":           ControllerNonMigrated, // KCM only
				"route-controller":                ControllerMigrated,
				"service-controller":              ControllerMigrated,
				"cloud-node-lifecycle-controller": ControllerMigrated,
			},
		},
		{
			name:      "wildcard config, ccm",
			config:    wildcardConfig,
			component: "cloud-controller-manager",
			expectResult: map[string]FilterResult{
				"cloud-node-controller":           ControllerNonMigrated, // CCM only
				"route-controller":                ControllerMigrated,
				"service-controller":              ControllerMigrated,
				"cloud-node-lifecycle-controller": ControllerMigrated,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			migrator := NewLeaderMigrator(tc.config, tc.component)
			for name, expected := range tc.expectResult {
				if result := migrator.FilterFunc(name); expected != result {
					t.Errorf("controller %s, expect %v, got %v", name, expected, result)
				}
			}
		})
	}
}
