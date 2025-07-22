/*
Copyright 2025 The Kubernetes Authors.

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

package dra

import (
	"context"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// deployDevicePlugin deploys https://github.com/kubernetes/kubernetes/tree/111a2a0d2dfe13639724506f674bc4f342ccfbab/test/images/sample-device-plugin
// on the chosen nodes.
//
// A test using this must run serially because the example plugin uses the hard-coded "example.com/resource"
// extended resource name (returned as result) and deploying it twice for the same nodes from different
// tests would conflict.
func deployDevicePlugin(ctx context.Context, f *framework.Framework, nodeNames []string) v1.ResourceName {
	err := utils.CreateFromManifests(ctx, f, f.Namespace, func(item interface{}) error {
		switch item := item.(type) {
		case *appsv1.DaemonSet:
			item.Spec.Template.Spec.Affinity = &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "kubernetes.io/hostname",
										Operator: v1.NodeSelectorOpIn,
										Values:   nodeNames,
									},
								},
							},
						},
					},
				},
			}
		}
		return nil
	}, "test/e2e/testing-manifests/sample-device-plugin/sample-device-plugin.yaml")
	framework.ExpectNoError(err, "deploy example device plugin DaemonSet")

	// Hard-coded in https://github.com/kubernetes/kubernetes/blob/111a2a0d2dfe13639724506f674bc4f342ccfbab/test/images/sample-device-plugin/sampledeviceplugin.go#L34C17-L34C39.
	return "example.com/resource"
}
