/*
Copyright 2018 The Kubernetes Authors.

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

package v1alpha1

import (
	"reflect"
	"testing"

	componentconfigtesting "k8s.io/apimachinery/pkg/apis/config/testing"
	"k8s.io/apimachinery/pkg/util/sets"
	kubectrlmgrconfigv1alpha1 "k8s.io/kube-controller-manager/config/v1alpha1"
)

func TestComponentConfigSetup(t *testing.T) {
	pkginfo := &componentconfigtesting.ComponentConfigPackage{
		ComponentName:      "cloud-controller-manager",
		GroupName:          GroupName,
		SchemeGroupVersion: SchemeGroupVersion,
		AddToScheme:        AddToScheme,
		// TODO: This whitelist should go away, and JSON tags should be applied on the external type
		// to make it serializable
		AllowedNoJSONTags: map[reflect.Type]sets.String{
			reflect.TypeOf(CloudControllerManagerConfiguration{}): sets.NewString(
				"Generic",
				"KubeCloudShared",
				"NodeStatusUpdateFrequency",
				"ServiceController",
			),
			reflect.TypeOf(kubectrlmgrconfigv1alpha1.GenericControllerManagerConfiguration{}): sets.NewString(
				"Address",
				"ClientConnection",
				"ControllerStartInterval",
				"Controllers",
				"Debugging",
				"LeaderElection",
				"MinResyncPeriod",
				"Port",
			),
			reflect.TypeOf(kubectrlmgrconfigv1alpha1.KubeCloudSharedConfiguration{}): sets.NewString(
				"AllocateNodeCIDRs",
				"AllowUntaggedCloud",
				"CIDRAllocatorType",
				"CloudProvider",
				"ClusterCIDR",
				"ClusterName",
				"ConfigureCloudRoutes",
				"ExternalCloudVolumePlugin",
				"NodeMonitorPeriod",
				"NodeSyncPeriod",
				"RouteReconciliationPeriod",
				"UseServiceAccountCredentials",
			),
			reflect.TypeOf(kubectrlmgrconfigv1alpha1.ServiceControllerConfiguration{}): sets.NewString(
				"ConcurrentServiceSyncs",
			),
		},
	}
	if err := componentconfigtesting.VerifyExternalTypePackage(pkginfo); err != nil {
		t.Errorf("failed TestComponentConfigSetup: %v", err)
	}
}
