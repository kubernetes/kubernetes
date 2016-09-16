/*
Copyright 2016 The Kubernetes Authors.

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

package e2e

import (
	"fmt"
	"time"

	federation_api "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	api "k8s.io/kubernetes/pkg/api"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

func createClusterObjectOrFail_14(f *framework.Framework, context *framework.E2EContext) {
	framework.Logf("Looking up cluster: %s", context.Name)
	foundCluster, err := f.FederationClientset_1_4.Federation().Clusters().Get(context.Name)
	if err == nil && foundCluster != nil {
		return
	}

	framework.Logf("Creating cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
	cluster := federation_api.Cluster{
		ObjectMeta: api_v1.ObjectMeta{
			Name: context.Name,
		},
		Spec: federation_api.ClusterSpec{
			ServerAddressByClientCIDRs: []federation_api.ServerAddressByClientCIDR{
				{
					ClientCIDR:    "0.0.0.0/0",
					ServerAddress: context.Cluster.Cluster.Server,
				},
			},
			SecretRef: &api_v1.LocalObjectReference{
				// Note: Name must correlate with federation build script secret name,
				//       which currently matches the cluster name.
				//       See federation/cluster/common.sh:132
				Name: context.Name,
			},
		},
	}
	_, err = f.FederationClientset_1_4.Federation().Clusters().Create(&cluster)
	framework.ExpectNoError(err, fmt.Sprintf("creating cluster: %+v", err))
	framework.Logf("Successfully created cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
}

func buildClustersOrFail_14(f *framework.Framework) []*federation_api.Cluster {
	contexts := f.GetUnderlyingFederatedContexts()

	for _, context := range contexts {
		createClusterObjectOrFail_14(f, &context)
	}

	// Wait for all clusters to become ready for up to 5 min.
	if err := wait.PollImmediate(5*time.Second, 5*time.Minute, func() (bool, error) {
		for _, context := range contexts {
			cluster, err := f.FederationClientset_1_4.Federation().Clusters().Get(context.Name)
			if err != nil {
				return false, err
			}
			ready := false
			for _, condition := range cluster.Status.Conditions {
				if condition.Type == federation_api.ClusterReady && condition.Status == api_v1.ConditionTrue {
					ready = true
				}
			}
			if !ready {
				return false, nil
			}
		}
		return true, nil
	}); err != nil {
		framework.Failf("Not all clusters are ready: %v", err)
	}

	clusterList, err := f.FederationClientset_1_4.Federation().Clusters().List(api.ListOptions{})
	if err != nil {
		framework.Failf("Error in get clusters: %v", err)
	}
	result := make([]*federation_api.Cluster, 0, len(contexts))
	for i := range clusterList.Items {
		result = append(result, &clusterList.Items[i])
	}
	return result
}
