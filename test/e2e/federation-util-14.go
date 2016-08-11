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
	_, err := f.FederationClientset_1_4.Federation().Clusters().Create(&cluster)
	framework.ExpectNoError(err, fmt.Sprintf("creating cluster: %+v", err))
	framework.Logf("Successfully created cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
}

func clusterIsReadyOrFail_14(f *framework.Framework, context *framework.E2EContext) {
	c, err := f.FederationClientset_1_4.Federation().Clusters().Get(context.Name)
	framework.ExpectNoError(err, fmt.Sprintf("get cluster: %+v", err))
	if c.ObjectMeta.Name != context.Name {
		framework.Failf("cluster name does not match input context: actual=%+v, expected=%+v", c, context)
	}
	err = isReady(context.Name, f.FederationClientset)
	framework.ExpectNoError(err, fmt.Sprintf("unexpected error in verifying if cluster %s is ready: %+v", context.Name, err))
	framework.Logf("Cluster %s is Ready", context.Name)
}

func getAllClustersOrFail_14(f *framework.Framework) []*federation_api.Cluster {
	contexts := f.GetUnderlyingFederatedContexts()

	for _, context := range contexts {
		createClusterObjectOrFail_14(f, &context)
	}

	var clusterList *federation_api.ClusterList
	framework.Logf("Obtaining a list of all the clusters")
	if err := wait.PollImmediate(framework.Poll, time.Minute, func() (bool, error) {
		var err error
		clusterList, err = f.FederationClientset_1_4.Federation().Clusters().List(api.ListOptions{})
		if err != nil {
			return false, err
		}
		framework.Logf("%d clusters registered, waiting for %d", len(clusterList.Items), len(contexts))
		if len(clusterList.Items) == len(contexts) {
			return true, nil
		}
		return false, nil
	}); err != nil {
		framework.Failf("Failed to list registered clusters: %+v", err)
	}

	framework.Logf("Checking that %d clusters are Ready", len(contexts))
	for _, context := range contexts {
		clusterIsReadyOrFail_14(f, &context)
	}
	framework.Logf("%d clusters are Ready", len(contexts))
	result := make([]*federation_api.Cluster, 0, len(contexts))
	for i := range clusterList.Items {
		result[i] = &clusterList.Items[i]
	}
	return result
}
