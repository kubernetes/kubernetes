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

	federationapi "k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	KubeAPIQPS   = 20.0
	KubeAPIBurst = 30
	DefaultFederationName = "federation"
)

/*
cluster keeps track of the assorted objects and state related to each cluster
in the federation
*/
type cluster struct {
	name string
	*release_1_3.Clientset
	namespaceCreated bool    // Did we need to create a new namespace in this cluster?  If so, we should delete it.
	backendPod       *v1.Pod // The backend pod, if one's been created.
}


func createClusterObjectOrFail(f *framework.Framework, context *framework.E2EContext) {
	framework.Logf("Creating cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
	cluster := federationapi.Cluster{
		ObjectMeta: api.ObjectMeta{
			Name: context.Name,
		},
		Spec: federationapi.ClusterSpec{
			ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
				{
					ClientCIDR:    "0.0.0.0/0",
					ServerAddress: context.Cluster.Cluster.Server,
				},
			},
			SecretRef: &api.LocalObjectReference{
				// Note: Name must correlate with federation build script secret name,
				//       which currently matches the cluster name.
				//       See federation/cluster/common.sh:132
				Name: context.Name,
			},
		},
	}
	_, err := f.FederationClientset.Federation().Clusters().Create(&cluster)
	framework.ExpectNoError(err, fmt.Sprintf("creating cluster: %+v", err))
	framework.Logf("Successfully created cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
}

func clusterIsReadyOrFail(f *framework.Framework, context *framework.E2EContext) {
	c, err := f.FederationClientset.Federation().Clusters().Get(context.Name)
	framework.ExpectNoError(err, fmt.Sprintf("get cluster: %+v", err))
	if c.ObjectMeta.Name != context.Name {
		framework.Failf("cluster name does not match input context: actual=%+v, expected=%+v", c, context)
	}
	err = isReady(context.Name, f.FederationClientset)
	framework.ExpectNoError(err, fmt.Sprintf("unexpected error in verifying if cluster %s is ready: %+v", context.Name, err))
	framework.Logf("Cluster %s is Ready", context.Name)
}

func waitforclustersReadness(f *framework.Framework, clusterSize int) {
	var clusterList *federationapi.ClusterList
	if err := wait.PollImmediate(framework.Poll, FederatedIngressTimeout, func() (bool, error) {
		var err error
		clusterList, err = f.FederationClientset.Federation().Clusters().List(api.ListOptions{})
		if err != nil {
			return false, err
		}
		framework.Logf("%d clusters registered, waiting for %d", len(clusterList.Items), clusterSize)
		if len(clusterList.Items) == clusterSize {
			return true, nil
		}
		return false, nil
	}); err != nil {
		framework.Failf("Failed to list registered clusters: %+v", err)
	}
}

func createClientsetForCluster(c federationapi.Cluster, i int, userAgentName string) *release_1_3.Clientset {
	kubecfg, err := clientcmd.LoadFromFile(framework.TestContext.KubeConfig)
	framework.ExpectNoError(err, "error loading KubeConfig: %v", err)

	cfgOverride := &clientcmd.ConfigOverrides{
		ClusterInfo: clientcmdapi.Cluster{
			Server: c.Spec.ServerAddressByClientCIDRs[0].ServerAddress,
		},
	}
	ccfg := clientcmd.NewNonInteractiveClientConfig(*kubecfg, c.Name, cfgOverride, clientcmd.NewDefaultClientConfigLoadingRules())
	cfg, err := ccfg.ClientConfig()
	framework.ExpectNoError(err, "Error creating client config in cluster #%d (%q)", i, c.Name)

	cfg.QPS = KubeAPIQPS
	cfg.Burst = KubeAPIBurst
	return release_1_3.NewForConfigOrDie(restclient.AddUserAgent(cfg, userAgentName))
}

func createNamespaceInClusters(clusters map[string]*cluster, f *framework.Framework) {
	for name, c := range clusters {
		// The e2e Framework created the required namespace in one of the clusters, but we need to create it in all the others, if it doesn't yet exist.
		if _, err := c.Clientset.Core().Namespaces().Get(f.Namespace.Name); errors.IsNotFound(err) {
			ns := &v1.Namespace{
				ObjectMeta: v1.ObjectMeta{
					Name: f.Namespace.Name,
				},
			}
			_, err := c.Clientset.Core().Namespaces().Create(ns)
			if err == nil {
				c.namespaceCreated = true
			}
			framework.ExpectNoError(err, "Couldn't create the namespace %s in cluster %q", f.Namespace.Name, name)
			framework.Logf("Namespace %s created in cluster %q", f.Namespace.Name, name)
		} else if err != nil {
			framework.Logf("Couldn't create the namespace %s in cluster %q: %v", f.Namespace.Name, name, err)
		}
	}
}
func teardownClusters(clusters map[string]*cluster, f *framework.Framework) {
	for name, c := range clusters {
		if c.namespaceCreated {
			if _, err := c.Clientset.Core().Namespaces().Get(f.Namespace.Name); !errors.IsNotFound(err) {
				err := c.Clientset.Core().Namespaces().Delete(f.Namespace.Name, &api.DeleteOptions{})
				framework.ExpectNoError(err, "Couldn't delete the namespace %s in cluster %q: %v", f.Namespace.Name, name, err)
			}
			framework.Logf("Namespace %s deleted in cluster %q", f.Namespace.Name, name)
		}
	}

	// Delete the registered clusters in the federation API server.
	clusterList, err := f.FederationClientset.Federation().Clusters().List(api.ListOptions{})
	framework.ExpectNoError(err, "Error listing clusters")
	for _, cluster := range clusterList.Items {
		err := f.FederationClientset.Federation().Clusters().Delete(cluster.Name, &api.DeleteOptions{})
		framework.ExpectNoError(err, "Error deleting cluster %q", cluster.Name)
	}
}

// can not be moved to util, as By and Expect must be put in Ginkgo test unit
func setupClusters(clusters map[string]*cluster, userAgentName, federationName string, f *framework.Framework) string {

	contexts := f.GetUnderlyingFederatedContexts()

	for _, context := range contexts {
		createClusterObjectOrFail(f, &context)
	}

	var clusterList *federationapi.ClusterList
	By("Obtaining a list of all the clusters")
	waitforclustersReadness(f, len(contexts))

	framework.Logf("Checking that %d clusters are Ready", len(contexts))
	for _, context := range contexts {
		clusterIsReadyOrFail(f, &context)
	}
	framework.Logf("%d clusters are Ready", len(contexts))

	primaryClusterName := clusterList.Items[0].Name
	By(fmt.Sprintf("Labeling %q as the first cluster", primaryClusterName))
	for i, c := range clusterList.Items {
		framework.Logf("Creating a clientset for the cluster %s", c.Name)
		Expect(framework.TestContext.KubeConfig).ToNot(Equal(""), "KubeConfig must be specified to load clusters' client config")
		clusters[c.Name] = &cluster{c.Name, createClientsetForCluster(c, i, userAgentName), false, nil}
	}
	createNamespaceInClusters(clusters, f)
	return primaryClusterName
}
