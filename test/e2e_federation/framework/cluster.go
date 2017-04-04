/*
Copyright 2017 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	KubeAPIQPS   float32 = 20.0
	KubeAPIBurst         = 30

	userAgentName = "federation-e2e"

	federatedNamespaceTimeout    = 5 * time.Minute
	federatedClustersWaitTimeout = 1 * time.Minute
)

// ClusterSlice is a slice of clusters
type ClusterSlice []*Cluster

// Cluster keeps track of the name and client of a cluster in the federation
type Cluster struct {
	Name string
	*kubeclientset.Clientset
}

func getRegisteredClusters(f *Framework) ClusterSlice {
	contexts := f.GetUnderlyingFederatedContexts()

	By("Obtaining a list of all the clusters")
	clusterList := waitForAllRegisteredClusters(f, len(contexts))

	framework.Logf("Checking that %d clusters are Ready", len(contexts))
	for _, context := range contexts {
		ClusterIsReadyOrFail(f, &context)
	}
	framework.Logf("%d clusters are Ready", len(contexts))

	clusters := ClusterSlice{}
	for i, c := range clusterList.Items {
		framework.Logf("Creating a clientset for the cluster %s", c.Name)
		Expect(framework.TestContext.KubeConfig).ToNot(Equal(""), "KubeConfig must be specified to load clusters' client config")
		clusters = append(clusters, &Cluster{
			Name:      c.Name,
			Clientset: createClientsetForCluster(c, i, userAgentName),
		})
	}
	waitForNamespaceInFederatedClusters(clusters, f.FederationNamespace.Name, federatedNamespaceTimeout)
	return clusters
}

// waitForAllRegisteredClusters waits for all clusters defined in e2e context to be created
// return ClusterList until the listed cluster items equals clusterCount
func waitForAllRegisteredClusters(f *Framework, clusterCount int) *federationapi.ClusterList {
	var clusterList *federationapi.ClusterList
	if err := wait.PollImmediate(framework.Poll, federatedClustersWaitTimeout, func() (bool, error) {
		var err error
		clusterList, err = f.FederationClientset.Federation().Clusters().List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		framework.Logf("%d clusters registered, waiting for %d", len(clusterList.Items), clusterCount)
		if len(clusterList.Items) == clusterCount {
			return true, nil
		}
		return false, nil
	}); err != nil {
		framework.Failf("Failed to list registered clusters: %+v", err)
	}
	return clusterList
}

func createClientsetForCluster(c federationapi.Cluster, i int, userAgentName string) *kubeclientset.Clientset {
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
	return kubeclientset.NewForConfigOrDie(restclient.AddUserAgent(cfg, userAgentName))
}

// waitForNamespaceInFederatedClusters waits for the federated namespace to be created in federated clusters
func waitForNamespaceInFederatedClusters(clusters ClusterSlice, nsName string, timeout time.Duration) {
	for _, c := range clusters {
		name := c.Name
		err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
			_, err := c.Clientset.Core().Namespaces().Get(nsName, metav1.GetOptions{})
			if err != nil {
				By(fmt.Sprintf("Waiting for namespace %q to be created in cluster %q, err: %v", nsName, name, err))
				return false, nil
			}
			By(fmt.Sprintf("Namespace %q exists in cluster %q", nsName, name))
			return true, nil
		})
		framework.ExpectNoError(err, "Failed to verify federated namespace %q creation in cluster %q", nsName, name)
	}
}

// ClusterIsReadyOrFail checks whether the federated cluster of the provided context is ready
func ClusterIsReadyOrFail(f *Framework, context *E2EContext) {
	c, err := f.FederationClientset.Federation().Clusters().Get(context.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("get cluster: %+v", err))
	if c.ObjectMeta.Name != context.Name {
		framework.Failf("cluster name does not match input context: actual=%+v, expected=%+v", c, context)
	}
	err = isReady(context.Name, f.FederationClientset)
	framework.ExpectNoError(err, fmt.Sprintf("unexpected error in verifying if cluster %s is ready: %+v", context.Name, err))
	framework.Logf("Cluster %s is Ready", context.Name)
}

// Verify that the cluster is marked ready.
func isReady(clusterName string, clientset *fedclientset.Clientset) error {
	return wait.PollImmediate(time.Second, 5*time.Minute, func() (bool, error) {
		c, err := clientset.Federation().Clusters().Get(clusterName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, condition := range c.Status.Conditions {
			if condition.Type == federationapi.ClusterReady && condition.Status == v1.ConditionTrue {
				return true, nil
			}
		}
		return false, nil
	})
}
