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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// ClusterSlice is a slice of clusters
type ClusterSlice []*Cluster

// Cluster keeps track of the name and client of a cluster in the federation
type Cluster struct {
	Name string
	*kubeclientset.Clientset
}

// Cache the cluster config to avoid having to retrieve it for each test
var clusterNames []string
var clusterConfigMap map[string][]byte

func getRegisteredClusters(f *Framework) ClusterSlice {
	if clusterNames == nil {
		clusterNames, clusterConfigMap = getClusterConfig(f)
	}
	clusters := ClusterSlice{}
	for _, clusterName := range clusterNames {
		clientset := clientsetForClusterOrFail(f, clusterName, clusterConfigMap[clusterName])
		clusters = append(clusters, &Cluster{
			Name:      clusterName,
			Clientset: clientset,
		})
	}

	waitForNamespaceInFederatedClusters(clusters, f.FederationNamespace.Name)

	return clusters
}

func getClusterConfig(f *Framework) ([]string, map[string][]byte) {
	By("Obtaining a list of registered clusters")
	clusterList, err := f.FederationClientset.Federation().Clusters().List(metav1.ListOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("Error retrieving list of federated clusters: %+v", err))
	if len(clusterList.Items) == 0 {
		framework.Failf("No registered clusters found")
	}

	configMap := map[string][]byte{}
	names := []string{}
	for _, c := range clusterList.Items {
		names = append(names, c.Name)
		ClusterIsReadyOrFail(f, c.Name)
		configMap[c.Name] = configForClusterOrFail(f, c.Name)
	}

	return names, configMap
}

func configForClusterOrFail(f *Framework, clusterName string) []byte {
	By(fmt.Sprintf("Loading configuration for cluster %q", clusterName))

	// TODO The system namespace should be configurable since it can vary across deployments
	secret, err := f.Framework.ClientSet.Core().Secrets("federation-system").Get(clusterName, metav1.GetOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("Error loading config secret for cluster %q: %+v", clusterName, err))

	data, ok := secret.Data[util.KubeconfigSecretDataKey]
	if !ok || len(data) == 0 {
		framework.Failf("Secret for cluster %q has no value for key %q", clusterName, util.KubeconfigSecretDataKey)
	}

	return data
}

func clientsetForClusterOrFail(f *Framework, clusterName string, data []byte) *kubeclientset.Clientset {
	cfg, err := clientcmd.Load(data)
	framework.ExpectNoError(err, fmt.Sprintf("Error loading configuration for cluster %q: %+v", clusterName, err))

	restConfig, err := clientcmd.NewDefaultClientConfig(*cfg, &clientcmd.ConfigOverrides{}).ClientConfig()
	framework.ExpectNoError(err, fmt.Sprintf("Error creating client for cluster %q: %+v", clusterName, err))

	restConfig.QPS = f.Framework.Options.ClientQPS
	restConfig.Burst = f.Framework.Options.ClientBurst

	return kubeclientset.NewForConfigOrDie(restclient.AddUserAgent(restConfig, "federation-e2e"))
}

// waitForNamespaceInFederatedClusters waits for the federated namespace to be created in federated clusters
func waitForNamespaceInFederatedClusters(clusters ClusterSlice, nsName string) {
	for _, c := range clusters {
		name := c.Name
		err := wait.PollImmediate(framework.Poll, FederatedDefaultTestTimeout, func() (bool, error) {
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

// ClusterIsReadyOrFail checks whether the named cluster is ready
func ClusterIsReadyOrFail(f *Framework, clusterName string) {
	By(fmt.Sprintf("Checking readiness of cluster %q", clusterName))
	err := wait.PollImmediate(framework.Poll, FederatedDefaultTestTimeout, func() (bool, error) {
		c, err := f.FederationClientset.Federation().Clusters().Get(clusterName, metav1.GetOptions{})
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
	framework.ExpectNoError(err, fmt.Sprintf("Unexpected error in verifying if cluster %q is ready: %+v", clusterName, err))
	framework.Logf("Cluster %s is Ready", clusterName)
}
