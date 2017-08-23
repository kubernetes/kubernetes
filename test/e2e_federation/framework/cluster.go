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
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const federatedClustersWaitTimeout = 1 * time.Minute

// ClusterSlice is a slice of clusters
type ClusterSlice []*Cluster

// Cluster keeps track of the name and client of a cluster in the federation
type Cluster struct {
	Name string
	*kubeclientset.Clientset
}

// registeredClustersFromConfig configures clientsets for registered clusters from the e2e kubeconfig
func registeredClustersFromConfig(f *Framework) ClusterSlice {
	contexts := f.GetUnderlyingFederatedContexts()

	By("Obtaining a list of all the clusters")
	clusterList := waitForAllRegisteredClusters(f, len(contexts))

	framework.Logf("Checking that %d clusters are Ready", len(contexts))
	for _, context := range contexts {
		ClusterIsReadyOrFail(f, context.Name)

	}
	framework.Logf("%d clusters are Ready", len(contexts))

	clusters := ClusterSlice{}
	for i, c := range clusterList.Items {
		framework.Logf("Creating a clientset for the cluster %s", c.Name)
		Expect(framework.TestContext.KubeConfig).ToNot(Equal(""), "KubeConfig must be specified to load clusters' client config")
		config := restConfigFromContext(c, i)
		clusters = append(clusters, &Cluster{
			Name:      c.Name,
			Clientset: clientsetFromConfig(f, config, c.Spec.ServerAddressByClientCIDRs[0].ServerAddress),
		})

	}
	waitForNamespaceInFederatedClusters(clusters, f.FederationNamespace.Name)
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

func restConfigFromContext(c federationapi.Cluster, i int) *restclient.Config {
	kubecfg, err := clientcmd.LoadFromFile(framework.TestContext.KubeConfig)
	framework.ExpectNoError(err, "error loading KubeConfig: %v", err)

	ccfg := clientcmd.NewNonInteractiveClientConfig(*kubecfg, c.Name, &clientcmd.ConfigOverrides{}, clientcmd.NewDefaultClientConfigLoadingRules())
	cfg, err := ccfg.ClientConfig()
	framework.ExpectNoError(err, "Error creating client config in cluster #%d (%q)", i, c.Name)
	return cfg
}

func clientsetFromConfig(f *Framework, cfg *restclient.Config, host string) *kubeclientset.Clientset {
	cfg.Host = host
	cfg.QPS = f.Framework.Options.ClientQPS
	cfg.Burst = f.Framework.Options.ClientBurst
	return kubeclientset.NewForConfigOrDie(restclient.AddUserAgent(cfg, "federation-e2e"))
}

// waitForNamespaceInFederatedClusters waits for the federated namespace to be created in federated clusters
func waitForNamespaceInFederatedClusters(clusters ClusterSlice, nsName string) {
	for _, c := range clusters {
		name := c.Name
		By(fmt.Sprintf("Waiting for namespace %q to be created in cluster %q", nsName, name))
		err := wait.PollImmediate(framework.Poll, FederatedDefaultTestTimeout, func() (bool, error) {
			_, err := c.Clientset.CoreV1().Namespaces().Get(nsName, metav1.GetOptions{})
			if errors.IsNotFound(err) {
				return false, nil
			} else if err != nil {
				framework.Logf("An error occurred waiting for namespace %q to be created in cluster %q: %v", nsName, name, err)
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

// Cache the cluster config to avoid having to retrieve it for each test
type clusterConfig struct {
	name   string
	host   string
	config []byte
}

var cachedClusterConfigs []*clusterConfig

// registeredClustersFromSecrets configures clientsets for cluster access from secrets in the host cluster
func registeredClustersFromSecrets(f *Framework) ClusterSlice {
	if cachedClusterConfigs == nil {
		cachedClusterConfigs = clusterConfigFromSecrets(f)
	}

	clusters := ClusterSlice{}
	for _, clusterConf := range cachedClusterConfigs {
		restConfig := restConfigForCluster(clusterConf)
		clientset := clientsetFromConfig(f, restConfig, clusterConf.host)
		clusters = append(clusters, &Cluster{
			Name:      clusterConf.name,
			Clientset: clientset,
		})
	}

	waitForNamespaceInFederatedClusters(clusters, f.FederationNamespace.Name)

	return clusters
}

// clusterConfigFromSecrets retrieves cluster configuration from
// secrets in the host cluster
func clusterConfigFromSecrets(f *Framework) []*clusterConfig {
	By("Obtaining a list of registered clusters")
	clusterList, err := f.FederationClientset.Federation().Clusters().List(metav1.ListOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("Error retrieving list of federated clusters: %+v", err))
	if len(clusterList.Items) == 0 {
		framework.Failf("No registered clusters found")
	}

	clusterConfigs := []*clusterConfig{}
	for _, c := range clusterList.Items {
		ClusterIsReadyOrFail(f, c.Name)
		config := clusterConfigFromSecret(f, c.Name, c.Spec.SecretRef.Name)
		clusterConfigs = append(clusterConfigs, &clusterConfig{
			name:   c.Name,
			host:   c.Spec.ServerAddressByClientCIDRs[0].ServerAddress,
			config: config,
		})
	}

	return clusterConfigs
}

// clusterConfigFromSecret retrieves configuration for a accessing a
// cluster from a secret in the host cluster
func clusterConfigFromSecret(f *Framework, clusterName string, secretName string) []byte {
	By(fmt.Sprintf("Loading configuration for cluster %q", clusterName))
	namespace := framework.FederationSystemNamespace()
	secret, err := f.Framework.ClientSet.Core().Secrets(namespace).Get(secretName, metav1.GetOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("Error loading config secret \"%s/%s\" for cluster %q: %+v", namespace, secretName, clusterName, err))

	config, ok := secret.Data[util.KubeconfigSecretDataKey]
	if !ok || len(config) == 0 {
		framework.Failf("Secret \"%s/%s\" for cluster %q has no value for key %q", namespace, secretName, clusterName, util.KubeconfigSecretDataKey)
	}

	return config
}

// restConfigForCluster creates a rest client config for the given cluster config
func restConfigForCluster(clusterConf *clusterConfig) *restclient.Config {
	cfg, err := clientcmd.Load(clusterConf.config)
	framework.ExpectNoError(err, fmt.Sprintf("Error loading configuration for cluster %q: %+v", clusterConf.name, err))

	restConfig, err := clientcmd.NewDefaultClientConfig(*cfg, &clientcmd.ConfigOverrides{}).ClientConfig()
	framework.ExpectNoError(err, fmt.Sprintf("Error creating client for cluster %q: %+v", clusterConf.name, err))
	return restConfig
}

func GetZoneFromClusterName(clusterName string) string {
	// Ref: https://github.com/kubernetes/kubernetes/blob/master/cluster/kube-util.sh#L55
	prefix := "federation-e2e-" + framework.TestContext.Provider + "-"
	return strings.TrimPrefix(clusterName, prefix)
}
