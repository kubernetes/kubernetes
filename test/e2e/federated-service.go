/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"os"
	"time"

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_internalclientset"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	UserAgentName = "federation-e2e-service-controller"
	// TODO(madhusudancs): Using the same values as defined in the federated
	// service controller. Replace it with the values from the e2e framework.
	KubeAPIQPS   = 20.0
	KubeAPIBurst = 30

	FederatedServiceTimeout = 5 * time.Minute

	FederatedServiceName = "federated-service"
	FederatedServicePod  = "federated-service-test-pod"

	// TODO: Only suppoprts IPv4 addresses. Also add a regexp for IPv6 addresses.
	FederatedIPAddrRegexp  = `(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])`
	FederatedDNS1123Regexp = `([a-z0-9]([-a-z0-9]*[a-z0-9])?\.)*([a-z0-9]([-a-z0-9]*[a-z0-9])?)`
)

var _ = framework.KubeDescribe("Service [Feature:Federation]", func() {
	var clusterClientSets []*release_1_3.Clientset
	var federationName string
	f := framework.NewDefaultFederatedFramework("service")

	BeforeEach(func() {
		framework.SkipUnlessFederated(f.Client)

		// TODO: Federation API server should be able to answer this.
		if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
			// Tests cannot proceed without this value, so fail early here.
			framework.Failf("FEDERATION_NAME environment variable must be set")
		}

		contexts := f.GetUnderlyingFederatedContexts()

		for _, context := range contexts {
			framework.Logf("Creating cluster object: %s (%s)", context.Name, context.Cluster.Cluster.Server)
			cluster := federation.Cluster{
				ObjectMeta: api.ObjectMeta{
					Name: context.Name,
				},
				Spec: federation.ClusterSpec{
					ServerAddressByClientCIDRs: []federation.ServerAddressByClientCIDR{
						{
							ClientCIDR:    "0.0.0.0/0",
							ServerAddress: context.Cluster.Cluster.Server,
						},
					},
				},
			}
			_, err := f.FederationClientset.Federation().Clusters().Create(&cluster)
			framework.ExpectNoError(err, "Creating cluster")
		}

		var clusterList *federation.ClusterList
		By("Obtaining a list of all the clusters")
		if err := wait.PollImmediate(framework.Poll, FederatedServiceTimeout, func() (bool, error) {
			var err error
			clusterList, err = f.FederationClientset.Federation().Clusters().List(api.ListOptions{})
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

		for _, cluster := range clusterList.Items {
			framework.Logf("Creating a clientset for the cluster %s", cluster.Name)

			Expect(framework.TestContext.KubeConfig).ToNot(Equal(""), "KubeConfig must be specified to load clusters' client config")
			kubecfg, err := clientcmd.LoadFromFile(framework.TestContext.KubeConfig)
			framework.ExpectNoError(err, "error loading KubeConfig: %v", err)

			cfgOverride := &clientcmd.ConfigOverrides{
				ClusterInfo: clientcmdapi.Cluster{
					Server: cluster.Spec.ServerAddressByClientCIDRs[0].ServerAddress,
				},
			}
			ccfg := clientcmd.NewNonInteractiveClientConfig(*kubecfg, cluster.Name, cfgOverride, clientcmd.NewDefaultClientConfigLoadingRules())
			cfg, err := ccfg.ClientConfig()
			Expect(err).NotTo(HaveOccurred())

			cfg.QPS = KubeAPIQPS
			cfg.Burst = KubeAPIBurst
			clset := release_1_3.NewForConfigOrDie(restclient.AddUserAgent(cfg, UserAgentName))
			clusterClientSets = append(clusterClientSets, clset)
		}
	})

	AfterEach(func() {
		framework.SkipUnlessFederated(f.Client)

		// Delete the registered clusters in the federation API server.
		clusterList, err := f.FederationClientset.Federation().Clusters().List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		for _, cluster := range clusterList.Items {
			err := f.FederationClientset.Federation().Clusters().Delete(cluster.Name, &api.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred())
		}
	})

	It("should be able to discover a federated service", func() {
		framework.SkipUnlessFederated(f.Client)

		createService(f.FederationClientset, clusterClientSets, f.Namespace.Name)

		svcDNSNames := []string{
			FederatedServiceName,
			fmt.Sprintf("%s.%s", FederatedServiceName, f.Namespace.Name),
			fmt.Sprintf("%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name),
			fmt.Sprintf("%s.%s.%s", FederatedServiceName, f.Namespace.Name, federationName),
			fmt.Sprintf("%s.%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name, federationName),
		}
		for _, name := range svcDNSNames {
			discoverService(f, name, true)
		}
	})

	It("should be able to discover a non-local federated service", func() {
		framework.SkipUnlessFederated(f.Client)

		createService(f.FederationClientset, clusterClientSets, f.Namespace.Name)

		// Delete a federated service shard in the default e2e Kubernetes cluster.
		err := f.Clientset_1_3.Core().Services(f.Namespace.Name).Delete(FederatedServiceName, &api.DeleteOptions{})
		Expect(err).NotTo(HaveOccurred())
		waitForFederatedServiceShard(f.Clientset_1_3, f.Namespace.Name, nil, 0)

		localSvcDNSNames := []string{
			FederatedServiceName,
			fmt.Sprintf("%s.%s", FederatedServiceName, f.Namespace.Name),
			fmt.Sprintf("%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name),
		}
		for _, name := range localSvcDNSNames {
			discoverService(f, name, false)
		}

		svcDNSNames := []string{
			fmt.Sprintf("%s.%s.%s", FederatedServiceName, f.Namespace.Name, federationName),
			fmt.Sprintf("%s.%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name, federationName),
		}
		for _, name := range svcDNSNames {
			discoverService(f, name, true)
		}
	})
})

// waitForFederatedServiceShard waits until the number of shards of a given federated
// service reaches the expected value, i.e. numSvcs in the given individual Kubernetes
// cluster. If the shard count, i.e. numSvcs is expected to be at least one, then
// it also checks if the first shard's name and spec matches that of the given service.
func waitForFederatedServiceShard(cs *release_1_3.Clientset, namespace string, service *api.Service, numSvcs int) {
	By("Fetching a federated service shard")
	var clSvcList *v1.ServiceList
	if err := wait.PollImmediate(framework.Poll, FederatedServiceTimeout, func() (bool, error) {
		var err error
		clSvcList, err = cs.Core().Services(namespace).List(api.ListOptions{})
		if err != nil {
			return false, err
		}
		n := len(clSvcList.Items)
		if n == numSvcs {
			return true, nil
		}
		framework.Logf("%d services found, waiting for %d, trying again in %s", n, numSvcs, framework.Poll)
		return false, nil
	}); err != nil {
		framework.Failf("Failed to list registered clusters: %+v", err)
	}

	if numSvcs > 0 && service != nil {
		// Renaming for clarity/readability
		clSvc := clSvcList.Items[0]
		Expect(clSvc.Name).To(Equal(service.Name))
		Expect(clSvc.Spec).To(Equal(service.Spec))
	}
}

func createService(fcs *federation_internalclientset.Clientset, clusterClientSets []*release_1_3.Clientset, namespace string) {
	By("Creating a federated service")
	labels := map[string]string{
		"foo": "bar",
	}

	svc1port := "svc1"
	svc2port := "svc2"

	service := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: FederatedServiceName,
		},
		Spec: api.ServiceSpec{
			Selector: labels,
			Ports: []api.ServicePort{
				{
					Name:       "portname1",
					Port:       80,
					TargetPort: intstr.FromString(svc1port),
				},
				{
					Name:       "portname2",
					Port:       81,
					TargetPort: intstr.FromString(svc2port),
				},
			},
		},
	}
	_, err := fcs.Core().Services(namespace).Create(service)
	Expect(err).NotTo(HaveOccurred())
	for _, cs := range clusterClientSets {
		waitForFederatedServiceShard(cs, namespace, service, 1)
	}
}

func discoverService(f *framework.Framework, name string, exists bool) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:   FederatedServicePod,
			Labels: map[string]string{"name": FederatedServicePod},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "federated-service-discovery-container",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"sh", "-c", "nslookup", name},
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}
	if exists {
		f.TestContainerOutputRegexp("federated service discovery", pod, 0, []string{
			`Name:\s+` + FederatedDNS1123Regexp + `\nAddress \d+:\s+` + FederatedIPAddrRegexp + `\s+` + FederatedDNS1123Regexp,
		})
	} else {
		f.TestContainerOutputRegexp("federated service discovery", pod, 0, []string{
			`nslookup: can't resolve '` + name + `'`,
		})
	}
}
