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
	f := framework.NewDefaultFederatedFramework("service")

	BeforeEach(func() {
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

		By("Obtaining a list of all the clusters")
		clusterList, err := f.FederationClientset.Federation().Clusters().List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("%d clusters found", len(clusterList.Items))

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

	It("should be able to discover a federated service", func() {
		framework.SkipUnlessFederated(f.Client)

		createService(f.FederationClientset, clusterClientSets, f.Namespace.Name)

		discoverService(f)
	})

	It("should be able to discover a non-local federated service", func() {
		framework.SkipUnlessFederated(f.Client)

		createService(f.FederationClientset, clusterClientSets, f.Namespace.Name)

		// Delete a federated service shard in the default e2e Kubernetes cluster.
		err := f.Clientset_1_3.Core().Services(f.Namespace.Name).Delete(FederatedServiceName, &api.DeleteOptions{})
		Expect(err).NotTo(HaveOccurred())
		waitForFederatedServiceShard(f.Clientset_1_3, f.Namespace.Name, nil, 0)

		discoverService(f)
	})
})

func waitForFederatedServiceShard(cs *release_1_3.Clientset, namespace string, service *api.Service, numSvcs int) {
	By("Fetching a federated service shard")
	var clSvcList *v1.ServiceList
	for t := time.Now(); time.Since(t) < FederatedServiceTimeout; time.Sleep(framework.Poll) {
		var err error
		clSvcList, err = cs.Core().Services(namespace).List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		n := len(clSvcList.Items)
		if n == numSvcs {
			break
		}
		framework.Logf("%d services found, trying again in %s", n, framework.Poll)
	}

	Expect(len(clSvcList.Items)).To(Equal(numSvcs))
	if numSvcs > 0 {
		// Renaming for clarity/readability
		clSvc := clSvcList.Items[0]
		Expect(clSvc.Name).To(Equal(service.Name))
		Expect(clSvc.Spec).To(Equal(service.Spec))
	}
}

func createService(fcs *federation_internalclientset.Clientset, clusterClientSets []*release_1_3.Clientset, ns string) {
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
	_, err := fcs.Core().Services(ns).Create(service)
	Expect(err).NotTo(HaveOccurred())
	for _, cs := range clusterClientSets {
		waitForFederatedServiceShard(cs, ns, service, 1)
	}
}

func discoverService(f *framework.Framework) {
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
					Command: []string{"sh", "-c", "nslookup", FederatedServiceName},
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}
	f.TestContainerOutputRegexp("federated service discovery", pod, 0, []string{
		`Name:\s+` + FederatedDNS1123Regexp + `\nAddress \d+:\s+` + FederatedIPAddrRegexp + `\s+` + FederatedDNS1123Regexp,
	})
}
