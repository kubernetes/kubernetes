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
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_3"
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

	FederatedServiceTimeout = 60 * time.Second

	FederatedServiceName = "federated-service"
	FederatedServicePod  = "federated-service-test-pod"

	DefaultFederationName = "federation"
)

var _ = framework.KubeDescribe("[Feature:Federation] Federated Services", func() {
	var clusterClientSets []*release_1_3.Clientset
	var federationName string
	f := framework.NewDefaultFederatedFramework("service")

	BeforeEach(func() {
		framework.SkipUnlessFederated(f.Client)

		// TODO: Federation API server should be able to answer this.
		if federationName = os.Getenv("FEDERATION_NAME"); federationName == "" {
			federationName = DefaultFederationName
		}

		contexts := f.GetUnderlyingFederatedContexts()

		for _, context := range contexts {
			createClusterObjectOrFail(f, &context)
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

		framework.Logf("Checking that %d clusters are Ready", len(contexts))
		for _, context := range contexts {
			clusterIsReadyOrFail(f, &context)
		}
		framework.Logf("%d clusters are Ready", len(contexts))

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

	Describe("DNS", func() {
		BeforeEach(func() {
			framework.SkipUnlessFederated(f.Client)
			createService(f.FederationClientset_1_3, clusterClientSets, f.Namespace.Name)
		})

		It("should be able to discover a federated service", func() {
			framework.SkipUnlessFederated(f.Client)

			svcDNSNames := []string{
				FederatedServiceName,
				fmt.Sprintf("%s.%s", FederatedServiceName, f.Namespace.Name),
				fmt.Sprintf("%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name),
				fmt.Sprintf("%s.%s.%s", FederatedServiceName, f.Namespace.Name, federationName),
				fmt.Sprintf("%s.%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name, federationName),
			}
			// TODO(mml): This could be much faster.  We can launch all the test
			// pods, perhaps in the BeforeEach, and then just poll until we get
			// successes/failures from them all.
			for _, name := range svcDNSNames {
				discoverService(f, name)
			}
		})

		Context("non-local federated service", func() {
			BeforeEach(func() {
				framework.SkipUnlessFederated(f.Client)

				// Delete a federated service shard in the default e2e Kubernetes cluster.
				err := f.Clientset_1_3.Core().Services(f.Namespace.Name).Delete(FederatedServiceName, &api.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred())
				waitForFederatedServiceShard(f.Clientset_1_3, f.Namespace.Name, nil, 0)
			})

			It("should be able to discover a non-local federated service", func() {
				framework.SkipUnlessFederated(f.Client)

				svcDNSNames := []string{
					fmt.Sprintf("%s.%s.%s", FederatedServiceName, f.Namespace.Name, federationName),
					fmt.Sprintf("%s.%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name, federationName),
				}
				for _, name := range svcDNSNames {
					discoverService(f, name)
				}

				// TODO(mml): Unclear how to make this meaningful and not terribly
				// slow.  How long (how many minutes?) do we verify that a given DNS
				// lookup *doesn't* work before we call it a success?  For now,
				// commenting out.
				/*
					localSvcDNSNames := []string{
						FederatedServiceName,
						fmt.Sprintf("%s.%s", FederatedServiceName, f.Namespace.Name),
						fmt.Sprintf("%s.%s.svc.cluster.local.", FederatedServiceName, f.Namespace.Name),
					}
					for _, name := range localSvcDNSNames {
						discoverService(f, name, false)
					}
				*/
			})
		})
	})
})

// waitForFederatedServiceShard waits until the number of shards of a given federated
// service reaches the expected value, i.e. numSvcs in the given individual Kubernetes
// cluster. If the shard count, i.e. numSvcs is expected to be at least one, then
// it also checks if the first shard's name and spec matches that of the given service.
func waitForFederatedServiceShard(cs *release_1_3.Clientset, namespace string, service *v1.Service, numSvcs int) {
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

		// The federation service has no cluster IP.  Clear any cluster IP before
		// comparison.
		clSvc.Spec.ClusterIP = ""

		Expect(clSvc.Name).To(Equal(service.Name))
		// Some fields are expected to be different, so make them the same before checking equality.
		clSvc.Spec.ClusterIP = service.Spec.ClusterIP
		clSvc.Spec.ExternalIPs = service.Spec.ExternalIPs
		clSvc.Spec.DeprecatedPublicIPs = service.Spec.DeprecatedPublicIPs
		clSvc.Spec.LoadBalancerIP = service.Spec.LoadBalancerIP
		clSvc.Spec.LoadBalancerSourceRanges = service.Spec.LoadBalancerSourceRanges
		Expect(clSvc.Spec).To(Equal(service.Spec))
	}
}

func createService(fcs *federation_release_1_3.Clientset, clusterClientSets []*release_1_3.Clientset, namespace string) {
	By(fmt.Sprintf("Creating federated service %q in namespace %q", FederatedServiceName, namespace))

	labels := map[string]string{
		"foo": "bar",
	}

	svc1port := "svc1"
	svc2port := "svc2"

	service := &v1.Service{
		ObjectMeta: v1.ObjectMeta{
			Name: FederatedServiceName,
		},
		Spec: v1.ServiceSpec{
			Selector: labels,
			Ports: []v1.ServicePort{
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
	nservice, err := fcs.Core().Services(namespace).Create(service)
	framework.Logf("Trying to create service %q in namespace %q", service.ObjectMeta.Name, service.ObjectMeta.Namespace)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("creating service %s: %+v", service.Name, err))
	for _, cs := range clusterClientSets {
		waitForFederatedServiceShard(cs, namespace, nservice, 1)
	}
}

func discoverService(f *framework.Framework, name string) {
	command := []string{"sh", "-c", fmt.Sprintf("until nslookup '%s'; do sleep 1; done", name)}

	defer f.Client.Pods(f.Namespace.Name).Delete(FederatedServicePod, api.NewDeleteOptions(0))

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
					Command: command,
				},
			},
			RestartPolicy: api.RestartPolicyOnFailure,
		},
	}

	_, err := f.Client.Pods(f.Namespace.Name).Create(pod)
	Expect(err).
		NotTo(HaveOccurred(), "Trying to create pod to run %q", command)

	// If we ever get any container logs, stash them here.
	logs := ""

	logerr := func(err error) error {
		if err == nil {
			return nil
		}
		if logs == "" {
			return err
		}
		return fmt.Errorf("%s (%v)", logs, err)
	}

	// TODO(mml): Eventually check the IP address is correct, too.
	Eventually(func() error {
		pod, err := f.Client.Pods(f.Namespace.Name).Get(FederatedServicePod)
		if err != nil {
			return logerr(err)
		}
		if len(pod.Status.ContainerStatuses) < 1 {
			return logerr(fmt.Errorf("no container statuses"))
		}

		// Best effort attempt to grab pod logs for debugging
		logs, err = framework.GetPodLogs(f.Client, f.Namespace.Name, FederatedServicePod, "federated-service-discovery-container")
		if err != nil {
			framework.Logf("Cannot fetch pod logs: %v", err)
		}

		status := pod.Status.ContainerStatuses[0]
		if status.State.Terminated == nil {
			return logerr(fmt.Errorf("container is not in terminated state"))
		}
		if status.State.Terminated.ExitCode == 0 {
			return nil
		}

		return logerr(fmt.Errorf("exited %d", status.State.Terminated.ExitCode))
	}, time.Minute*2, time.Second*2).
		Should(BeNil(), "%q should exit 0, but it never did", command)
}
