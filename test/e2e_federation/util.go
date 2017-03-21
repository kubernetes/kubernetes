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

package e2e_federation

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	KubeAPIQPS            float32 = 20.0
	KubeAPIBurst                  = 30
	DefaultFederationName         = "e2e-federation"
	UserAgentName                 = "federation-e2e"
	// We use this to decide how long to wait for our DNS probes to succeed.
	DNSTTL = 180 * time.Second // TODO: make k8s.io/kubernetes/federation/pkg/federation-controller/service.minDnsTtl exported, and import it here.
)

const (
	federatedDefaultTestTimeout  = 5 * time.Minute
	federatedClustersWaitTimeout = 1 * time.Minute

	// [30000, 32767] is the allowed default service nodeport range and our
	// tests just use the defaults.
	FederatedSvcNodePortFirst = 30000
	FederatedSvcNodePortLast  = 32767
)

var FederationSuite common.Suite

// cluster keeps track of the assorted objects and state related to each cluster
// in the federation
type cluster struct {
	name string
	*kubeclientset.Clientset
	namespaceCreated bool    // Did we need to create a new namespace in this cluster?  If so, we should delete it.
	backendPod       *v1.Pod // The backend pod, if one's been created.
}

func createClusterObjectOrFail(f *fedframework.Framework, context *fedframework.E2EContext) {
	framework.Logf("Creating cluster object: %s (%s, secret: %s)", context.Name, context.Cluster.Cluster.Server, context.Name)
	cluster := federationapi.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: context.Name,
		},
		Spec: federationapi.ClusterSpec{
			ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
				{
					ClientCIDR:    "0.0.0.0/0",
					ServerAddress: context.Cluster.Cluster.Server,
				},
			},
			SecretRef: &v1.LocalObjectReference{
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

func clusterIsReadyOrFail(f *fedframework.Framework, context *fedframework.E2EContext) {
	c, err := f.FederationClientset.Federation().Clusters().Get(context.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("get cluster: %+v", err))
	if c.ObjectMeta.Name != context.Name {
		framework.Failf("cluster name does not match input context: actual=%+v, expected=%+v", c, context)
	}
	err = isReady(context.Name, f.FederationClientset)
	framework.ExpectNoError(err, fmt.Sprintf("unexpected error in verifying if cluster %s is ready: %+v", context.Name, err))
	framework.Logf("Cluster %s is Ready", context.Name)
}

// waitForAllRegisteredClusters waits for all clusters defined in e2e context to be created
// return ClusterList until the listed cluster items equals clusterCount
func waitForAllRegisteredClusters(f *fedframework.Framework, clusterCount int) *federationapi.ClusterList {
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

// Creates the federation namespace in all underlying clusters.
func createNamespaceInClusters(clusters map[string]*cluster, f *fedframework.Framework) {
	nsName := f.FederationNamespace.Name
	for name, c := range clusters {
		// The e2e Framework created the required namespace in federation control plane, but we need to create it in all the others, if it doesn't yet exist.
		// TODO(nikhiljindal): remove this once we have the namespace controller working as expected.
		if _, err := c.Clientset.Core().Namespaces().Get(nsName, metav1.GetOptions{}); errors.IsNotFound(err) {
			ns := &v1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: nsName,
				},
			}
			_, err := c.Clientset.Core().Namespaces().Create(ns)
			if err == nil {
				c.namespaceCreated = true
			}
			framework.ExpectNoError(err, "Couldn't create the namespace %s in cluster %q", nsName, name)
			framework.Logf("Namespace %s created in cluster %q", nsName, name)
		} else if err != nil {
			framework.Logf("Couldn't create the namespace %s in cluster %q: %v", nsName, name, err)
		}
	}
}

// Unregisters the given clusters from federation control plane.
// Also deletes the federation namespace from each cluster.
func unregisterClusters(clusters map[string]*cluster, f *fedframework.Framework) {
	nsName := f.FederationNamespace.Name
	for name, c := range clusters {
		if c.namespaceCreated {
			if _, err := c.Clientset.Core().Namespaces().Get(nsName, metav1.GetOptions{}); !errors.IsNotFound(err) {
				err := c.Clientset.Core().Namespaces().Delete(nsName, &metav1.DeleteOptions{})
				framework.ExpectNoError(err, "Couldn't delete the namespace %s in cluster %q: %v", nsName, name, err)
			}
			framework.Logf("Namespace %s deleted in cluster %q", nsName, name)
		}
	}

	// Delete the registered clusters in the federation API server.
	clusterList, err := f.FederationClientset.Federation().Clusters().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "Error listing clusters")
	for _, cluster := range clusterList.Items {
		err := f.FederationClientset.Federation().Clusters().Delete(cluster.Name, &metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Error deleting cluster %q", cluster.Name)
	}
}

// waitForNamespaceInFederatedClusters waits for the federated namespace to be created in federated clusters
func waitForNamespaceInFederatedClusters(clusters map[string]*cluster, nsName string, timeout time.Duration) {
	for name, c := range clusters {
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

// can not be moved to util, as By and Expect must be put in Ginkgo test unit
func getRegisteredClusters(userAgentName string, f *fedframework.Framework) (map[string]*cluster, string) {
	clusters := make(map[string]*cluster)
	contexts := f.GetUnderlyingFederatedContexts()

	By("Obtaining a list of all the clusters")
	clusterList := waitForAllRegisteredClusters(f, len(contexts))

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
	waitForNamespaceInFederatedClusters(clusters, f.FederationNamespace.Name, federatedDefaultTestTimeout)
	return clusters, primaryClusterName
}

// waitForServiceOrFail waits until a service is either present or absent in the cluster specified by clientset.
// If the condition is not met within timout, it fails the calling test.
func waitForServiceOrFail(clientset *kubeclientset.Clientset, namespace string, service *v1.Service, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated service shard of service %q in namespace %q from cluster", service.Name, namespace))
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterService, err := clientset.Services(namespace).Get(service.Name, metav1.GetOptions{})
		if (!present) && errors.IsNotFound(err) { // We want it gone, and it's gone.
			By(fmt.Sprintf("Success: shard of federated service %q in namespace %q in cluster is absent", service.Name, namespace))
			return true, nil // Success
		}
		if present && err == nil { // We want it present, and the Get succeeded, so we're all good.
			if equivalent(*clusterService, *service) {
				By(fmt.Sprintf("Success: shard of federated service %q in namespace %q in cluster is present", service.Name, namespace))
				return true, nil // Success
			}
			return false, nil
		}
		By(fmt.Sprintf("Service %q in namespace %q in cluster.  Found: %v, waiting for Found: %v, trying again in %s (err=%v)", service.Name, namespace, clusterService != nil && err == nil, present, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify service %q in namespace %q in cluster: Present=%v", service.Name, namespace, present)
}

// waitForServiceShardsOrFail waits for the service to appear in all clusters
func waitForServiceShardsOrFail(namespace string, service *v1.Service, clusters map[string]*cluster) {
	framework.Logf("Waiting for service %q in %d clusters", service.Name, len(clusters))
	for _, c := range clusters {
		waitForServiceOrFail(c.Clientset, namespace, service, true, federatedDefaultTestTimeout)
	}
}

func createService(clientset *fedclientset.Clientset, namespace, name string) (*v1.Service, error) {
	if clientset == nil || len(namespace) == 0 {
		return nil, fmt.Errorf("Internal error: invalid parameters passed to createService: clientset: %v, namespace: %v", clientset, namespace)
	}
	By(fmt.Sprintf("Creating federated service %q in namespace %q", name, namespace))

	// Tests can be run in parallel, so we need a different nodePort for
	// each test.
	// We add 1 to FederatedSvcNodePortLast because IntnRange's range end
	// is not inclusive.
	nodePort := int32(rand.IntnRange(FederatedSvcNodePortFirst, FederatedSvcNodePortLast+1))

	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.ServiceSpec{
			Selector: FederatedServiceLabels,
			Type:     "LoadBalancer",
			Ports: []v1.ServicePort{
				{
					Name:       "http",
					Protocol:   v1.ProtocolTCP,
					Port:       80,
					TargetPort: intstr.FromInt(8080),
					NodePort:   nodePort,
				},
			},
			SessionAffinity: v1.ServiceAffinityNone,
		},
	}
	By(fmt.Sprintf("Trying to create service %q in namespace %q", service.Name, namespace))
	return clientset.Services(namespace).Create(service)
}

func createServiceOrFail(clientset *fedclientset.Clientset, namespace, name string) *v1.Service {
	service, err := createService(clientset, namespace, name)
	framework.ExpectNoError(err, "Creating service %q in namespace %q", service.Name, namespace)
	By(fmt.Sprintf("Successfully created federated service %q in namespace %q", name, namespace))
	return service
}

func deleteServiceOrFail(clientset *fedclientset.Clientset, namespace string, serviceName string, orphanDependents *bool) {
	if clientset == nil || len(namespace) == 0 || len(serviceName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to deleteServiceOrFail: clientset: %v, namespace: %v, service: %v", clientset, namespace, serviceName))
	}
	framework.Logf("Deleting service %q in namespace %v", serviceName, namespace)
	err := clientset.Services(namespace).Delete(serviceName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	framework.ExpectNoError(err, "Error deleting service %q from namespace %q", serviceName, namespace)
	// Wait for the service to be deleted.
	err = wait.Poll(5*time.Second, federatedDefaultTestTimeout, func() (bool, error) {
		_, err := clientset.Core().Services(namespace).Get(serviceName, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		framework.DescribeSvc(namespace)
		framework.Failf("Error in deleting service %s: %v", serviceName, err)
	}
}

func cleanupServiceShardsAndProviderResources(namespace string, service *v1.Service, clusters map[string]*cluster) {
	framework.Logf("Deleting service %q in %d clusters", service.Name, len(clusters))
	for name, c := range clusters {
		var cSvc *v1.Service

		err := wait.PollImmediate(framework.Poll, federatedDefaultTestTimeout, func() (bool, error) {
			var err error
			cSvc, err = c.Clientset.Services(namespace).Get(service.Name, metav1.GetOptions{})
			if err != nil && !errors.IsNotFound(err) {
				// Get failed with an error, try again.
				framework.Logf("Failed to find service %q in namespace %q, in cluster %q: %v. Trying again in %s", service.Name, namespace, name, err, framework.Poll)
				return false, nil
			} else if errors.IsNotFound(err) {
				cSvc = nil
				By(fmt.Sprintf("Service %q in namespace %q in cluster %q not found", service.Name, namespace, name))
				return true, err
			}
			By(fmt.Sprintf("Service %q in namespace %q in cluster %q found", service.Name, namespace, name))
			return true, err
		})

		if err != nil || cSvc == nil {
			By(fmt.Sprintf("Failed to find service %q in namespace %q, in cluster %q in %s", service.Name, namespace, name, federatedDefaultTestTimeout))
			continue
		}

		err = cleanupServiceShard(c.Clientset, name, namespace, cSvc, federatedDefaultTestTimeout)
		if err != nil {
			framework.Logf("Failed to delete service %q in namespace %q, in cluster %q: %v", service.Name, namespace, name, err)
		}
		err = cleanupServiceShardLoadBalancer(name, cSvc, federatedDefaultTestTimeout)
		if err != nil {
			framework.Logf("Failed to delete cloud provider resources for service %q in namespace %q, in cluster %q", service.Name, namespace, name)
		}
	}
}

func cleanupServiceShard(clientset *kubeclientset.Clientset, clusterName, namespace string, service *v1.Service, timeout time.Duration) error {
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		err := clientset.Services(namespace).Delete(service.Name, &metav1.DeleteOptions{})
		if err != nil && !errors.IsNotFound(err) {
			// Deletion failed with an error, try again.
			framework.Logf("Failed to delete service %q in namespace %q, in cluster %q", service.Name, namespace, clusterName)
			return false, nil
		}
		By(fmt.Sprintf("Service %q in namespace %q in cluster %q deleted", service.Name, namespace, clusterName))
		return true, nil
	})
	return err
}

func cleanupServiceShardLoadBalancer(clusterName string, service *v1.Service, timeout time.Duration) error {
	provider := framework.TestContext.CloudConfig.Provider
	if provider == nil {
		return fmt.Errorf("cloud provider undefined")
	}

	internalSvc := &v1.Service{}
	err := api.Scheme.Convert(service, internalSvc, nil)
	if err != nil {
		return fmt.Errorf("failed to convert versioned service object to internal type: %v", err)
	}

	err = wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		lbi, supported := provider.LoadBalancer()
		if !supported {
			return false, fmt.Errorf("%q doesn't support load balancers", provider.ProviderName())
		}
		err := lbi.EnsureLoadBalancerDeleted(clusterName, internalSvc)
		if err != nil {
			// Deletion failed with an error, try again.
			framework.Logf("Failed to delete cloud provider resources for service %q in namespace %q, in cluster %q: %v", service.Name, service.Namespace, clusterName, err)
			return false, nil
		}
		By(fmt.Sprintf("Cloud provider resources for Service %q in namespace %q in cluster %q deleted", service.Name, service.Namespace, clusterName))
		return true, nil
	})
	return err
}

func podExitCodeDetector(f *fedframework.Framework, name, namespace string, code int32) func() error {
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

	return func() error {
		pod, err := f.ClientSet.Core().Pods(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return logerr(err)
		}
		if len(pod.Status.ContainerStatuses) < 1 {
			return logerr(fmt.Errorf("no container statuses"))
		}

		// Best effort attempt to grab pod logs for debugging
		logs, err = framework.GetPodLogs(f.ClientSet, namespace, name, pod.Spec.Containers[0].Name)
		if err != nil {
			framework.Logf("Cannot fetch pod logs: %v", err)
		}

		status := pod.Status.ContainerStatuses[0]
		if status.State.Terminated == nil {
			return logerr(fmt.Errorf("container is not in terminated state"))
		}
		if status.State.Terminated.ExitCode == code {
			return nil
		}

		return logerr(fmt.Errorf("exited %d", status.State.Terminated.ExitCode))
	}
}

func discoverService(f *fedframework.Framework, name string, exists bool, podName string) {
	command := []string{"sh", "-c", fmt.Sprintf("until nslookup '%s'; do sleep 10; done", name)}
	By(fmt.Sprintf("Looking up %q", name))

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "federated-service-discovery-container",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: command,
				},
			},
			RestartPolicy: v1.RestartPolicyOnFailure,
		},
	}

	nsName := f.FederationNamespace.Name
	By(fmt.Sprintf("Creating pod %q in namespace %q", pod.Name, nsName))
	_, err := f.ClientSet.Core().Pods(nsName).Create(pod)
	framework.ExpectNoError(err, "Trying to create pod to run %q", command)
	By(fmt.Sprintf("Successfully created pod %q in namespace %q", pod.Name, nsName))
	defer func() {
		By(fmt.Sprintf("Deleting pod %q from namespace %q", podName, nsName))
		err := f.ClientSet.Core().Pods(nsName).Delete(podName, metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "Deleting pod %q from namespace %q", podName, nsName)
		By(fmt.Sprintf("Deleted pod %q from namespace %q", podName, nsName))
	}()

	if exists {
		// TODO(mml): Eventually check the IP address is correct, too.
		Eventually(podExitCodeDetector(f, podName, nsName, 0), 3*DNSTTL, time.Second*2).
			Should(BeNil(), "%q should exit 0, but it never did", command)
	} else {
		Eventually(podExitCodeDetector(f, podName, nsName, 0), 3*DNSTTL, time.Second*2).
			ShouldNot(BeNil(), "%q should eventually not exit 0, but it always did", command)
	}
}

// createBackendPodsOrFail creates one pod in each cluster, and returns the created pods (in the same order as clusterClientSets).
// If creation of any pod fails, the test fails (possibly with a partially created set of pods). No retries are attempted.
func createBackendPodsOrFail(clusters map[string]*cluster, namespace string, name string) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			// Namespace: namespace,
			Labels: FederatedServiceLabels,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  name,
					Image: "gcr.io/google_containers/echoserver:1.4",
				},
			},
			RestartPolicy: v1.RestartPolicyAlways,
		},
	}
	for name, c := range clusters {
		By(fmt.Sprintf("Creating pod %q in namespace %q in cluster %q", pod.Name, namespace, name))
		createdPod, err := c.Clientset.Core().Pods(namespace).Create(pod)
		framework.ExpectNoError(err, "Creating pod %q in namespace %q in cluster %q", name, namespace, name)
		By(fmt.Sprintf("Successfully created pod %q in namespace %q in cluster %q: %v", pod.Name, namespace, name, *createdPod))
		c.backendPod = createdPod
	}
}

// deleteOneBackendPodOrFail deletes exactly one backend pod which must not be nil
// The test fails if there are any errors.
func deleteOneBackendPodOrFail(c *cluster) {
	pod := c.backendPod
	Expect(pod).ToNot(BeNil())
	err := c.Clientset.Core().Pods(pod.Namespace).Delete(pod.Name, metav1.NewDeleteOptions(0))
	msgFmt := fmt.Sprintf("Deleting Pod %q in namespace %q in cluster %q %%v", pod.Name, pod.Namespace, c.name)
	if errors.IsNotFound(err) {
		framework.Logf(msgFmt, "does not exist. No need to delete it.")
		return
	}
	framework.ExpectNoError(err, msgFmt, "")
	framework.Logf(msgFmt, "was deleted")
}

// deleteBackendPodsOrFail deletes one pod from each cluster that has one.
// If deletion of any pod fails, the test fails (possibly with a partially deleted set of pods). No retries are attempted.
func deleteBackendPodsOrFail(clusters map[string]*cluster, namespace string) {
	for name, c := range clusters {
		if c.backendPod != nil {
			deleteOneBackendPodOrFail(c)
			c.backendPod = nil
		} else {
			By(fmt.Sprintf("No backend pod to delete for cluster %q", name))
		}
	}
}

// waitForReplicatSetToBeDeletedOrFail waits for the named ReplicaSet in namespace to be deleted.
// If the deletion fails, the enclosing test fails.
func waitForReplicaSetToBeDeletedOrFail(clientset *fedclientset.Clientset, namespace string, replicaSet string) {
	err := wait.Poll(5*time.Second, federatedDefaultTestTimeout, func() (bool, error) {
		_, err := clientset.Extensions().ReplicaSets(namespace).Get(replicaSet, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})

	if err != nil {
		framework.Failf("Error in deleting replica set %s: %v", replicaSet, err)
	}
}
