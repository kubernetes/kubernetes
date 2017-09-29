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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeclientset "k8s.io/client-go/kubernetes"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedframework "k8s.io/kubernetes/federation/test/e2e/framework"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var (
	DefaultFederationName = "e2e-federation"
	// We use this to decide how long to wait for our DNS probes to succeed.
	DNSTTL = 180 * time.Second // TODO: make k8s.io/kubernetes/federation/pkg/federation-controller/service.minDnsTtl exported, and import it here.
)

const (
	// [30000, 32767] is the allowed default service nodeport range and our
	// tests just use the defaults.
	FederatedSvcNodePortFirst = 30000
	FederatedSvcNodePortLast  = 32767
)

var FederationSuite common.Suite

func createClusterObjectOrFail(f *fedframework.Framework, context *fedframework.E2EContext, clusterNamePrefix string) {
	clusterName := clusterNamePrefix + context.Name
	framework.Logf("Creating cluster object: %s (%s, secret: %s)", clusterName, context.Cluster.Cluster.Server, context.Name)
	cluster := federationapi.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: clusterName,
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
	if clusterNamePrefix != "" {
		cluster.Labels = map[string]string{"prefix": clusterNamePrefix}
	}
	_, err := f.FederationClientset.Federation().Clusters().Create(&cluster)
	framework.ExpectNoError(err, fmt.Sprintf("creating cluster: %+v", err))
	framework.Logf("Successfully created cluster object: %s (%s, secret: %s)", clusterName, context.Cluster.Cluster.Server, context.Name)
}

// waitForServiceOrFail waits until a service is either present or absent in the cluster specified by clientset.
// If the condition is not met within timout, it fails the calling test.
func waitForServiceOrFail(clientset *kubeclientset.Clientset, namespace string, service *v1.Service, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated service shard of service %q in namespace %q from cluster", service.Name, namespace))
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterService, err := clientset.CoreV1().Services(namespace).Get(service.Name, metav1.GetOptions{})
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
func waitForServiceShardsOrFail(namespace string, service *v1.Service, clusters fedframework.ClusterSlice) {
	framework.Logf("Waiting for service %q in %d clusters", service.Name, len(clusters))
	for _, c := range clusters {
		waitForServiceOrFail(c.Clientset, namespace, service, true, fedframework.FederatedDefaultTestTimeout)
	}
}

func createService(clientset *fedclientset.Clientset, namespace, name string) (*v1.Service, error) {
	if clientset == nil || len(namespace) == 0 {
		return nil, fmt.Errorf("Internal error: invalid parameters passed to createService: clientset: %v, namespace: %v", clientset, namespace)
	}
	By(fmt.Sprintf("Creating federated service %q in namespace %q", name, namespace))

	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.ServiceSpec{
			Selector: FederatedServiceLabels,
			Type:     v1.ServiceTypeClusterIP,
			Ports: []v1.ServicePort{
				{
					Name:       "http",
					Protocol:   v1.ProtocolTCP,
					Port:       80,
					TargetPort: intstr.FromInt(8080),
				},
			},
			SessionAffinity: v1.ServiceAffinityNone,
		},
	}

	By(fmt.Sprintf("Trying to create service %q in namespace %q", service.Name, namespace))
	return clientset.CoreV1().Services(namespace).Create(service)
}

func createLBService(clientset *fedclientset.Clientset, namespace, name string, clusters fedframework.ClusterSlice) (*v1.Service, error) {
	if clientset == nil || len(namespace) == 0 {
		return nil, fmt.Errorf("Internal error: invalid parameters passed to createService: clientset: %v, namespace: %v", clientset, namespace)
	}
	By(fmt.Sprintf("Creating federated service (type: load balancer) %q in namespace %q", name, namespace))

	// Tests can be run in parallel, so we need a different nodePort for
	// each test.
	// we add in a array all the "available" ports
	availablePorts := make([]int32, FederatedSvcNodePortLast-FederatedSvcNodePortFirst)
	for i := range availablePorts {
		availablePorts[i] = int32(FederatedSvcNodePortFirst + i)
	}

	var err error
	var service *v1.Service
	retry := 10 // the function should retry the service creation on different port only 10 time.

	// until the availablePort list is not empty, lets try to create the service
	for len(availablePorts) > 0 && retry > 0 {
		// select the Id of an available port
		i := rand.Intn(len(availablePorts))

		By(fmt.Sprintf("try creating federated service %q in namespace %q with nodePort %d", name, namespace, availablePorts[i]))

		service, err = createServiceWithNodePort(clientset, namespace, name, availablePorts[i])
		if err == nil {
			// check if service have been created properly in all clusters.
			// if the service is not present in one of the clusters, we should cleanup all services
			if err = checkServicesCreation(namespace, name, clusters); err == nil {
				// everything was created properly so returns the federated service.
				return service, nil
			}
		}

		// in case of error, cleanup everything
		if service != nil {
			if err = deleteService(clientset, namespace, name, nil); err != nil {
				framework.ExpectNoError(err, "Deleting service %q after a partial createService() error", service.Name)
				return nil, err
			}

			cleanupServiceShardsAndProviderResources(namespace, service, clusters)
		}

		// creation failed, lets try with another port
		// first remove from the availablePorts the port with which the creation failed
		availablePorts = append(availablePorts[:i], availablePorts[i+1:]...)
		retry--
	}

	return nil, err
}

func createServiceWithNodePort(clientset *fedclientset.Clientset, namespace, name string, nodePort int32) (*v1.Service, error) {
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.ServiceSpec{
			Selector: FederatedServiceLabels,
			Type:     v1.ServiceTypeLoadBalancer,
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
	return clientset.CoreV1().Services(namespace).Create(service)
}

// checkServicesCreation checks if the service have been created successfuly in all the clusters.
// if the service is not present in at least one of the clusters, this function returns an error.
func checkServicesCreation(namespace, serviceName string, clusters fedframework.ClusterSlice) error {
	framework.Logf("check if service %q have been created in %d clusters", serviceName, len(clusters))
	for _, cluster := range clusters {
		name := cluster.Name
		err := wait.PollImmediate(framework.Poll, fedframework.FederatedDefaultTestTimeout, func() (bool, error) {
			var err error
			_, err = cluster.Clientset.CoreV1().Services(namespace).Get(serviceName, metav1.GetOptions{})
			if err != nil && !errors.IsNotFound(err) {
				// Get failed with an error, try again.
				framework.Logf("Failed to find service %q in namespace %q, in cluster %q: %v. Trying again in %s", serviceName, namespace, name, err, framework.Poll)
				return false, err
			} else if errors.IsNotFound(err) {
				framework.Logf("Service %q in namespace %q in cluster %q not found. Trying again in %s", serviceName, namespace, name, framework.Poll)
				return false, nil
			}
			By(fmt.Sprintf("Service %q in namespace %q in cluster %q found", serviceName, namespace, name))
			return true, nil
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func createServiceOrFail(clientset *fedclientset.Clientset, namespace, name string) *v1.Service {
	service, err := createService(clientset, namespace, name)
	framework.ExpectNoError(err, "Creating service %q in namespace %q", service.Name, namespace)
	By(fmt.Sprintf("Successfully created federated service %q in namespace %q", name, namespace))
	return service
}

func createLBServiceOrFail(clientset *fedclientset.Clientset, namespace, name string, clusters fedframework.ClusterSlice) *v1.Service {
	service, err := createLBService(clientset, namespace, name, clusters)
	framework.ExpectNoError(err, "Creating service %q in namespace %q", service.Name, namespace)
	By(fmt.Sprintf("Successfully created federated service (type: load balancer) %q in namespace %q", name, namespace))
	return service
}

func deleteServiceOrFail(clientset *fedclientset.Clientset, namespace string, serviceName string, orphanDependents *bool) {
	if clientset == nil || len(namespace) == 0 || len(serviceName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to deleteServiceOrFail: clientset: %v, namespace: %v, service: %v", clientset, namespace, serviceName))
	}
	framework.Logf("Deleting service %q in namespace %v", serviceName, namespace)

	err := deleteService(clientset, namespace, serviceName, orphanDependents)
	if err != nil {
		framework.ExpectNoError(err, "Error deleting service %q from namespace %q", serviceName, namespace)
	}
}

func deleteService(clientset *fedclientset.Clientset, namespace string, serviceName string, orphanDependents *bool) error {
	err := clientset.CoreV1().Services(namespace).Delete(serviceName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil {
		return err
	}
	// Wait for the service to be deleted.
	err = wait.Poll(5*time.Second, fedframework.FederatedDefaultTestTimeout, func() (bool, error) {
		_, err := clientset.Core().Services(namespace).Get(serviceName, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	return err
}

func cleanupServiceShardsAndProviderResources(namespace string, service *v1.Service, clusters fedframework.ClusterSlice) {
	framework.Logf("Deleting service %q in %d clusters", service.Name, len(clusters))
	for _, c := range clusters {
		name := c.Name
		var cSvc *v1.Service

		err := wait.PollImmediate(framework.Poll, fedframework.FederatedDefaultTestTimeout, func() (bool, error) {
			var err error
			cSvc, err = c.Clientset.CoreV1().Services(namespace).Get(service.Name, metav1.GetOptions{})
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
			By(fmt.Sprintf("Failed to find service %q in namespace %q, in cluster %q in %s", service.Name, namespace, name, fedframework.FederatedDefaultTestTimeout))
			continue
		}

		if cSvc.Spec.Type == v1.ServiceTypeLoadBalancer {
			// In federation tests, e2e zone names are used to derive federation member cluster names
			zone := fedframework.GetZoneFromClusterName(name)
			serviceLBName := cloudprovider.GetLoadBalancerName(cSvc)
			framework.Logf("cleaning cloud provider resource for service %q in namespace %q, in cluster %q", service.Name, namespace, name)
			framework.CleanupServiceResources(c.Clientset, serviceLBName, zone)
		}

		err = cleanupServiceShard(c.Clientset, name, namespace, cSvc, fedframework.FederatedDefaultTestTimeout)
		if err != nil {
			framework.Logf("Failed to delete service %q in namespace %q, in cluster %q: %v", service.Name, namespace, name, err)
		}
	}
}

func cleanupServiceShard(clientset *kubeclientset.Clientset, clusterName, namespace string, service *v1.Service, timeout time.Duration) error {
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		err := clientset.CoreV1().Services(namespace).Delete(service.Name, &metav1.DeleteOptions{})
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
					Image:   imageutils.GetBusyBoxImage(),
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

// BackendPodMap maps a cluster name to a backend pod created in that cluster
type BackendPodMap map[string]*v1.Pod

// createBackendPodsOrFail creates one pod in each cluster, and returns the created pods.  If creation of any pod fails,
// the test fails (possibly with a partially created set of pods). No retries are attempted.
func createBackendPodsOrFail(clusters fedframework.ClusterSlice, namespace string, name string) BackendPodMap {
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
					Image: "gcr.io/google_containers/echoserver:1.6",
				},
			},
			RestartPolicy: v1.RestartPolicyAlways,
		},
	}
	podMap := make(BackendPodMap)
	for _, c := range clusters {
		name := c.Name
		By(fmt.Sprintf("Creating pod %q in namespace %q in cluster %q", pod.Name, namespace, name))
		createdPod, err := c.Clientset.CoreV1().Pods(namespace).Create(pod)
		framework.ExpectNoError(err, "Creating pod %q in namespace %q in cluster %q", name, namespace, name)
		By(fmt.Sprintf("Successfully created pod %q in namespace %q in cluster %q: %v", pod.Name, namespace, name, *createdPod))
		podMap[name] = createdPod
	}
	return podMap
}

// deleteOneBackendPodOrFail deletes exactly one backend pod which must not be nil
// The test fails if there are any errors.
func deleteOneBackendPodOrFail(c *fedframework.Cluster, pod *v1.Pod) {
	Expect(pod).ToNot(BeNil())
	err := c.Clientset.CoreV1().Pods(pod.Namespace).Delete(pod.Name, metav1.NewDeleteOptions(0))
	msgFmt := fmt.Sprintf("Deleting Pod %q in namespace %q in cluster %q %%v", pod.Name, pod.Namespace, c.Name)
	if errors.IsNotFound(err) {
		framework.Logf(msgFmt, "does not exist. No need to delete it.")
		return
	}
	framework.ExpectNoError(err, msgFmt, "")
	framework.Logf(msgFmt, "was deleted")
}

// deleteBackendPodsOrFail deletes one pod from each cluster that has one.
// If deletion of any pod fails, the test fails (possibly with a partially deleted set of pods). No retries are attempted.
func deleteBackendPodsOrFail(clusters fedframework.ClusterSlice, backendPods BackendPodMap) {
	if backendPods == nil {
		return
	}
	for _, c := range clusters {
		if pod, ok := backendPods[c.Name]; ok {
			deleteOneBackendPodOrFail(c, pod)
		} else {
			By(fmt.Sprintf("No backend pod to delete for cluster %q", c.Name))
		}
	}
}

// waitForReplicatSetToBeDeletedOrFail waits for the named ReplicaSet in namespace to be deleted.
// If the deletion fails, the enclosing test fails.
func waitForReplicaSetToBeDeletedOrFail(clientset *fedclientset.Clientset, namespace string, replicaSet string) {
	err := wait.Poll(5*time.Second, fedframework.FederatedDefaultTestTimeout, func() (bool, error) {
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
