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

package service

import (
	"bytes"
	"fmt"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// NodePortRange should match whatever the default/configured range is
var NodePortRange = utilnet.PortRange{Base: 30000, Size: 2768}

// PauseDeploymentLabels are unique deployment selector labels for pause pod
var PauseDeploymentLabels = map[string]string{"deployment": "agnhost-pause"}

// TestJig is a test jig to help service testing.
type TestJig struct {
	ID     string
	Name   string
	Client clientset.Interface
	Labels map[string]string
}

// NewTestJig allocates and inits a new TestJig.
func NewTestJig(client clientset.Interface, name string) *TestJig {
	j := &TestJig{}
	j.Client = client
	j.Name = name
	j.ID = j.Name + "-" + string(uuid.NewUUID())
	j.Labels = map[string]string{"testid": j.ID}

	return j
}

// newServiceTemplate returns the default v1.Service template for this j, but
// does not actually create the Service.  The default Service has the same name
// as the j and exposes the given port.
func (j *TestJig) newServiceTemplate(namespace string, proto v1.Protocol, port int32) *v1.Service {
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      j.Name,
			Labels:    j.Labels,
		},
		Spec: v1.ServiceSpec{
			Selector: j.Labels,
			Ports: []v1.ServicePort{
				{
					Protocol: proto,
					Port:     port,
				},
			},
		},
	}
	return service
}

// CreateTCPServiceWithPort creates a new TCP Service with given port based on the
// j's defaults. Callers can provide a function to tweak the Service object before
// it is created.
func (j *TestJig) CreateTCPServiceWithPort(namespace string, tweak func(svc *v1.Service), port int32) *v1.Service {
	svc := j.newServiceTemplate(namespace, v1.ProtocolTCP, port)
	if tweak != nil {
		tweak(svc)
	}
	result, err := j.Client.CoreV1().Services(namespace).Create(svc)
	if err != nil {
		e2elog.Failf("Failed to create TCP Service %q: %v", svc.Name, err)
	}
	return result
}

// CreateTCPServiceOrFail creates a new TCP Service based on the j's
// defaults.  Callers can provide a function to tweak the Service object before
// it is created.
func (j *TestJig) CreateTCPServiceOrFail(namespace string, tweak func(svc *v1.Service)) *v1.Service {
	svc := j.newServiceTemplate(namespace, v1.ProtocolTCP, 80)
	if tweak != nil {
		tweak(svc)
	}
	result, err := j.Client.CoreV1().Services(namespace).Create(svc)
	if err != nil {
		e2elog.Failf("Failed to create TCP Service %q: %v", svc.Name, err)
	}
	return result
}

// CreateUDPServiceOrFail creates a new UDP Service based on the j's
// defaults.  Callers can provide a function to tweak the Service object before
// it is created.
func (j *TestJig) CreateUDPServiceOrFail(namespace string, tweak func(svc *v1.Service)) *v1.Service {
	svc := j.newServiceTemplate(namespace, v1.ProtocolUDP, 80)
	if tweak != nil {
		tweak(svc)
	}
	result, err := j.Client.CoreV1().Services(namespace).Create(svc)
	if err != nil {
		e2elog.Failf("Failed to create UDP Service %q: %v", svc.Name, err)
	}
	return result
}

// CreateExternalNameServiceOrFail creates a new ExternalName type Service based on the j's defaults.
// Callers can provide a function to tweak the Service object before it is created.
func (j *TestJig) CreateExternalNameServiceOrFail(namespace string, tweak func(svc *v1.Service)) *v1.Service {
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      j.Name,
			Labels:    j.Labels,
		},
		Spec: v1.ServiceSpec{
			Selector:     j.Labels,
			ExternalName: "foo.example.com",
			Type:         v1.ServiceTypeExternalName,
		},
	}
	if tweak != nil {
		tweak(svc)
	}
	result, err := j.Client.CoreV1().Services(namespace).Create(svc)
	if err != nil {
		e2elog.Failf("Failed to create ExternalName Service %q: %v", svc.Name, err)
	}
	return result
}

// CreateServiceWithServicePort creates a new Service with ServicePort.
func (j *TestJig) CreateServiceWithServicePort(labels map[string]string, namespace string, ports []v1.ServicePort) (*v1.Service, error) {
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: j.Name,
		},
		Spec: v1.ServiceSpec{
			Selector: labels,
			Ports:    ports,
		},
	}
	return j.Client.CoreV1().Services(namespace).Create(service)
}

// ChangeServiceType updates the given service's ServiceType to the given newType.
func (j *TestJig) ChangeServiceType(namespace, name string, newType v1.ServiceType, timeout time.Duration) {
	ingressIP := ""
	svc := j.UpdateServiceOrFail(namespace, name, func(s *v1.Service) {
		for _, ing := range s.Status.LoadBalancer.Ingress {
			if ing.IP != "" {
				ingressIP = ing.IP
			}
		}
		s.Spec.Type = newType
		s.Spec.Ports[0].NodePort = 0
	})
	if ingressIP != "" {
		j.WaitForLoadBalancerDestroyOrFail(namespace, svc.Name, ingressIP, int(svc.Spec.Ports[0].Port), timeout)
	}
}

// CreateOnlyLocalNodePortService creates a NodePort service with
// ExternalTrafficPolicy set to Local and sanity checks its nodePort.
// If createPod is true, it also creates an RC with 1 replica of
// the standard netexec container used everywhere in this test.
func (j *TestJig) CreateOnlyLocalNodePortService(namespace, serviceName string, createPod bool) *v1.Service {
	ginkgo.By("creating a service " + namespace + "/" + serviceName + " with type=NodePort and ExternalTrafficPolicy=Local")
	svc := j.CreateTCPServiceOrFail(namespace, func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeNodePort
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		svc.Spec.Ports = []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: 80}}
	})

	if createPod {
		ginkgo.By("creating a pod to be part of the service " + serviceName)
		j.RunOrFail(namespace, nil)
	}
	j.SanityCheckService(svc, v1.ServiceTypeNodePort)
	return svc
}

// CreateOnlyLocalLoadBalancerService creates a loadbalancer service with
// ExternalTrafficPolicy set to Local and waits for it to acquire an ingress IP.
// If createPod is true, it also creates an RC with 1 replica of
// the standard netexec container used everywhere in this test.
func (j *TestJig) CreateOnlyLocalLoadBalancerService(namespace, serviceName string, timeout time.Duration, createPod bool,
	tweak func(svc *v1.Service)) *v1.Service {
	ginkgo.By("creating a service " + namespace + "/" + serviceName + " with type=LoadBalancer and ExternalTrafficPolicy=Local")
	svc := j.CreateTCPServiceOrFail(namespace, func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		// We need to turn affinity off for our LB distribution tests
		svc.Spec.SessionAffinity = v1.ServiceAffinityNone
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		if tweak != nil {
			tweak(svc)
		}
	})

	if createPod {
		ginkgo.By("creating a pod to be part of the service " + serviceName)
		j.RunOrFail(namespace, nil)
	}
	ginkgo.By("waiting for loadbalancer for service " + namespace + "/" + serviceName)
	svc = j.WaitForLoadBalancerOrFail(namespace, serviceName, timeout)
	j.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)
	return svc
}

// CreateLoadBalancerService creates a loadbalancer service and waits
// for it to acquire an ingress IP.
func (j *TestJig) CreateLoadBalancerService(namespace, serviceName string, timeout time.Duration, tweak func(svc *v1.Service)) *v1.Service {
	ginkgo.By("creating a service " + namespace + "/" + serviceName + " with type=LoadBalancer")
	svc := j.CreateTCPServiceOrFail(namespace, func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeLoadBalancer
		// We need to turn affinity off for our LB distribution tests
		svc.Spec.SessionAffinity = v1.ServiceAffinityNone
		if tweak != nil {
			tweak(svc)
		}
	})

	ginkgo.By("waiting for loadbalancer for service " + namespace + "/" + serviceName)
	svc = j.WaitForLoadBalancerOrFail(namespace, serviceName, timeout)
	j.SanityCheckService(svc, v1.ServiceTypeLoadBalancer)
	return svc
}

// GetEndpointNodes returns a map of nodenames:external-ip on which the
// endpoints of the given Service are running.
func (j *TestJig) GetEndpointNodes(svc *v1.Service) map[string][]string {
	nodes := j.GetNodes(MaxNodesForEndpointsTests)
	endpoints, err := j.Client.CoreV1().Endpoints(svc.Namespace).Get(svc.Name, metav1.GetOptions{})
	if err != nil {
		e2elog.Failf("Get endpoints for service %s/%s failed (%s)", svc.Namespace, svc.Name, err)
	}
	if len(endpoints.Subsets) == 0 {
		e2elog.Failf("Endpoint has no subsets, cannot determine node addresses.")
	}
	epNodes := sets.NewString()
	for _, ss := range endpoints.Subsets {
		for _, e := range ss.Addresses {
			if e.NodeName != nil {
				epNodes.Insert(*e.NodeName)
			}
		}
	}
	nodeMap := map[string][]string{}
	for _, n := range nodes.Items {
		if epNodes.Has(n.Name) {
			nodeMap[n.Name] = e2enode.GetAddresses(&n, v1.NodeExternalIP)
		}
	}
	return nodeMap
}

// GetNodes returns the first maxNodesForTest nodes. Useful in large clusters
// where we don't eg: want to create an endpoint per node.
func (j *TestJig) GetNodes(maxNodesForTest int) (nodes *v1.NodeList) {
	nodes = framework.GetReadySchedulableNodesOrDie(j.Client)
	if len(nodes.Items) <= maxNodesForTest {
		maxNodesForTest = len(nodes.Items)
	}
	nodes.Items = nodes.Items[:maxNodesForTest]
	return nodes
}

// GetNodesNames returns a list of names of the first maxNodesForTest nodes
func (j *TestJig) GetNodesNames(maxNodesForTest int) []string {
	nodes := j.GetNodes(maxNodesForTest)
	nodesNames := []string{}
	for _, node := range nodes.Items {
		nodesNames = append(nodesNames, node.Name)
	}
	return nodesNames
}

// WaitForEndpointOnNode waits for a service endpoint on the given node.
func (j *TestJig) WaitForEndpointOnNode(namespace, serviceName, nodeName string) {
	err := wait.PollImmediate(framework.Poll, LoadBalancerCreateTimeoutDefault, func() (bool, error) {
		endpoints, err := j.Client.CoreV1().Endpoints(namespace).Get(serviceName, metav1.GetOptions{})
		if err != nil {
			e2elog.Logf("Get endpoints for service %s/%s failed (%s)", namespace, serviceName, err)
			return false, nil
		}
		if len(endpoints.Subsets) == 0 {
			e2elog.Logf("Expect endpoints with subsets, got none.")
			return false, nil
		}
		// TODO: Handle multiple endpoints
		if len(endpoints.Subsets[0].Addresses) == 0 {
			e2elog.Logf("Expected Ready endpoints - found none")
			return false, nil
		}
		epHostName := *endpoints.Subsets[0].Addresses[0].NodeName
		e2elog.Logf("Pod for service %s/%s is on node %s", namespace, serviceName, epHostName)
		if epHostName != nodeName {
			e2elog.Logf("Found endpoint on wrong node, expected %v, got %v", nodeName, epHostName)
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err)
}

// WaitForAvailableEndpoint waits for at least 1 endpoint to be available till timeout
func (j *TestJig) WaitForAvailableEndpoint(namespace, serviceName string, timeout time.Duration) {
	//Wait for endpoints to be created, this may take longer time if service backing pods are taking longer time to run
	endpointSelector := fields.OneTermEqualSelector("metadata.name", serviceName)
	stopCh := make(chan struct{})
	endpointAvailable := false
	var controller cache.Controller
	_, controller = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.FieldSelector = endpointSelector.String()
				obj, err := j.Client.CoreV1().Endpoints(namespace).List(options)
				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.FieldSelector = endpointSelector.String()
				return j.Client.CoreV1().Endpoints(namespace).Watch(options)
			},
		},
		&v1.Endpoints{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				if e, ok := obj.(*v1.Endpoints); ok {
					if len(e.Subsets) > 0 && len(e.Subsets[0].Addresses) > 0 {
						endpointAvailable = true
					}
				}
			},
			UpdateFunc: func(old, cur interface{}) {
				if e, ok := cur.(*v1.Endpoints); ok {
					if len(e.Subsets) > 0 && len(e.Subsets[0].Addresses) > 0 {
						endpointAvailable = true
					}
				}
			},
		},
	)
	defer func() {
		close(stopCh)
	}()

	go controller.Run(stopCh)

	err := wait.Poll(1*time.Second, timeout, func() (bool, error) {
		return endpointAvailable, nil
	})
	framework.ExpectNoError(err, "No subset of available IP address found for the endpoint %s within timeout %v", serviceName, timeout)
}

// SanityCheckService performs sanity checks on the given service
func (j *TestJig) SanityCheckService(svc *v1.Service, svcType v1.ServiceType) {
	if svc.Spec.Type != svcType {
		e2elog.Failf("unexpected Spec.Type (%s) for service, expected %s", svc.Spec.Type, svcType)
	}

	if svcType != v1.ServiceTypeExternalName {
		if svc.Spec.ExternalName != "" {
			e2elog.Failf("unexpected Spec.ExternalName (%s) for service, expected empty", svc.Spec.ExternalName)
		}
		if svc.Spec.ClusterIP != api.ClusterIPNone && svc.Spec.ClusterIP == "" {
			e2elog.Failf("didn't get ClusterIP for non-ExternamName service")
		}
	} else {
		if svc.Spec.ClusterIP != "" {
			e2elog.Failf("unexpected Spec.ClusterIP (%s) for ExternamName service, expected empty", svc.Spec.ClusterIP)
		}
	}

	expectNodePorts := false
	if svcType != v1.ServiceTypeClusterIP && svcType != v1.ServiceTypeExternalName {
		expectNodePorts = true
	}
	for i, port := range svc.Spec.Ports {
		hasNodePort := (port.NodePort != 0)
		if hasNodePort != expectNodePorts {
			e2elog.Failf("unexpected Spec.Ports[%d].NodePort (%d) for service", i, port.NodePort)
		}
		if hasNodePort {
			if !NodePortRange.Contains(int(port.NodePort)) {
				e2elog.Failf("out-of-range nodePort (%d) for service", port.NodePort)
			}
		}
	}
	expectIngress := false
	if svcType == v1.ServiceTypeLoadBalancer {
		expectIngress = true
	}
	hasIngress := len(svc.Status.LoadBalancer.Ingress) != 0
	if hasIngress != expectIngress {
		e2elog.Failf("unexpected number of Status.LoadBalancer.Ingress (%d) for service", len(svc.Status.LoadBalancer.Ingress))
	}
	if hasIngress {
		for i, ing := range svc.Status.LoadBalancer.Ingress {
			if ing.IP == "" && ing.Hostname == "" {
				e2elog.Failf("unexpected Status.LoadBalancer.Ingress[%d] for service: %#v", i, ing)
			}
		}
	}
}

// UpdateService fetches a service, calls the update function on it, and
// then attempts to send the updated service. It tries up to 3 times in the
// face of timeouts and conflicts.
func (j *TestJig) UpdateService(namespace, name string, update func(*v1.Service)) (*v1.Service, error) {
	for i := 0; i < 3; i++ {
		service, err := j.Client.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("failed to get Service %q: %v", name, err)
		}
		update(service)
		service, err = j.Client.CoreV1().Services(namespace).Update(service)
		if err == nil {
			return service, nil
		}
		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			return nil, fmt.Errorf("failed to update Service %q: %v", name, err)
		}
	}
	return nil, fmt.Errorf("too many retries updating Service %q", name)
}

// UpdateServiceOrFail fetches a service, calls the update function on it, and
// then attempts to send the updated service. It tries up to 3 times in the
// face of timeouts and conflicts.
func (j *TestJig) UpdateServiceOrFail(namespace, name string, update func(*v1.Service)) *v1.Service {
	svc, err := j.UpdateService(namespace, name, update)
	if err != nil {
		e2elog.Failf(err.Error())
	}
	return svc
}

// WaitForNewIngressIPOrFail waits for the given service to get a new ingress IP, or fails after the given timeout
func (j *TestJig) WaitForNewIngressIPOrFail(namespace, name, existingIP string, timeout time.Duration) *v1.Service {
	e2elog.Logf("Waiting up to %v for service %q to get a new ingress IP", timeout, name)
	service := j.waitForConditionOrFail(namespace, name, timeout, "have a new ingress IP", func(svc *v1.Service) bool {
		if len(svc.Status.LoadBalancer.Ingress) == 0 {
			return false
		}
		ip := svc.Status.LoadBalancer.Ingress[0].IP
		if ip == "" || ip == existingIP {
			return false
		}
		return true
	})
	return service
}

// ChangeServiceNodePortOrFail changes node ports of the given service.
func (j *TestJig) ChangeServiceNodePortOrFail(namespace, name string, initial int) *v1.Service {
	var err error
	var service *v1.Service
	for i := 1; i < NodePortRange.Size; i++ {
		offs1 := initial - NodePortRange.Base
		offs2 := (offs1 + i) % NodePortRange.Size
		newPort := NodePortRange.Base + offs2
		service, err = j.UpdateService(namespace, name, func(s *v1.Service) {
			s.Spec.Ports[0].NodePort = int32(newPort)
		})
		if err != nil && strings.Contains(err.Error(), portallocator.ErrAllocated.Error()) {
			e2elog.Logf("tried nodePort %d, but it is in use, will try another", newPort)
			continue
		}
		// Otherwise err was nil or err was a real error
		break
	}
	if err != nil {
		e2elog.Failf("Could not change the nodePort: %v", err)
	}
	return service
}

// WaitForLoadBalancerOrFail waits the given service to have a LoadBalancer, or fails after the given timeout
func (j *TestJig) WaitForLoadBalancerOrFail(namespace, name string, timeout time.Duration) *v1.Service {
	e2elog.Logf("Waiting up to %v for service %q to have a LoadBalancer", timeout, name)
	service := j.waitForConditionOrFail(namespace, name, timeout, "have a load balancer", func(svc *v1.Service) bool {
		return len(svc.Status.LoadBalancer.Ingress) > 0
	})
	return service
}

// WaitForLoadBalancerDestroyOrFail waits the given service to destroy a LoadBalancer, or fails after the given timeout
func (j *TestJig) WaitForLoadBalancerDestroyOrFail(namespace, name string, ip string, port int, timeout time.Duration) *v1.Service {
	// TODO: once support ticket 21807001 is resolved, reduce this timeout back to something reasonable
	defer func() {
		if err := framework.EnsureLoadBalancerResourcesDeleted(ip, strconv.Itoa(port)); err != nil {
			e2elog.Logf("Failed to delete cloud resources for service: %s %d (%v)", ip, port, err)
		}
	}()

	e2elog.Logf("Waiting up to %v for service %q to have no LoadBalancer", timeout, name)
	service := j.waitForConditionOrFail(namespace, name, timeout, "have no load balancer", func(svc *v1.Service) bool {
		return len(svc.Status.LoadBalancer.Ingress) == 0
	})
	return service
}

func (j *TestJig) waitForConditionOrFail(namespace, name string, timeout time.Duration, message string, conditionFn func(*v1.Service) bool) *v1.Service {
	var service *v1.Service
	pollFunc := func() (bool, error) {
		svc, err := j.Client.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if conditionFn(svc) {
			service = svc
			return true, nil
		}
		return false, nil
	}
	if err := wait.PollImmediate(framework.Poll, timeout, pollFunc); err != nil {
		e2elog.Failf("Timed out waiting for service %q to %s", name, message)
	}
	return service
}

// newRCTemplate returns the default v1.ReplicationController object for
// this j, but does not actually create the RC.  The default RC has the same
// name as the j and runs the "netexec" container.
func (j *TestJig) newRCTemplate(namespace string) *v1.ReplicationController {
	var replicas int32 = 1
	var grace int64 = 3 // so we don't race with kube-proxy when scaling up/down

	rc := &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      j.Name,
			Labels:    j.Labels,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &replicas,
			Selector: j.Labels,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: j.Labels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "netexec",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Args:  []string{"netexec", "--http-port=80", "--udp-port=80"},
							ReadinessProbe: &v1.Probe{
								PeriodSeconds: 3,
								Handler: v1.Handler{
									HTTPGet: &v1.HTTPGetAction{
										Port: intstr.FromInt(80),
										Path: "/hostName",
									},
								},
							},
						},
					},
					TerminationGracePeriodSeconds: &grace,
				},
			},
		},
	}
	return rc
}

// AddRCAntiAffinity adds AntiAffinity to the given ReplicationController.
func (j *TestJig) AddRCAntiAffinity(rc *v1.ReplicationController) {
	var replicas int32 = 2

	rc.Spec.Replicas = &replicas
	if rc.Spec.Template.Spec.Affinity == nil {
		rc.Spec.Template.Spec.Affinity = &v1.Affinity{}
	}
	if rc.Spec.Template.Spec.Affinity.PodAntiAffinity == nil {
		rc.Spec.Template.Spec.Affinity.PodAntiAffinity = &v1.PodAntiAffinity{}
	}
	rc.Spec.Template.Spec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution = append(
		rc.Spec.Template.Spec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution,
		v1.PodAffinityTerm{
			LabelSelector: &metav1.LabelSelector{MatchLabels: j.Labels},
			Namespaces:    nil,
			TopologyKey:   "kubernetes.io/hostname",
		})
}

// CreatePDBOrFail returns a PodDisruptionBudget for the given ReplicationController, or fails if a PodDisruptionBudget isn't ready
func (j *TestJig) CreatePDBOrFail(namespace string, rc *v1.ReplicationController) *policyv1beta1.PodDisruptionBudget {
	pdb := j.newPDBTemplate(namespace, rc)
	newPdb, err := j.Client.PolicyV1beta1().PodDisruptionBudgets(namespace).Create(pdb)
	if err != nil {
		e2elog.Failf("Failed to create PDB %q %v", pdb.Name, err)
	}
	if err := j.waitForPdbReady(namespace); err != nil {
		e2elog.Failf("Failed waiting for PDB to be ready: %v", err)
	}

	return newPdb
}

// newPDBTemplate returns the default policyv1beta1.PodDisruptionBudget object for
// this j, but does not actually create the PDB.  The default PDB specifies a
// MinAvailable of N-1 and matches the pods created by the RC.
func (j *TestJig) newPDBTemplate(namespace string, rc *v1.ReplicationController) *policyv1beta1.PodDisruptionBudget {
	minAvailable := intstr.FromInt(int(*rc.Spec.Replicas) - 1)

	pdb := &policyv1beta1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      j.Name,
			Labels:    j.Labels,
		},
		Spec: policyv1beta1.PodDisruptionBudgetSpec{
			MinAvailable: &minAvailable,
			Selector:     &metav1.LabelSelector{MatchLabels: j.Labels},
		},
	}

	return pdb
}

// RunOrFail creates a ReplicationController and Pod(s) and waits for the
// Pod(s) to be running. Callers can provide a function to tweak the RC object
// before it is created.
func (j *TestJig) RunOrFail(namespace string, tweak func(rc *v1.ReplicationController)) *v1.ReplicationController {
	rc := j.newRCTemplate(namespace)
	if tweak != nil {
		tweak(rc)
	}
	result, err := j.Client.CoreV1().ReplicationControllers(namespace).Create(rc)
	if err != nil {
		e2elog.Failf("Failed to create RC %q: %v", rc.Name, err)
	}
	pods, err := j.waitForPodsCreated(namespace, int(*(rc.Spec.Replicas)))
	if err != nil {
		e2elog.Failf("Failed to create pods: %v", err)
	}
	if err := j.waitForPodsReady(namespace, pods); err != nil {
		e2elog.Failf("Failed waiting for pods to be running: %v", err)
	}
	return result
}

// Scale scales pods to the given replicas
func (j *TestJig) Scale(namespace string, replicas int) {
	rc := j.Name
	scale, err := j.Client.CoreV1().ReplicationControllers(namespace).GetScale(rc, metav1.GetOptions{})
	if err != nil {
		e2elog.Failf("Failed to get scale for RC %q: %v", rc, err)
	}

	scale.Spec.Replicas = int32(replicas)
	_, err = j.Client.CoreV1().ReplicationControllers(namespace).UpdateScale(rc, scale)
	if err != nil {
		e2elog.Failf("Failed to scale RC %q: %v", rc, err)
	}
	pods, err := j.waitForPodsCreated(namespace, replicas)
	if err != nil {
		e2elog.Failf("Failed waiting for pods: %v", err)
	}
	if err := j.waitForPodsReady(namespace, pods); err != nil {
		e2elog.Failf("Failed waiting for pods to be running: %v", err)
	}
}

func (j *TestJig) waitForPdbReady(namespace string) error {
	timeout := 2 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(2 * time.Second) {
		pdb, err := j.Client.PolicyV1beta1().PodDisruptionBudgets(namespace).Get(j.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if pdb.Status.PodDisruptionsAllowed > 0 {
			return nil
		}
	}

	return fmt.Errorf("timeout waiting for PDB %q to be ready", j.Name)
}

func (j *TestJig) waitForPodsCreated(namespace string, replicas int) ([]string, error) {
	timeout := 2 * time.Minute
	// List the pods, making sure we observe all the replicas.
	label := labels.SelectorFromSet(labels.Set(j.Labels))
	e2elog.Logf("Waiting up to %v for %d pods to be created", timeout, replicas)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(2 * time.Second) {
		options := metav1.ListOptions{LabelSelector: label.String()}
		pods, err := j.Client.CoreV1().Pods(namespace).List(options)
		if err != nil {
			return nil, err
		}

		found := []string{}
		for _, pod := range pods.Items {
			if pod.DeletionTimestamp != nil {
				continue
			}
			found = append(found, pod.Name)
		}
		if len(found) == replicas {
			e2elog.Logf("Found all %d pods", replicas)
			return found, nil
		}
		e2elog.Logf("Found %d/%d pods - will retry", len(found), replicas)
	}
	return nil, fmt.Errorf("timeout waiting for %d pods to be created", replicas)
}

func (j *TestJig) waitForPodsReady(namespace string, pods []string) error {
	timeout := 2 * time.Minute
	if !e2epod.CheckPodsRunningReady(j.Client, namespace, pods, timeout) {
		return fmt.Errorf("timeout waiting for %d pods to be ready", len(pods))
	}
	return nil
}

func testReachabilityOverServiceName(serviceName string, sp v1.ServicePort, execPod *v1.Pod) {
	testEndpointReachability(serviceName, sp.Port, sp.Protocol, execPod)
}

func testReachabilityOverClusterIP(clusterIP string, sp v1.ServicePort, execPod *v1.Pod) {
	// If .spec.clusterIP is set to "" or "None" for service, ClusterIP is not created, so reachability can not be tested over clusterIP:servicePort
	isClusterIPV46, err := regexp.MatchString(framework.RegexIPv4+"||"+framework.RegexIPv6, clusterIP)
	framework.ExpectNoError(err, "Unable to parse ClusterIP: %s", clusterIP)
	if isClusterIPV46 {
		testEndpointReachability(clusterIP, sp.Port, sp.Protocol, execPod)
	}
}

func testReachabilityOverNodePorts(nodes *v1.NodeList, sp v1.ServicePort, pod *v1.Pod) {
	internalAddrs := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)
	externalAddrs := e2enode.CollectAddresses(nodes, v1.NodeExternalIP)
	for _, internalAddr := range internalAddrs {
		// If the node's internal address points to localhost, then we are not
		// able to test the service reachability via that address
		if isInvalidOrLocalhostAddress(internalAddr) {
			e2elog.Logf("skipping testEndpointReachability() for internal adddress %s", internalAddr)
			continue
		}
		testEndpointReachability(internalAddr, sp.NodePort, sp.Protocol, pod)
	}
	for _, externalAddr := range externalAddrs {
		testEndpointReachability(externalAddr, sp.NodePort, sp.Protocol, pod)
	}
}

// isInvalidOrLocalhostAddress returns `true` if the provided `ip` is either not
// parsable or the loopback address. Otherwise it will return `false`.
func isInvalidOrLocalhostAddress(ip string) bool {
	parsedIP := net.ParseIP(ip)
	if parsedIP == nil || parsedIP.IsLoopback() {
		return true
	}
	return false
}

// testEndpointReachability tests reachability to endpoints (i.e. IP, ServiceName) and ports. Test request is initiated from specified execPod.
// TCP and UDP protocol based service are supported at this moment
// TODO: add support to test SCTP Protocol based services.
func testEndpointReachability(endpoint string, port int32, protocol v1.Protocol, execPod *v1.Pod) {
	ep := net.JoinHostPort(endpoint, strconv.Itoa(int(port)))
	cmd := ""
	switch protocol {
	case v1.ProtocolTCP:
		cmd = fmt.Sprintf("nc -zv -t -w 2 %s %v", endpoint, port)
	case v1.ProtocolUDP:
		cmd = fmt.Sprintf("nc -zv -u -w 2 %s %v", endpoint, port)
	default:
		e2elog.Failf("Service reachablity check is not supported for %v", protocol)
	}
	if cmd != "" {
		err := wait.PollImmediate(1*time.Second, ServiceReachabilityShortPollTimeout, func() (bool, error) {
			if _, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd); err != nil {
				e2elog.Logf("Service reachability failing with error: %v\nRetrying...", err)
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(err, "Service is not reachable within %v timeout on endpoint %s over %s protocol", ServiceReachabilityShortPollTimeout, ep, protocol)
	}
}

// checkClusterIPServiceReachability ensures that service of type ClusterIP is reachable over
// - ServiceName:ServicePort, ClusterIP:ServicePort
func (j *TestJig) checkClusterIPServiceReachability(namespace string, svc *v1.Service, pod *v1.Pod) {
	clusterIP := svc.Spec.ClusterIP
	servicePorts := svc.Spec.Ports

	j.WaitForAvailableEndpoint(namespace, svc.Name, ServiceEndpointsTimeout)

	for _, servicePort := range servicePorts {
		testReachabilityOverServiceName(svc.Name, servicePort, pod)
		testReachabilityOverClusterIP(clusterIP, servicePort, pod)
	}
}

// checkNodePortServiceReachability ensures that service of type nodePort are reachable
// - Internal clients should be reachable to service over -
//   	ServiceName:ServicePort, ClusterIP:ServicePort and NodeInternalIPs:NodePort
// - External clients should be reachable to service over -
//   	NodePublicIPs:NodePort
func (j *TestJig) checkNodePortServiceReachability(namespace string, svc *v1.Service, pod *v1.Pod) {
	clusterIP := svc.Spec.ClusterIP
	servicePorts := svc.Spec.Ports

	// Consider only 2 nodes for testing
	nodes := j.GetNodes(2)

	j.WaitForAvailableEndpoint(namespace, svc.Name, ServiceEndpointsTimeout)

	for _, servicePort := range servicePorts {
		testReachabilityOverServiceName(svc.Name, servicePort, pod)
		testReachabilityOverClusterIP(clusterIP, servicePort, pod)
		testReachabilityOverNodePorts(nodes, servicePort, pod)
	}
}

// checkExternalServiceReachability ensures service of type externalName resolves to IP address and no fake externalName is set
// FQDN of kubernetes is used as externalName(for air tight platforms).
func (j *TestJig) checkExternalServiceReachability(svc *v1.Service, pod *v1.Pod) {
	// Service must resolve to IP
	cmd := fmt.Sprintf("nslookup %s", svc.Name)
	_, err := framework.RunHostCmd(pod.Namespace, pod.Name, cmd)
	framework.ExpectNoError(err, "ExternalName service must resolve to IP")
}

// CheckServiceReachability ensures that request are served by the services. Only supports Services with type ClusterIP, NodePort and ExternalName.
func (j *TestJig) CheckServiceReachability(namespace string, svc *v1.Service, pod *v1.Pod) {
	svcType := svc.Spec.Type

	j.SanityCheckService(svc, svcType)

	switch svcType {
	case v1.ServiceTypeClusterIP:
		j.checkClusterIPServiceReachability(namespace, svc, pod)
	case v1.ServiceTypeNodePort:
		j.checkNodePortServiceReachability(namespace, svc, pod)
	case v1.ServiceTypeExternalName:
		j.checkExternalServiceReachability(svc, pod)
	default:
		e2elog.Failf("Unsupported service type \"%s\" to verify service reachability for \"%s\" service. This may due to diverse implementation of the service type.", svcType, svc.Name)
	}
}

// TestReachableHTTP tests that the given host serves HTTP on the given port.
func (j *TestJig) TestReachableHTTP(host string, port int, timeout time.Duration) {
	j.TestReachableHTTPWithRetriableErrorCodes(host, port, []int{}, timeout)
}

// TestReachableHTTPWithRetriableErrorCodes tests that the given host serves HTTP on the given port with the given retriableErrCodes.
func (j *TestJig) TestReachableHTTPWithRetriableErrorCodes(host string, port int, retriableErrCodes []int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := framework.PokeHTTP(host, port, "/echo?msg=hello",
			&framework.HTTPPokeParams{
				BodyContains:   "hello",
				RetriableCodes: retriableErrCodes,
			})
		if result.Status == framework.HTTPSuccess {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		if err == wait.ErrWaitTimeout {
			e2elog.Failf("Could not reach HTTP service through %v:%v after %v", host, port, timeout)
		} else {
			e2elog.Failf("Failed to reach HTTP service through %v:%v: %v", host, port, err)
		}
	}
}

// TestNotReachableHTTP tests that a HTTP request doesn't connect to the given host and port.
func (j *TestJig) TestNotReachableHTTP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := framework.PokeHTTP(host, port, "/", nil)
		if result.Code == 0 {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		e2elog.Failf("HTTP service %v:%v reachable after %v: %v", host, port, timeout, err)
	}
}

// TestRejectedHTTP tests that the given host rejects a HTTP request on the given port.
func (j *TestJig) TestRejectedHTTP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := framework.PokeHTTP(host, port, "/", nil)
		if result.Status == framework.HTTPRefused {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		e2elog.Failf("HTTP service %v:%v not rejected: %v", host, port, err)
	}
}

// TestReachableUDP tests that the given host serves UDP on the given port.
func (j *TestJig) TestReachableUDP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := framework.PokeUDP(host, port, "echo hello", &framework.UDPPokeParams{
			Timeout:  3 * time.Second,
			Response: "hello",
		})
		if result.Status == framework.UDPSuccess {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		e2elog.Failf("Could not reach UDP service through %v:%v after %v: %v", host, port, timeout, err)
	}
}

// TestNotReachableUDP tests that the given host doesn't serve UDP on the given port.
func (j *TestJig) TestNotReachableUDP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := framework.PokeUDP(host, port, "echo hello", &framework.UDPPokeParams{Timeout: 3 * time.Second})
		if result.Status != framework.UDPSuccess && result.Status != framework.UDPError {
			return true, nil
		}
		return false, nil // caller can retry
	}
	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		e2elog.Failf("UDP service %v:%v reachable after %v: %v", host, port, timeout, err)
	}
}

// TestRejectedUDP tests that the given host rejects a UDP request on the given port.
func (j *TestJig) TestRejectedUDP(host string, port int, timeout time.Duration) {
	pollfn := func() (bool, error) {
		result := framework.PokeUDP(host, port, "echo hello", &framework.UDPPokeParams{Timeout: 3 * time.Second})
		if result.Status == framework.UDPRefused {
			return true, nil
		}
		return false, nil // caller can retry
	}
	if err := wait.PollImmediate(framework.Poll, timeout, pollfn); err != nil {
		e2elog.Failf("UDP service %v:%v not rejected: %v", host, port, err)
	}
}

// GetHTTPContent returns the content of the given url by HTTP.
func (j *TestJig) GetHTTPContent(host string, port int, timeout time.Duration, url string) bytes.Buffer {
	var body bytes.Buffer
	if pollErr := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		result := framework.PokeHTTP(host, port, url, nil)
		if result.Status == framework.HTTPSuccess {
			body.Write(result.Body)
			return true, nil
		}
		return false, nil
	}); pollErr != nil {
		e2elog.Failf("Could not reach HTTP service through %v:%v%v after %v: %v", host, port, url, timeout, pollErr)
	}
	return body
}

// TestHTTPHealthCheckNodePort tests a HTTP connection by the given request to the given host and port.
func (j *TestJig) TestHTTPHealthCheckNodePort(host string, port int, request string, timeout time.Duration, expectSucceed bool, threshold int) error {
	count := 0
	condition := func() (bool, error) {
		success, _ := testHTTPHealthCheckNodePort(host, port, request)
		if success && expectSucceed ||
			!success && !expectSucceed {
			count++
		}
		if count >= threshold {
			return true, nil
		}
		return false, nil
	}

	if err := wait.PollImmediate(time.Second, timeout, condition); err != nil {
		return fmt.Errorf("error waiting for healthCheckNodePort: expected at least %d succeed=%v on %v%v, got %d", threshold, expectSucceed, host, port, count)
	}
	return nil
}

// CreateServicePods creates a replication controller with the label same as service
func (j *TestJig) CreateServicePods(c clientset.Interface, ns string, replica int) {
	config := testutils.RCConfig{
		Client:       c,
		Name:         j.Name,
		Image:        framework.ServeHostnameImage,
		Command:      []string{"/agnhost", "serve-hostname"},
		Namespace:    ns,
		Labels:       j.Labels,
		PollInterval: 3 * time.Second,
		Timeout:      framework.PodReadyBeforeTimeout,
		Replicas:     replica,
	}
	err := framework.RunRC(config)
	framework.ExpectNoError(err, "Replica must be created")
}

// CheckAffinity function tests whether the service affinity works as expected.
// If affinity is expected, the test will return true once affinityConfirmCount
// number of same response observed in a row. If affinity is not expected, the
// test will keep observe until different responses observed. The function will
// return false only in case of unexpected errors.
func (j *TestJig) CheckAffinity(execPod *v1.Pod, targetIP string, targetPort int, shouldHold bool) bool {
	targetIPPort := net.JoinHostPort(targetIP, strconv.Itoa(targetPort))
	cmd := fmt.Sprintf(`curl -q -s --connect-timeout 2 http://%s/`, targetIPPort)
	timeout := TestTimeout
	if execPod == nil {
		timeout = LoadBalancerPollTimeout
	}
	var tracker affinityTracker
	if pollErr := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		if execPod != nil {
			stdout, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd)
			if err != nil {
				e2elog.Logf("Failed to get response from %s. Retry until timeout", targetIPPort)
				return false, nil
			}
			tracker.recordHost(stdout)
		} else {
			rawResponse := j.GetHTTPContent(targetIP, targetPort, timeout, "")
			tracker.recordHost(rawResponse.String())
		}
		trackerFulfilled, affinityHolds := tracker.checkHostTrace(AffinityConfirmCount)
		if !shouldHold && !affinityHolds {
			return true, nil
		}
		if shouldHold && trackerFulfilled && affinityHolds {
			return true, nil
		}
		return false, nil
	}); pollErr != nil {
		trackerFulfilled, _ := tracker.checkHostTrace(AffinityConfirmCount)
		if pollErr != wait.ErrWaitTimeout {
			checkAffinityFailed(tracker, pollErr.Error())
			return false
		}
		if !trackerFulfilled {
			checkAffinityFailed(tracker, fmt.Sprintf("Connection to %s timed out or not enough responses.", targetIPPort))
		}
		if shouldHold {
			checkAffinityFailed(tracker, "Affinity should hold but didn't.")
		} else {
			checkAffinityFailed(tracker, "Affinity shouldn't hold but did.")
		}
		return true
	}
	return true
}

func testHTTPHealthCheckNodePort(ip string, port int, request string) (bool, error) {
	ipPort := net.JoinHostPort(ip, strconv.Itoa(port))
	url := fmt.Sprintf("http://%s%s", ipPort, request)
	if ip == "" || port == 0 {
		e2elog.Failf("Got empty IP for reachability check (%s)", url)
		return false, fmt.Errorf("invalid input ip or port")
	}
	e2elog.Logf("Testing HTTP health check on %v", url)
	resp, err := httpGetNoConnectionPoolTimeout(url, 5*time.Second)
	if err != nil {
		e2elog.Logf("Got error testing for reachability of %s: %v", url, err)
		return false, err
	}
	defer resp.Body.Close()
	if err != nil {
		e2elog.Logf("Got error reading response from %s: %v", url, err)
		return false, err
	}
	// HealthCheck responder returns 503 for no local endpoints
	if resp.StatusCode == 503 {
		return false, nil
	}
	// HealthCheck responder returns 200 for non-zero local endpoints
	if resp.StatusCode == 200 {
		return true, nil
	}
	return false, fmt.Errorf("unexpected HTTP response code %s from health check responder at %s", resp.Status, url)
}

// Does an HTTP GET, but does not reuse TCP connections
// This masks problems where the iptables rule has changed, but we don't see it
func httpGetNoConnectionPoolTimeout(url string, timeout time.Duration) (*http.Response, error) {
	tr := utilnet.SetTransportDefaults(&http.Transport{
		DisableKeepAlives: true,
	})
	client := &http.Client{
		Transport: tr,
		Timeout:   timeout,
	}
	return client.Get(url)
}

// CreatePausePodDeployment creates a deployment for agnhost-pause pod running in different nodes
func (j *TestJig) CreatePausePodDeployment(name, ns string, replica int32) *appsv1.Deployment {
	// terminationGracePeriod is set to 0 to reduce deployment deletion time for infinitely running pause pod.
	terminationGracePeriod := int64(0)
	pauseDeployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: PauseDeploymentLabels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replica,
			Selector: &metav1.LabelSelector{
				MatchLabels: PauseDeploymentLabels,
			},
			Strategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: PauseDeploymentLabels,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &terminationGracePeriod,
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{MatchLabels: PauseDeploymentLabels},
									TopologyKey:   "kubernetes.io/hostname",
									Namespaces:    []string{ns},
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:  "agnhost-pause",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Args:  []string{"pause"},
						},
					},
				},
			},
		},
	}
	deployment, err := j.Client.AppsV1().Deployments(ns).Create(pauseDeployment)
	framework.ExpectNoError(err, "Error in creating deployment for pause pod")
	return deployment
}
