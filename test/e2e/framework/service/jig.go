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
	"context"
	"fmt"
	"net"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// NodePortRange should match whatever the default/configured range is
var NodePortRange = utilnet.PortRange{Base: 30000, Size: 2768}

// TestJig is a test jig to help service testing.
type TestJig struct {
	Client    clientset.Interface
	Namespace string
	Name      string
	ID        string
	Labels    map[string]string
}

// NewTestJig allocates and inits a new TestJig.
func NewTestJig(client clientset.Interface, namespace, name string) *TestJig {
	j := &TestJig{}
	j.Client = client
	j.Namespace = namespace
	j.Name = name
	j.ID = j.Name + "-" + string(uuid.NewUUID())
	j.Labels = map[string]string{"testid": j.ID}

	return j
}

// newServiceTemplate returns the default v1.Service template for this j, but
// does not actually create the Service.  The default Service has the same name
// as the j and exposes the given port.
func (j *TestJig) newServiceTemplate(proto v1.Protocol, port int32) *v1.Service {
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: j.Namespace,
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
func (j *TestJig) CreateTCPServiceWithPort(tweak func(svc *v1.Service), port int32) (*v1.Service, error) {
	svc := j.newServiceTemplate(v1.ProtocolTCP, port)
	if tweak != nil {
		tweak(svc)
	}
	result, err := j.Client.CoreV1().Services(j.Namespace).Create(context.TODO(), svc, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create TCP Service %q: %v", svc.Name, err)
	}
	return j.sanityCheckService(result, svc.Spec.Type)
}

// CreateTCPService creates a new TCP Service based on the j's
// defaults.  Callers can provide a function to tweak the Service object before
// it is created.
func (j *TestJig) CreateTCPService(tweak func(svc *v1.Service)) (*v1.Service, error) {
	return j.CreateTCPServiceWithPort(tweak, 80)
}

// CreateUDPService creates a new UDP Service based on the j's
// defaults.  Callers can provide a function to tweak the Service object before
// it is created.
func (j *TestJig) CreateUDPService(tweak func(svc *v1.Service)) (*v1.Service, error) {
	svc := j.newServiceTemplate(v1.ProtocolUDP, 80)
	if tweak != nil {
		tweak(svc)
	}
	result, err := j.Client.CoreV1().Services(j.Namespace).Create(context.TODO(), svc, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create UDP Service %q: %v", svc.Name, err)
	}
	return j.sanityCheckService(result, svc.Spec.Type)
}

// CreateExternalNameService creates a new ExternalName type Service based on the j's defaults.
// Callers can provide a function to tweak the Service object before it is created.
func (j *TestJig) CreateExternalNameService(tweak func(svc *v1.Service)) (*v1.Service, error) {
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: j.Namespace,
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
	result, err := j.Client.CoreV1().Services(j.Namespace).Create(context.TODO(), svc, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create ExternalName Service %q: %v", svc.Name, err)
	}
	return j.sanityCheckService(result, svc.Spec.Type)
}

// ChangeServiceType updates the given service's ServiceType to the given newType.
func (j *TestJig) ChangeServiceType(newType v1.ServiceType, timeout time.Duration) error {
	ingressIP := ""
	svc, err := j.UpdateService(func(s *v1.Service) {
		for _, ing := range s.Status.LoadBalancer.Ingress {
			if ing.IP != "" {
				ingressIP = ing.IP
			}
		}
		s.Spec.Type = newType
		s.Spec.Ports[0].NodePort = 0
	})
	if err != nil {
		return err
	}
	if ingressIP != "" {
		_, err = j.WaitForLoadBalancerDestroy(ingressIP, int(svc.Spec.Ports[0].Port), timeout)
	}
	return err
}

// CreateOnlyLocalNodePortService creates a NodePort service with
// ExternalTrafficPolicy set to Local and sanity checks its nodePort.
// If createPod is true, it also creates an RC with 1 replica of
// the standard netexec container used everywhere in this test.
func (j *TestJig) CreateOnlyLocalNodePortService(createPod bool) (*v1.Service, error) {
	ginkgo.By("creating a service " + j.Namespace + "/" + j.Name + " with type=NodePort and ExternalTrafficPolicy=Local")
	svc, err := j.CreateTCPService(func(svc *v1.Service) {
		svc.Spec.Type = v1.ServiceTypeNodePort
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		svc.Spec.Ports = []v1.ServicePort{{Protocol: v1.ProtocolTCP, Port: 80}}
	})
	if err != nil {
		return nil, err
	}

	if createPod {
		ginkgo.By("creating a pod to be part of the service " + j.Name)
		_, err = j.Run(nil)
		if err != nil {
			return nil, err
		}
	}
	return svc, nil
}

// CreateOnlyLocalLoadBalancerService creates a loadbalancer service with
// ExternalTrafficPolicy set to Local and waits for it to acquire an ingress IP.
// If createPod is true, it also creates an RC with 1 replica of
// the standard netexec container used everywhere in this test.
func (j *TestJig) CreateOnlyLocalLoadBalancerService(timeout time.Duration, createPod bool,
	tweak func(svc *v1.Service)) (*v1.Service, error) {
	_, err := j.CreateLoadBalancerService(timeout, func(svc *v1.Service) {
		ginkgo.By("setting ExternalTrafficPolicy=Local")
		svc.Spec.ExternalTrafficPolicy = v1.ServiceExternalTrafficPolicyTypeLocal
		if tweak != nil {
			tweak(svc)
		}
	})
	if err != nil {
		return nil, err
	}

	if createPod {
		ginkgo.By("creating a pod to be part of the service " + j.Name)
		_, err = j.Run(nil)
		if err != nil {
			return nil, err
		}
	}
	ginkgo.By("waiting for loadbalancer for service " + j.Namespace + "/" + j.Name)
	return j.WaitForLoadBalancer(timeout)
}

// CreateLoadBalancerService creates a loadbalancer service and waits
// for it to acquire an ingress IP.
func (j *TestJig) CreateLoadBalancerService(timeout time.Duration, tweak func(svc *v1.Service)) (*v1.Service, error) {
	ginkgo.By("creating a service " + j.Namespace + "/" + j.Name + " with type=LoadBalancer")
	svc := j.newServiceTemplate(v1.ProtocolTCP, 80)
	svc.Spec.Type = v1.ServiceTypeLoadBalancer
	// We need to turn affinity off for our LB distribution tests
	svc.Spec.SessionAffinity = v1.ServiceAffinityNone
	if tweak != nil {
		tweak(svc)
	}
	_, err := j.Client.CoreV1().Services(j.Namespace).Create(context.TODO(), svc, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create LoadBalancer Service %q: %v", svc.Name, err)
	}

	ginkgo.By("waiting for loadbalancer for service " + j.Namespace + "/" + j.Name)
	return j.WaitForLoadBalancer(timeout)
}

// GetEndpointNodes returns a map of nodenames:external-ip on which the
// endpoints of the Service are running.
func (j *TestJig) GetEndpointNodes() (map[string][]string, error) {
	nodes, err := e2enode.GetBoundedReadySchedulableNodes(j.Client, MaxNodesForEndpointsTests)
	if err != nil {
		return nil, err
	}
	epNodes, err := j.GetEndpointNodeNames()
	if err != nil {
		return nil, err
	}
	nodeMap := map[string][]string{}
	for _, n := range nodes.Items {
		if epNodes.Has(n.Name) {
			nodeMap[n.Name] = e2enode.GetAddresses(&n, v1.NodeExternalIP)
		}
	}
	return nodeMap, nil
}

// GetEndpointNodeNames returns a string set of node names on which the
// endpoints of the given Service are running.
func (j *TestJig) GetEndpointNodeNames() (sets.String, error) {
	endpoints, err := j.Client.CoreV1().Endpoints(j.Namespace).Get(context.TODO(), j.Name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("get endpoints for service %s/%s failed (%s)", j.Namespace, j.Name, err)
	}
	if len(endpoints.Subsets) == 0 {
		return nil, fmt.Errorf("endpoint has no subsets, cannot determine node addresses")
	}
	epNodes := sets.NewString()
	for _, ss := range endpoints.Subsets {
		for _, e := range ss.Addresses {
			if e.NodeName != nil {
				epNodes.Insert(*e.NodeName)
			}
		}
	}
	return epNodes, nil
}

// WaitForEndpointOnNode waits for a service endpoint on the given node.
func (j *TestJig) WaitForEndpointOnNode(nodeName string) error {
	return wait.PollImmediate(framework.Poll, LoadBalancerPropagationTimeoutDefault, func() (bool, error) {
		endpoints, err := j.Client.CoreV1().Endpoints(j.Namespace).Get(context.TODO(), j.Name, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Get endpoints for service %s/%s failed (%s)", j.Namespace, j.Name, err)
			return false, nil
		}
		if len(endpoints.Subsets) == 0 {
			framework.Logf("Expect endpoints with subsets, got none.")
			return false, nil
		}
		// TODO: Handle multiple endpoints
		if len(endpoints.Subsets[0].Addresses) == 0 {
			framework.Logf("Expected Ready endpoints - found none")
			return false, nil
		}
		epHostName := *endpoints.Subsets[0].Addresses[0].NodeName
		framework.Logf("Pod for service %s/%s is on node %s", j.Namespace, j.Name, epHostName)
		if epHostName != nodeName {
			framework.Logf("Found endpoint on wrong node, expected %v, got %v", nodeName, epHostName)
			return false, nil
		}
		return true, nil
	})
}

// WaitForAvailableEndpoint waits for at least 1 endpoint to be available till timeout
func (j *TestJig) WaitForAvailableEndpoint(timeout time.Duration) error {
	//Wait for endpoints to be created, this may take longer time if service backing pods are taking longer time to run
	endpointSelector := fields.OneTermEqualSelector("metadata.name", j.Name)
	stopCh := make(chan struct{})
	endpointAvailable := false
	var controller cache.Controller
	_, controller = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.FieldSelector = endpointSelector.String()
				obj, err := j.Client.CoreV1().Endpoints(j.Namespace).List(context.TODO(), options)
				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.FieldSelector = endpointSelector.String()
				return j.Client.CoreV1().Endpoints(j.Namespace).Watch(context.TODO(), options)
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
	if err != nil {
		return fmt.Errorf("no subset of available IP address found for the endpoint %s within timeout %v", j.Name, timeout)
	}
	return nil
}

// sanityCheckService performs sanity checks on the given service; in particular, ensuring
// that creating/updating a service allocates IPs, ports, etc, as needed. It does not
// check for ingress assignment as that happens asynchronously after the Service is created.
func (j *TestJig) sanityCheckService(svc *v1.Service, svcType v1.ServiceType) (*v1.Service, error) {
	if svcType == "" {
		svcType = v1.ServiceTypeClusterIP
	}
	if svc.Spec.Type != svcType {
		return nil, fmt.Errorf("unexpected Spec.Type (%s) for service, expected %s", svc.Spec.Type, svcType)
	}

	if svcType != v1.ServiceTypeExternalName {
		if svc.Spec.ExternalName != "" {
			return nil, fmt.Errorf("unexpected Spec.ExternalName (%s) for service, expected empty", svc.Spec.ExternalName)
		}
		if svc.Spec.ClusterIP == "" {
			return nil, fmt.Errorf("didn't get ClusterIP for non-ExternalName service")
		}
	} else {
		if svc.Spec.ClusterIP != "" {
			return nil, fmt.Errorf("unexpected Spec.ClusterIP (%s) for ExternalName service, expected empty", svc.Spec.ClusterIP)
		}
	}

	expectNodePorts := false
	if svcType != v1.ServiceTypeClusterIP && svcType != v1.ServiceTypeExternalName {
		expectNodePorts = true
	}
	for i, port := range svc.Spec.Ports {
		hasNodePort := (port.NodePort != 0)
		if hasNodePort != expectNodePorts {
			return nil, fmt.Errorf("unexpected Spec.Ports[%d].NodePort (%d) for service", i, port.NodePort)
		}
		if hasNodePort {
			if !NodePortRange.Contains(int(port.NodePort)) {
				return nil, fmt.Errorf("out-of-range nodePort (%d) for service", port.NodePort)
			}
		}
	}

	// FIXME: this fails for tests that were changed from LoadBalancer to ClusterIP.
	// if svcType != v1.ServiceTypeLoadBalancer {
	// 	if len(svc.Status.LoadBalancer.Ingress) != 0 {
	// 		return nil, fmt.Errorf("unexpected Status.LoadBalancer.Ingress on non-LoadBalancer service")
	// 	}
	// }

	return svc, nil
}

// UpdateService fetches a service, calls the update function on it, and
// then attempts to send the updated service. It tries up to 3 times in the
// face of timeouts and conflicts.
func (j *TestJig) UpdateService(update func(*v1.Service)) (*v1.Service, error) {
	for i := 0; i < 3; i++ {
		service, err := j.Client.CoreV1().Services(j.Namespace).Get(context.TODO(), j.Name, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("failed to get Service %q: %v", j.Name, err)
		}
		update(service)
		result, err := j.Client.CoreV1().Services(j.Namespace).Update(context.TODO(), service, metav1.UpdateOptions{})
		if err == nil {
			return j.sanityCheckService(result, service.Spec.Type)
		}
		if !apierrors.IsConflict(err) && !apierrors.IsServerTimeout(err) {
			return nil, fmt.Errorf("failed to update Service %q: %v", j.Name, err)
		}
	}
	return nil, fmt.Errorf("too many retries updating Service %q", j.Name)
}

// WaitForNewIngressIP waits for the given service to get a new ingress IP, or returns an error after the given timeout
func (j *TestJig) WaitForNewIngressIP(existingIP string, timeout time.Duration) (*v1.Service, error) {
	framework.Logf("Waiting up to %v for service %q to get a new ingress IP", timeout, j.Name)
	service, err := j.waitForCondition(timeout, "have a new ingress IP", func(svc *v1.Service) bool {
		if len(svc.Status.LoadBalancer.Ingress) == 0 {
			return false
		}
		ip := svc.Status.LoadBalancer.Ingress[0].IP
		if ip == "" || ip == existingIP {
			return false
		}
		return true
	})
	if err != nil {
		return nil, err
	}
	return j.sanityCheckService(service, v1.ServiceTypeLoadBalancer)
}

// ChangeServiceNodePort changes node ports of the given service.
func (j *TestJig) ChangeServiceNodePort(initial int) (*v1.Service, error) {
	var err error
	var service *v1.Service
	for i := 1; i < NodePortRange.Size; i++ {
		offs1 := initial - NodePortRange.Base
		offs2 := (offs1 + i) % NodePortRange.Size
		newPort := NodePortRange.Base + offs2
		service, err = j.UpdateService(func(s *v1.Service) {
			s.Spec.Ports[0].NodePort = int32(newPort)
		})
		if err != nil && strings.Contains(err.Error(), portallocator.ErrAllocated.Error()) {
			framework.Logf("tried nodePort %d, but it is in use, will try another", newPort)
			continue
		}
		// Otherwise err was nil or err was a real error
		break
	}
	return service, err
}

// WaitForLoadBalancer waits the given service to have a LoadBalancer, or returns an error after the given timeout
func (j *TestJig) WaitForLoadBalancer(timeout time.Duration) (*v1.Service, error) {
	framework.Logf("Waiting up to %v for service %q to have a LoadBalancer", timeout, j.Name)
	service, err := j.waitForCondition(timeout, "have a load balancer", func(svc *v1.Service) bool {
		return len(svc.Status.LoadBalancer.Ingress) > 0
	})
	if err != nil {
		return nil, err
	}

	for i, ing := range service.Status.LoadBalancer.Ingress {
		if ing.IP == "" && ing.Hostname == "" {
			return nil, fmt.Errorf("unexpected Status.LoadBalancer.Ingress[%d] for service: %#v", i, ing)
		}
	}

	return j.sanityCheckService(service, v1.ServiceTypeLoadBalancer)
}

// WaitForLoadBalancerDestroy waits the given service to destroy a LoadBalancer, or returns an error after the given timeout
func (j *TestJig) WaitForLoadBalancerDestroy(ip string, port int, timeout time.Duration) (*v1.Service, error) {
	// TODO: once support ticket 21807001 is resolved, reduce this timeout back to something reasonable
	defer func() {
		if err := framework.EnsureLoadBalancerResourcesDeleted(ip, strconv.Itoa(port)); err != nil {
			framework.Logf("Failed to delete cloud resources for service: %s %d (%v)", ip, port, err)
		}
	}()

	framework.Logf("Waiting up to %v for service %q to have no LoadBalancer", timeout, j.Name)
	service, err := j.waitForCondition(timeout, "have no load balancer", func(svc *v1.Service) bool {
		return len(svc.Status.LoadBalancer.Ingress) == 0
	})
	if err != nil {
		return nil, err
	}
	return j.sanityCheckService(service, service.Spec.Type)
}

func (j *TestJig) waitForCondition(timeout time.Duration, message string, conditionFn func(*v1.Service) bool) (*v1.Service, error) {
	var service *v1.Service
	pollFunc := func() (bool, error) {
		svc, err := j.Client.CoreV1().Services(j.Namespace).Get(context.TODO(), j.Name, metav1.GetOptions{})
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
		return nil, fmt.Errorf("timed out waiting for service %q to %s", j.Name, message)
	}
	return service, nil
}

// newRCTemplate returns the default v1.ReplicationController object for
// this j, but does not actually create the RC.  The default RC has the same
// name as the j and runs the "netexec" container.
func (j *TestJig) newRCTemplate() *v1.ReplicationController {
	var replicas int32 = 1
	var grace int64 = 3 // so we don't race with kube-proxy when scaling up/down

	rc := &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: j.Namespace,
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

// CreatePDB returns a PodDisruptionBudget for the given ReplicationController, or returns an error if a PodDisruptionBudget isn't ready
func (j *TestJig) CreatePDB(rc *v1.ReplicationController) (*policyv1beta1.PodDisruptionBudget, error) {
	pdb := j.newPDBTemplate(rc)
	newPdb, err := j.Client.PolicyV1beta1().PodDisruptionBudgets(j.Namespace).Create(context.TODO(), pdb, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create PDB %q %v", pdb.Name, err)
	}
	if err := j.waitForPdbReady(); err != nil {
		return nil, fmt.Errorf("failed waiting for PDB to be ready: %v", err)
	}

	return newPdb, nil
}

// newPDBTemplate returns the default policyv1beta1.PodDisruptionBudget object for
// this j, but does not actually create the PDB.  The default PDB specifies a
// MinAvailable of N-1 and matches the pods created by the RC.
func (j *TestJig) newPDBTemplate(rc *v1.ReplicationController) *policyv1beta1.PodDisruptionBudget {
	minAvailable := intstr.FromInt(int(*rc.Spec.Replicas) - 1)

	pdb := &policyv1beta1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: j.Namespace,
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

// Run creates a ReplicationController and Pod(s) and waits for the
// Pod(s) to be running. Callers can provide a function to tweak the RC object
// before it is created.
func (j *TestJig) Run(tweak func(rc *v1.ReplicationController)) (*v1.ReplicationController, error) {
	rc := j.newRCTemplate()
	if tweak != nil {
		tweak(rc)
	}
	result, err := j.Client.CoreV1().ReplicationControllers(j.Namespace).Create(context.TODO(), rc, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create RC %q: %v", rc.Name, err)
	}
	pods, err := j.waitForPodsCreated(int(*(rc.Spec.Replicas)))
	if err != nil {
		return nil, fmt.Errorf("failed to create pods: %v", err)
	}
	if err := j.waitForPodsReady(pods); err != nil {
		return nil, fmt.Errorf("failed waiting for pods to be running: %v", err)
	}
	return result, nil
}

// Scale scales pods to the given replicas
func (j *TestJig) Scale(replicas int) error {
	rc := j.Name
	scale, err := j.Client.CoreV1().ReplicationControllers(j.Namespace).GetScale(context.TODO(), rc, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get scale for RC %q: %v", rc, err)
	}

	scale.ResourceVersion = "" // indicate the scale update should be unconditional
	scale.Spec.Replicas = int32(replicas)
	_, err = j.Client.CoreV1().ReplicationControllers(j.Namespace).UpdateScale(context.TODO(), rc, scale, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to scale RC %q: %v", rc, err)
	}
	pods, err := j.waitForPodsCreated(replicas)
	if err != nil {
		return fmt.Errorf("failed waiting for pods: %v", err)
	}
	if err := j.waitForPodsReady(pods); err != nil {
		return fmt.Errorf("failed waiting for pods to be running: %v", err)
	}
	return nil
}

func (j *TestJig) waitForPdbReady() error {
	timeout := 2 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(2 * time.Second) {
		pdb, err := j.Client.PolicyV1beta1().PodDisruptionBudgets(j.Namespace).Get(context.TODO(), j.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if pdb.Status.DisruptionsAllowed > 0 {
			return nil
		}
	}

	return fmt.Errorf("timeout waiting for PDB %q to be ready", j.Name)
}

func (j *TestJig) waitForPodsCreated(replicas int) ([]string, error) {
	timeout := 2 * time.Minute
	// List the pods, making sure we observe all the replicas.
	label := labels.SelectorFromSet(labels.Set(j.Labels))
	framework.Logf("Waiting up to %v for %d pods to be created", timeout, replicas)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(2 * time.Second) {
		options := metav1.ListOptions{LabelSelector: label.String()}
		pods, err := j.Client.CoreV1().Pods(j.Namespace).List(context.TODO(), options)
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
			framework.Logf("Found all %d pods", replicas)
			return found, nil
		}
		framework.Logf("Found %d/%d pods - will retry", len(found), replicas)
	}
	return nil, fmt.Errorf("timeout waiting for %d pods to be created", replicas)
}

func (j *TestJig) waitForPodsReady(pods []string) error {
	timeout := 2 * time.Minute
	if !e2epod.CheckPodsRunningReady(j.Client, j.Namespace, pods, timeout) {
		return fmt.Errorf("timeout waiting for %d pods to be ready", len(pods))
	}
	return nil
}

func testReachabilityOverServiceName(serviceName string, sp v1.ServicePort, execPod *v1.Pod) error {
	return testEndpointReachability(serviceName, sp.Port, sp.Protocol, execPod)
}

func testReachabilityOverClusterIP(clusterIP string, sp v1.ServicePort, execPod *v1.Pod) error {
	// If .spec.clusterIP is set to "" or "None" for service, ClusterIP is not created, so reachability can not be tested over clusterIP:servicePort
	isClusterIPV46, err := regexp.MatchString(e2enetwork.RegexIPv4+"||"+e2enetwork.RegexIPv6, clusterIP)
	if err != nil {
		return fmt.Errorf("unable to parse ClusterIP: %s", clusterIP)
	}
	if isClusterIPV46 {
		return testEndpointReachability(clusterIP, sp.Port, sp.Protocol, execPod)
	}
	return nil
}

func testReachabilityOverNodePorts(nodes *v1.NodeList, sp v1.ServicePort, pod *v1.Pod, clusterIP string) error {
	internalAddrs := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)
	externalAddrs := e2enode.CollectAddresses(nodes, v1.NodeExternalIP)
	isClusterIPV4 := net.ParseIP(clusterIP).To4() != nil

	for _, internalAddr := range internalAddrs {
		// If the node's internal address points to localhost, then we are not
		// able to test the service reachability via that address
		if isInvalidOrLocalhostAddress(internalAddr) {
			framework.Logf("skipping testEndpointReachability() for internal adddress %s", internalAddr)
			continue
		}
		isNodeInternalIPV4 := net.ParseIP(internalAddr).To4() != nil
		// Check service reachability on the node internalIP which is same family
		// as clusterIP
		if isClusterIPV4 != isNodeInternalIPV4 {
			framework.Logf("skipping testEndpointReachability() for internal adddress %s as it does not match clusterIP (%s) family", internalAddr, clusterIP)
			continue
		}

		err := testEndpointReachability(internalAddr, sp.NodePort, sp.Protocol, pod)
		if err != nil {
			return err
		}
	}
	for _, externalAddr := range externalAddrs {
		isNodeExternalIPV4 := net.ParseIP(externalAddr).To4() != nil
		if isClusterIPV4 != isNodeExternalIPV4 {
			framework.Logf("skipping testEndpointReachability() for external adddress %s as it does not match clusterIP (%s) family", externalAddr, clusterIP)
			continue
		}
		err := testEndpointReachability(externalAddr, sp.NodePort, sp.Protocol, pod)
		if err != nil {
			return err
		}
	}
	return nil
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
func testEndpointReachability(endpoint string, port int32, protocol v1.Protocol, execPod *v1.Pod) error {
	ep := net.JoinHostPort(endpoint, strconv.Itoa(int(port)))
	cmd := ""
	switch protocol {
	case v1.ProtocolTCP:
		cmd = fmt.Sprintf("nc -zv -t -w 2 %s %v", endpoint, port)
	case v1.ProtocolUDP:
		cmd = fmt.Sprintf("nc -zv -u -w 2 %s %v", endpoint, port)
	default:
		return fmt.Errorf("service reachablity check is not supported for %v", protocol)
	}

	err := wait.PollImmediate(1*time.Second, ServiceReachabilityShortPollTimeout, func() (bool, error) {
		if _, err := framework.RunHostCmd(execPod.Namespace, execPod.Name, cmd); err != nil {
			framework.Logf("Service reachability failing with error: %v\nRetrying...", err)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("service is not reachable within %v timeout on endpoint %s over %s protocol", ServiceReachabilityShortPollTimeout, ep, protocol)
	}
	return nil
}

// checkClusterIPServiceReachability ensures that service of type ClusterIP is reachable over
// - ServiceName:ServicePort, ClusterIP:ServicePort
func (j *TestJig) checkClusterIPServiceReachability(svc *v1.Service, pod *v1.Pod) error {
	clusterIP := svc.Spec.ClusterIP
	servicePorts := svc.Spec.Ports

	err := j.WaitForAvailableEndpoint(ServiceEndpointsTimeout)
	if err != nil {
		return err
	}

	for _, servicePort := range servicePorts {
		err = testReachabilityOverServiceName(svc.Name, servicePort, pod)
		if err != nil {
			return err
		}
		err = testReachabilityOverClusterIP(clusterIP, servicePort, pod)
		if err != nil {
			return err
		}
	}
	return nil
}

// checkNodePortServiceReachability ensures that service of type nodePort are reachable
// - Internal clients should be reachable to service over -
//   	ServiceName:ServicePort, ClusterIP:ServicePort and NodeInternalIPs:NodePort
// - External clients should be reachable to service over -
//   	NodePublicIPs:NodePort
func (j *TestJig) checkNodePortServiceReachability(svc *v1.Service, pod *v1.Pod) error {
	clusterIP := svc.Spec.ClusterIP
	servicePorts := svc.Spec.Ports

	// Consider only 2 nodes for testing
	nodes, err := e2enode.GetBoundedReadySchedulableNodes(j.Client, 2)
	if err != nil {
		return err
	}

	err = j.WaitForAvailableEndpoint(ServiceEndpointsTimeout)
	if err != nil {
		return err
	}

	for _, servicePort := range servicePorts {
		err = testReachabilityOverServiceName(svc.Name, servicePort, pod)
		if err != nil {
			return err
		}
		err = testReachabilityOverClusterIP(clusterIP, servicePort, pod)
		if err != nil {
			return err
		}
		err = testReachabilityOverNodePorts(nodes, servicePort, pod, clusterIP)
		if err != nil {
			return err
		}
	}

	return nil
}

// checkExternalServiceReachability ensures service of type externalName resolves to IP address and no fake externalName is set
// FQDN of kubernetes is used as externalName(for air tight platforms).
func (j *TestJig) checkExternalServiceReachability(svc *v1.Service, pod *v1.Pod) error {
	// Service must resolve to IP
	cmd := fmt.Sprintf("nslookup %s", svc.Name)
	_, err := framework.RunHostCmd(pod.Namespace, pod.Name, cmd)
	if err != nil {
		return fmt.Errorf("ExternalName service %q must resolve to IP", pod.Namespace+"/"+pod.Name)
	}
	return nil
}

// CheckServiceReachability ensures that request are served by the services. Only supports Services with type ClusterIP, NodePort and ExternalName.
func (j *TestJig) CheckServiceReachability(svc *v1.Service, pod *v1.Pod) error {
	svcType := svc.Spec.Type

	_, err := j.sanityCheckService(svc, svcType)
	if err != nil {
		return err
	}

	switch svcType {
	case v1.ServiceTypeClusterIP:
		return j.checkClusterIPServiceReachability(svc, pod)
	case v1.ServiceTypeNodePort:
		return j.checkNodePortServiceReachability(svc, pod)
	case v1.ServiceTypeExternalName:
		return j.checkExternalServiceReachability(svc, pod)
	default:
		return fmt.Errorf("unsupported service type \"%s\" to verify service reachability for \"%s\" service. This may due to diverse implementation of the service type", svcType, svc.Name)
	}
}

// CreateServicePods creates a replication controller with the label same as service. Service listens to HTTP.
func (j *TestJig) CreateServicePods(replica int) error {
	config := testutils.RCConfig{
		Client:       j.Client,
		Name:         j.Name,
		Image:        framework.ServeHostnameImage,
		Command:      []string{"/agnhost", "serve-hostname"},
		Namespace:    j.Namespace,
		Labels:       j.Labels,
		PollInterval: 3 * time.Second,
		Timeout:      framework.PodReadyBeforeTimeout,
		Replicas:     replica,
	}
	return e2erc.RunRC(config)
}

// CreateTCPUDPServicePods creates a replication controller with the label same as service. Service listens to TCP and UDP.
func (j *TestJig) CreateTCPUDPServicePods(replica int) error {
	config := testutils.RCConfig{
		Client:       j.Client,
		Name:         j.Name,
		Image:        framework.ServeHostnameImage,
		Command:      []string{"/agnhost", "serve-hostname", "--http=false", "--tcp", "--udp"},
		Namespace:    j.Namespace,
		Labels:       j.Labels,
		PollInterval: 3 * time.Second,
		Timeout:      framework.PodReadyBeforeTimeout,
		Replicas:     replica,
	}
	return e2erc.RunRC(config)
}
