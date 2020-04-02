/*
Copyright 2019 The Kubernetes Authors.

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
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	testutils "k8s.io/kubernetes/test/utils"
)

// GetServicesProxyRequest returns a request for a service proxy.
func GetServicesProxyRequest(c clientset.Interface, request *restclient.Request) (*restclient.Request, error) {
	return request.Resource("services").SubResource("proxy"), nil
}

// CreateServiceSpec returns a Service object for testing.
func CreateServiceSpec(serviceName, externalName string, isHeadless bool, selector map[string]string) *v1.Service {
	headlessService := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: serviceName,
		},
		Spec: v1.ServiceSpec{
			Selector: selector,
		},
	}
	if externalName != "" {
		headlessService.Spec.Type = v1.ServiceTypeExternalName
		headlessService.Spec.ExternalName = externalName
	} else {
		headlessService.Spec.Ports = []v1.ServicePort{
			{Port: 80, Name: "http", Protocol: v1.ProtocolTCP},
		}
	}
	if isHeadless {
		headlessService.Spec.ClusterIP = "None"
	}
	return headlessService
}

// UpdateService fetches a service, calls the update function on it,
// and then attempts to send the updated service. It retries up to 2
// times in the face of timeouts and conflicts.
func UpdateService(c clientset.Interface, namespace, serviceName string, update func(*v1.Service)) (*v1.Service, error) {
	var service *v1.Service
	var err error
	for i := 0; i < 3; i++ {
		service, err = c.CoreV1().Services(namespace).Get(context.TODO(), serviceName, metav1.GetOptions{})
		if err != nil {
			return service, err
		}

		update(service)

		service, err = c.CoreV1().Services(namespace).Update(context.TODO(), service, metav1.UpdateOptions{})

		if !apierrors.IsConflict(err) && !apierrors.IsServerTimeout(err) {
			return service, err
		}
	}
	return service, err
}

// CleanupServiceResources cleans up service Type=LoadBalancer resources.
func CleanupServiceResources(c clientset.Interface, loadBalancerName, region, zone string) {
	framework.TestContext.CloudConfig.Provider.CleanupServiceResources(c, loadBalancerName, region, zone)
}

// GetIngressPoint returns a host on which ingress serves.
func GetIngressPoint(ing *v1.LoadBalancerIngress) string {
	host := ing.IP
	if host == "" {
		host = ing.Hostname
	}
	return host
}

// GetServiceLoadBalancerCreationTimeout returns a timeout value for creating a load balancer of a service.
func GetServiceLoadBalancerCreationTimeout(cs clientset.Interface) time.Duration {
	nodes, err := e2enode.GetReadySchedulableNodes(cs)
	framework.ExpectNoError(err)
	if len(nodes.Items) > LargeClusterMinNodesNumber {
		return loadBalancerCreateTimeoutLarge
	}
	return loadBalancerCreateTimeoutDefault
}

// GetServiceLoadBalancerPropagationTimeout returns a timeout value for propagating a load balancer of a service.
func GetServiceLoadBalancerPropagationTimeout(cs clientset.Interface) time.Duration {
	nodes, err := e2enode.GetReadySchedulableNodes(cs)
	framework.ExpectNoError(err)
	if len(nodes.Items) > LargeClusterMinNodesNumber {
		return loadBalancerPropagationTimeoutLarge
	}
	return loadBalancerPropagationTimeoutDefault
}

// CreateServiceForSimpleAppWithPods is a convenience wrapper to create a service and its matching pods all at once.
func CreateServiceForSimpleAppWithPods(c clientset.Interface, contPort int, svcPort int, namespace, appName string, podSpec func(n v1.Node) v1.PodSpec, count int, block bool) (*v1.Service, error) {
	var err error
	theService := CreateServiceForSimpleApp(c, contPort, svcPort, namespace, appName)
	e2enode.CreatePodsPerNodeForSimpleApp(c, namespace, appName, podSpec, count)
	if block {
		err = testutils.WaitForPodsWithLabelRunning(c, namespace, labels.SelectorFromSet(labels.Set(theService.Spec.Selector)))
	}
	return theService, err
}

// CreateServiceForSimpleApp returns a service that selects/exposes pods (send -1 ports if no exposure needed) with an app label.
func CreateServiceForSimpleApp(c clientset.Interface, contPort, svcPort int, namespace, appName string) *v1.Service {
	if appName == "" {
		panic(fmt.Sprintf("no app name provided"))
	}

	serviceSelector := map[string]string{
		"app": appName + "-pod",
	}

	// For convenience, user sending ports are optional.
	portsFunc := func() []v1.ServicePort {
		if contPort < 1 || svcPort < 1 {
			return nil
		}
		return []v1.ServicePort{{
			Protocol:   v1.ProtocolTCP,
			Port:       int32(svcPort),
			TargetPort: intstr.FromInt(contPort),
		}}
	}
	framework.Logf("Creating a service-for-%v for selecting app=%v-pod", appName, appName)
	service, err := c.CoreV1().Services(namespace).Create(context.TODO(), &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "service-for-" + appName,
			Labels: map[string]string{
				"app": appName + "-service",
			},
		},
		Spec: v1.ServiceSpec{
			Ports:    portsFunc(),
			Selector: serviceSelector,
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return service
}
