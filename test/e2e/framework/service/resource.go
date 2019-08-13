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
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
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
		service, err = c.CoreV1().Services(namespace).Get(serviceName, metav1.GetOptions{})
		if err != nil {
			return service, err
		}

		update(service)

		service, err = c.CoreV1().Services(namespace).Update(service)

		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
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

// EnableAndDisableInternalLB returns two functions for enabling and disabling the internal load balancer
// setting for the supported cloud providers (currently GCE/GKE and Azure) and empty functions for others.
func EnableAndDisableInternalLB() (enable func(svc *v1.Service), disable func(svc *v1.Service)) {
	return framework.TestContext.CloudConfig.Provider.EnableAndDisableInternalLB()
}

// DescribeSvc logs the output of kubectl describe svc for the given namespace
func DescribeSvc(ns string) {
	e2elog.Logf("\nOutput of kubectl describe svc:\n")
	desc, _ := framework.RunKubectl(
		"describe", "svc", fmt.Sprintf("--namespace=%v", ns))
	e2elog.Logf(desc)
}

// GetServiceLoadBalancerCreationTimeout returns a timeout value for creating a load balancer of a service.
func GetServiceLoadBalancerCreationTimeout(cs clientset.Interface) time.Duration {
	if nodes := framework.GetReadySchedulableNodesOrDie(cs); len(nodes.Items) > LargeClusterMinNodesNumber {
		return LoadBalancerCreateTimeoutLarge
	}
	return LoadBalancerCreateTimeoutDefault
}
