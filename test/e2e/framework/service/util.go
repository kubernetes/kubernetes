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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	netutils "k8s.io/utils/net"
	"strings"
	"time"

	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
)

// TestReachableHTTP tests that the given host serves HTTP on the given port.
func TestReachableHTTP(ctx context.Context, host string, port int, timeout time.Duration) {
	TestReachableHTTPWithRetriableErrorCodes(ctx, host, port, []int{}, timeout)
}

// TestReachableHTTPWithRetriableErrorCodes tests that the given host serves HTTP on the given port with the given retriableErrCodes.
func TestReachableHTTPWithRetriableErrorCodes(ctx context.Context, host string, port int, retriableErrCodes []int, timeout time.Duration) {
	pollfn := func(ctx context.Context) (bool, error) {
		result := e2enetwork.PokeHTTP(host, port, "/echo?msg=hello",
			&e2enetwork.HTTPPokeParams{
				BodyContains:   "hello",
				RetriableCodes: retriableErrCodes,
			})
		if result.Status == e2enetwork.HTTPSuccess {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, true, pollfn); err != nil {
		if wait.Interrupted(err) {
			framework.Failf("Could not reach HTTP service through %v:%v after %v", host, port, timeout)
		} else {
			framework.Failf("Failed to reach HTTP service through %v:%v: %v", host, port, err)
		}
	}
}

func TestAllEndpointsReachable(ctx context.Context, client clientset.Interface, svc *v1.Service, pod *v1.Pod) error {
	switch svc.Spec.Type {
	case v1.ServiceTypeExternalName:
		return checkExternalServiceReachability(ctx, svc, pod)

	case v1.ServiceTypeClusterIP, v1.ServiceTypeLoadBalancer, v1.ServiceTypeNodePort:
		err := waitForAvailableEndpoint(ctx, client, svc.Namespace, svc.Name, ServiceEndpointsTimeout)
		if err != nil {
			return err
		}

		clusterIP := svc.Spec.ClusterIP
		servicePorts := svc.Spec.Ports

		if netutils.ParseIPSloppy(clusterIP) == nil {
			return fmt.Errorf("unable to parse ClusterIP: %s", clusterIP)
		}

		for _, servicePort := range servicePorts {
			// Check reachability using service name
			err = testEndpointReachability(ctx, svc.Name, servicePort.Port, servicePort.Protocol, pod)
			if err != nil {
				return err
			}

			// Check reachability using ClusterIP
			err = testEndpointReachability(ctx, clusterIP, servicePort.Port, servicePort.Protocol, pod)
			if err != nil {
				return err
			}

			// Check ExternalIP for ClusterIP and LoadBalancer type services
			if svc.Spec.Type == v1.ServiceTypeClusterIP || svc.Spec.Type == v1.ServiceTypeLoadBalancer {
				for _, externalIP := range svc.Spec.ExternalIPs {
					// Check reachability using ExternalIP
					err := testEndpointReachability(ctx, externalIP, servicePort.Port, servicePort.Protocol, pod)

					if err != nil {
						return err
					}
				}
			}

			// Check NodePorts for NodePort type services
			if svc.Spec.Type == v1.ServiceTypeNodePort {
				nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, client, 2)
				if err != nil {
					return err
				}

				err = testReachabilityOverNodePorts(ctx, nodes, servicePort, pod, clusterIP, false)
				if err != nil {
					return err
				}
			}

		}

	default:
		return fmt.Errorf("unsupported service type \"%s\" to verify service reachability for \"%s\" service. This may due to diverse implementation of the service type",
			svc.Spec.Type,
			svc.Name)
	}

	return nil
}

// checkExternalServiceReachability ensures service of type externalName resolves to IP address and no fake externalName is set
// FQDN of kubernetes is used as externalName(for air tight platforms).
func checkExternalServiceReachability(ctx context.Context, svc *v1.Service, pod *v1.Pod) error {
	// NOTE(claudiub): Windows does not support PQDN.
	svcName := fmt.Sprintf("%s.%s.svc.%s", svc.Name, svc.Namespace, framework.TestContext.ClusterDNSDomain)

	// Service must resolve to IP
	cmd := fmt.Sprintf("nslookup %s", svcName)

	return wait.PollUntilContextTimeout(ctx,
		framework.Poll,
		ServiceReachabilityShortPollTimeout,
		true,
		func(ctx context.Context) (done bool, err error) {
			_, stderr, err := e2epodoutput.RunHostCmdWithFullOutput(pod.Namespace, pod.Name, cmd)

			// NOTE(claudiub): nslookup may return 0 on Windows, even though the DNS name was not found. In this case,
			// we can check stderr for the error.
			if err != nil || (framework.NodeOSDistroIs("windows") && strings.Contains(stderr,
				fmt.Sprintf("can't find %s", svcName))) {
				framework.Logf("ExternalName service %q failed to resolve to IP", pod.Namespace+"/"+pod.Name)
				return false, nil
			}

			return true, nil
		})
}

// waitForAvailableEndpoint waits for at least 1 endpoint to be available till timeout
func waitForAvailableEndpoint(ctx context.Context, client clientset.Interface, ns string, name string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Wait for endpoints to be created, this may take longer time if service backing pods are taking longer time to run
	endpointSelector := fields.OneTermEqualSelector("metadata.name", name)
	endpointAvailable := false
	endpointSliceAvailable := false

	var controller cache.Controller

	_, controller = cache.NewInformerWithOptions(cache.InformerOptions{
		ListerWatcher: &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.FieldSelector = endpointSelector.String()

				obj, err := client.CoreV1().Endpoints(ns).List(ctx, options)

				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.FieldSelector = endpointSelector.String()

				return client.CoreV1().Endpoints(ns).Watch(ctx, options)
			},
		},
		ObjectType: &v1.Endpoints{},
		Handler: cache.ResourceEventHandlerFuncs{
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
		ResyncPeriod: 0,
	})

	go controller.Run(ctx.Done())

	var esController cache.Controller

	_, esController = cache.NewInformerWithOptions(cache.InformerOptions{
		ListerWatcher: &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.LabelSelector = "kubernetes.io/service-name=" + name

				obj, err := client.DiscoveryV1().EndpointSlices(ns).List(ctx, options)

				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = "kubernetes.io/service-name=" + name

				return client.DiscoveryV1().EndpointSlices(ns).Watch(ctx, options)
			},
		},
		ObjectType: &discoveryv1.EndpointSlice{},
		Handler: cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				if es, ok := obj.(*discoveryv1.EndpointSlice); ok {
					// TODO: currently we only consider addresses in 1 slice, but services with
					// a large number of endpoints (>1000) may have multiple slices. Some slices
					// with only a few addresses. We should check the addresses in all slices.
					if len(es.Endpoints) > 0 && len(es.Endpoints[0].Addresses) > 0 {
						endpointSliceAvailable = true
					}
				}
			},
			UpdateFunc: func(old, cur interface{}) {
				if es, ok := cur.(*discoveryv1.EndpointSlice); ok {
					// TODO: currently we only consider addresses in 1 slice, but services with
					// a large number of endpoints (>1000) may have multiple slices. Some slices
					// with only a few addresses. We should check the addresses in all slices.
					if len(es.Endpoints) > 0 && len(es.Endpoints[0].Addresses) > 0 {
						endpointSliceAvailable = true
					}
				}
			},
		},
		ResyncPeriod: 0,
	})

	go esController.Run(ctx.Done())

	err := wait.PollUntilContextCancel(ctx, 1*time.Second, false, func(ctx context.Context) (bool, error) {
		return endpointAvailable && endpointSliceAvailable, nil
	})

	if err != nil {
		return fmt.Errorf("no subset of available IP address found for the endpoint %s within timeout %v",
			name,
			timeout)
	}

	return nil
}
