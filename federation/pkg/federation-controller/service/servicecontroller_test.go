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
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/golang/glog"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fakefedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/pkg/federation-controller/service/ingress"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	"k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	fakekubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
)

const (
	retryInterval = 100 * time.Millisecond

	clusters  string = "clusters"
	services  string = "services"
	endpoints string = "endpoints"

	lbIngress1       = "10.20.30.40"
	lbIngress2       = "10.20.30.50"
	serviceEndpoint1 = "192.168.0.1"
	serviceEndpoint2 = "192.168.1.1"
)

var awfulError error = errors.NewGone("Something bad happened")

func TestServiceController(t *testing.T) {
	glog.Infof("Creating fake infrastructure")
	fedClient := &fakefedclientset.Clientset{}
	cluster1 := NewCluster("cluster1", v1.ConditionTrue)
	cluster2 := NewCluster("cluster2", v1.ConditionTrue)

	RegisterFakeClusterGet(&fedClient.Fake, &v1beta1.ClusterList{Items: []v1beta1.Cluster{*cluster1, *cluster2}})
	RegisterFakeList(clusters, &fedClient.Fake, &v1beta1.ClusterList{Items: []v1beta1.Cluster{*cluster1, *cluster2}})
	fedclusterWatch := RegisterFakeWatch(clusters, &fedClient.Fake)
	RegisterFakeList(services, &fedClient.Fake, &v1.ServiceList{Items: []v1.Service{}})
	fedServiceWatch := RegisterFakeWatch(services, &fedClient.Fake)
	RegisterFakeOnCreate(clusters, &fedClient.Fake, fedclusterWatch)
	RegisterFakeOnUpdate(clusters, &fedClient.Fake, fedclusterWatch)
	RegisterFakeOnCreate(services, &fedClient.Fake, fedServiceWatch)
	RegisterFakeOnUpdate(services, &fedClient.Fake, fedServiceWatch)
	RegisterFakeOnDelete(services, &fedClient.Fake, fedServiceWatch, serviceObjectGetter)

	cluster1Client := &fakekubeclientset.Clientset{}
	RegisterFakeList(services, &cluster1Client.Fake, &v1.ServiceList{Items: []v1.Service{}})
	c1ServiceWatch := RegisterFakeWatch(services, &cluster1Client.Fake)
	RegisterFakeList(endpoints, &cluster1Client.Fake, &v1.EndpointsList{Items: []v1.Endpoints{}})
	c1EndpointWatch := RegisterFakeWatch(endpoints, &cluster1Client.Fake)
	RegisterFakeOnCreate(services, &cluster1Client.Fake, c1ServiceWatch)
	RegisterFakeOnUpdate(services, &cluster1Client.Fake, c1ServiceWatch)
	RegisterFakeOnDelete(services, &cluster1Client.Fake, c1ServiceWatch, serviceObjectGetter)
	RegisterFakeOnCreate(endpoints, &cluster1Client.Fake, c1EndpointWatch)
	RegisterFakeOnUpdate(endpoints, &cluster1Client.Fake, c1EndpointWatch)

	cluster2Client := &fakekubeclientset.Clientset{}
	RegisterFakeList(services, &cluster2Client.Fake, &v1.ServiceList{Items: []v1.Service{}})
	c2ServiceWatch := RegisterFakeWatch(services, &cluster2Client.Fake)
	RegisterFakeList(endpoints, &cluster2Client.Fake, &v1.EndpointsList{Items: []v1.Endpoints{}})
	c2EndpointWatch := RegisterFakeWatch(endpoints, &cluster2Client.Fake)
	RegisterFakeOnCreate(services, &cluster2Client.Fake, c2ServiceWatch)
	RegisterFakeOnUpdate(services, &cluster2Client.Fake, c2ServiceWatch)
	RegisterFakeOnDelete(services, &cluster2Client.Fake, c2ServiceWatch, serviceObjectGetter)
	RegisterFakeOnCreate(endpoints, &cluster2Client.Fake, c2EndpointWatch)
	RegisterFakeOnUpdate(endpoints, &cluster2Client.Fake, c2EndpointWatch)

	fedInformerClientFactory := func(cluster *v1beta1.Cluster) (kubeclientset.Interface, error) {
		switch cluster.Name {
		case cluster1.Name:
			return cluster1Client, nil
		case cluster2.Name:
			return cluster2Client, nil
		default:
			return nil, fmt.Errorf("Unknown cluster: %v", cluster.Name)
		}
	}

	sc := New(fedClient)
	ToFederatedInformerForTestOnly(sc.federatedInformer).SetClientFactory(fedInformerClientFactory)
	ToFederatedInformerForTestOnly(sc.endpointFederatedInformer).SetClientFactory(fedInformerClientFactory)
	sc.clusterAvailableDelay = 100 * time.Millisecond
	sc.reviewDelay = 50 * time.Millisecond
	sc.updateTimeout = 5 * time.Second

	stop := make(chan struct{})
	glog.Infof("Running Service Controller")
	go sc.Run(5, stop)

	glog.Infof("Adding cluster 1")
	fedclusterWatch.Add(cluster1)

	service := NewService("test-service-1", 80)

	glog.Infof("Adding federated service")
	fedServiceWatch.Add(service)
	key := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}.String()

	glog.Infof("Test service was correctly created in cluster 1")
	require.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster1.Name,
		key, service, wait.ForeverTestTimeout))

	glog.Infof("Adding cluster 2")
	fedclusterWatch.Add(cluster2)

	glog.Infof("Test service was correctly created in cluster 2")
	require.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster2.Name,
		key, service, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster1 service status is updated")
	service.Status = v1.ServiceStatus{
		LoadBalancer: v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{IP: lbIngress1},
			}}}

	desiredStatus := service.Status
	desiredService := &v1.Service{Status: desiredStatus}

	c1ServiceWatch.Modify(service)
	require.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster1.Name,
		key, service, wait.ForeverTestTimeout))
	require.NoError(t, WaitForFederatedServiceUpdate(t, sc.serviceStore,
		key, desiredService, serviceStatusCompare, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster1 endpoint for the service is created")
	desiredIngressAnnotation := ingress.NewFederatedServiceIngress().
		AddEndpoints("cluster1", []string{lbIngress1}).
		String()
	desiredService = &v1.Service{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{ingress.FederatedServiceIngressAnnotation: desiredIngressAnnotation}}}
	c1EndpointWatch.Add(NewEndpoint("test-service-1", serviceEndpoint1))
	require.NoError(t, WaitForFederatedServiceUpdate(t, sc.serviceStore,
		key, desiredService, serviceIngressCompare, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster2 service status is updated")
	service.Status = v1.ServiceStatus{
		LoadBalancer: v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{IP: lbIngress2},
			}}}
	desiredStatus.LoadBalancer.Ingress = append(desiredStatus.LoadBalancer.Ingress, v1.LoadBalancerIngress{IP: lbIngress2})
	desiredService = &v1.Service{Status: desiredStatus}

	c2ServiceWatch.Modify(service)
	require.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster2.Name,
		key, service, wait.ForeverTestTimeout))
	require.NoError(t, WaitForFederatedServiceUpdate(t, sc.serviceStore,
		key, desiredService, serviceStatusCompare, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster2 endpoint for the service is created")
	desiredIngressAnnotation = ingress.NewFederatedServiceIngress().
		AddEndpoints("cluster1", []string{lbIngress1}).
		AddEndpoints("cluster2", []string{lbIngress2}).
		String()
	desiredService = &v1.Service{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{ingress.FederatedServiceIngressAnnotation: desiredIngressAnnotation}}}
	c2EndpointWatch.Add(NewEndpoint("test-service-1", serviceEndpoint2))
	require.NoError(t, WaitForFederatedServiceUpdate(t, sc.serviceStore,
		key, desiredService, serviceIngressCompare, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster1 endpoint for the service is deleted")
	desiredIngressAnnotation = ingress.NewFederatedServiceIngress().
		AddEndpoints("cluster2", []string{lbIngress2}).
		String()
	desiredService = &v1.Service{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{ingress.FederatedServiceIngressAnnotation: desiredIngressAnnotation}}}
	c1EndpointWatch.Delete(NewEndpoint("test-service-1", serviceEndpoint1))
	require.NoError(t, WaitForFederatedServiceUpdate(t, sc.serviceStore,
		key, desiredService, serviceIngressCompare, wait.ForeverTestTimeout))

	glog.Infof("Test modifying federated service by changing the port")
	service.Spec.Ports[0].Port = 9090
	fedServiceWatch.Modify(service)
	require.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster1.Name,
		key, service, wait.ForeverTestTimeout))

	glog.Infof("Test cluster service is recreated when deleted")
	c1Service := NewService("test-service-1", 80)
	c1Service.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	c1ServiceWatch.Delete(c1Service)
	require.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster1.Name,
		key, service, wait.ForeverTestTimeout))

	glog.Infof("Test cluster services are deleted when federated service is deleted")
	service.ObjectMeta.Finalizers = append(service.ObjectMeta.Finalizers, deletionhelper.FinalizerDeleteFromUnderlyingClusters)
	service.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	fedServiceWatch.Modify(service)
	require.NoError(t, WaitForClusterServiceDelete(t, sc.federatedInformer.GetTargetStore(), cluster1.Name,
		key, wait.ForeverTestTimeout))
	require.NoError(t, WaitForClusterServiceDelete(t, sc.federatedInformer.GetTargetStore(), cluster2.Name,
		key, wait.ForeverTestTimeout))

	close(stop)
}

func TestGetOperationsToPerformOnCluster(t *testing.T) {
	obj := NewService("test-service-1", 80)
	cluster1 := NewCluster("cluster1", v1.ConditionTrue)
	fedClient := &fakefedclientset.Clientset{}
	sc := New(fedClient)

	testCases := map[string]struct {
		expectedSendErr bool
		sendToCluster   bool
		operationType   fedutil.FederatedOperationType
	}{
		"sendToCluster error returned": {
			expectedSendErr: true,
		},
		"Missing object and not matching ClusterSelector should result in no operations": {
			sendToCluster: false,
		},
		"Missing object and matching ClusterSelector should result in add operation": {
			operationType: fedutil.OperationTypeAdd,
			sendToCluster: true,
		},
		// Update and Delete scenarios are tested in TestServiceController
	}
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {

			operations, err := getOperationsToPerformOnCluster(sc.federatedInformer, cluster1, obj, func(map[string]string, map[string]string) (bool, error) {
				if testCase.expectedSendErr {
					return false, awfulError
				}
				return testCase.sendToCluster, nil
			})
			if testCase.expectedSendErr {
				require.Error(t, err, "An error was expected")
			} else {
				require.NoError(t, err, "An error was not expected")
			}
			if len(testCase.operationType) == 0 {
				require.Nil(t, operations, "An operation was not expected")
			} else {
				require.NotNil(t, operations, "A single operation was expected")
				require.Equal(t, testCase.operationType, operations.Type, "Unexpected operation returned")
			}
		})
	}
}

// serviceObjectGetter gives dummy service objects to use with RegisterFakeOnDelete
// This is just so that federated informer can be tested for delete scenarios.
func serviceObjectGetter(name, namespace string) runtime.Object {
	service := new(v1.Service)
	service.Namespace = namespace
	service.Name = name
	return service
}

func NewService(name string, port int32) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/services/" + name,
			Labels:    map[string]string{"app": name},
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{Port: port}},
			Type:  v1.ServiceTypeLoadBalancer,
		},
	}
}

func NewEndpoint(name, ip string) *v1.Endpoints {
	return &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: v1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/endpoints/" + name,
			Labels:    map[string]string{"app": name},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: []v1.EndpointAddress{{
				IP: ip,
			}}},
		},
	}
}

// WaitForClusterService waits for the cluster service to be created matching the desiredService.
func WaitForClusterService(t *testing.T, store fedutil.FederatedReadOnlyStore, clusterName, key string, desiredService *v1.Service, timeout time.Duration) error {
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		obj, found, err := store.GetByKey(clusterName, key)
		if !found || err != nil {
			return false, err
		}
		service := obj.(*v1.Service)
		if !Equivalent(service, desiredService) {
			glog.V(5).Infof("Waiting for clustered service, Desired: %v, Current: %v", desiredService, service)
			return false, nil
		}
		glog.V(5).Infof("Clustered service is up to date: %v", service)
		return true, nil
	})
	return err
}

// WaitForClusterServiceDelete waits for the cluster service to be deleted.
func WaitForClusterServiceDelete(t *testing.T, store fedutil.FederatedReadOnlyStore, clusterName, key string, timeout time.Duration) error {
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		_, found, _ := store.GetByKey(clusterName, key)
		if !found {
			return true, nil
		}
		return false, nil
	})
	return err
}

type serviceCompare func(current, desired *v1.Service) (match bool)

func serviceStatusCompare(current, desired *v1.Service) bool {
	if !reflect.DeepEqual(current.Status.LoadBalancer, desired.Status.LoadBalancer) {
		glog.V(5).Infof("Waiting for loadbalancer status, Current: %v, Desired: %v", current.Status.LoadBalancer, desired.Status.LoadBalancer)
		return false
	}
	glog.V(5).Infof("Loadbalancer status match: %v", current.Status.LoadBalancer)
	return true
}

func serviceIngressCompare(current, desired *v1.Service) bool {
	if strings.Compare(current.Annotations[ingress.FederatedServiceIngressAnnotation], desired.Annotations[ingress.FederatedServiceIngressAnnotation]) != 0 {
		glog.V(5).Infof("Waiting for loadbalancer ingress, Current: %v, Desired: %v", current.Annotations[ingress.FederatedServiceIngressAnnotation], desired.Annotations[ingress.FederatedServiceIngressAnnotation])
		return false
	}
	glog.V(5).Infof("Loadbalancer ingress match: %v", current.Annotations[ingress.FederatedServiceIngressAnnotation])
	return true
}

// WaitForFederatedServiceUpdate waits for federated service updates to match the desiredService.
func WaitForFederatedServiceUpdate(t *testing.T, store corelisters.ServiceLister, key string, desiredService *v1.Service, match serviceCompare, timeout time.Duration) error {
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			return false, err
		}
		service, err := store.Services(namespace).Get(name)
		switch {
		case errors.IsNotFound(err):
			return false, nil
		case err != nil:
			return false, err
		case !match(service, desiredService):
			return false, nil
		default:
			return true, nil
		}
	})
	return err
}
