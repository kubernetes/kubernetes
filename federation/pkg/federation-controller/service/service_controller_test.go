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
	"testing"
	"time"

	"github.com/golang/glog"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fakefedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	si "k8s.io/kubernetes/federation/pkg/federation-controller/service/ingress"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"
	"k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	fakekubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/client/legacylisters"
)

const (
	retryInterval = 100 * time.Millisecond
)

func TestServiceController(t *testing.T) {
	glog.Infof("Creating fake infrastructure")
	fedClient := &fakefedclientset.Clientset{}
	RegisterFakeList("clusters", &fedClient.Fake, &v1beta1.ClusterList{Items: []v1beta1.Cluster{}})
	fedclusterWatch := RegisterFakeWatch("clusters", &fedClient.Fake)
	RegisterFakeList("services", &fedClient.Fake, &v1.ServiceList{Items: []v1.Service{}})
	fedServiceWatch := RegisterFakeWatch("services", &fedClient.Fake)
	RegisterFakeOnCreate("clusters", &fedClient.Fake, fedclusterWatch)
	RegisterFakeOnUpdate("clusters", &fedClient.Fake, fedclusterWatch)
	RegisterFakeOnCreate("services", &fedClient.Fake, fedServiceWatch)
	RegisterFakeOnUpdate("services", &fedClient.Fake, fedServiceWatch)

	cluster1Client := &fakekubeclientset.Clientset{}
	RegisterFakeList("services", &cluster1Client.Fake, &v1.ServiceList{Items: []v1.Service{}})
	c1ServiceWatch := RegisterFakeWatch("services", &cluster1Client.Fake)
	RegisterFakeList("endpoints", &cluster1Client.Fake, &v1.EndpointsList{Items: []v1.Endpoints{}})
	c1EndpointWatch := RegisterFakeWatch("endpoints", &cluster1Client.Fake)
	RegisterFakeOnCreate("services", &cluster1Client.Fake, c1ServiceWatch)
	RegisterFakeOnUpdate("services", &cluster1Client.Fake, c1ServiceWatch)
	RegisterFakeOnCreate("endpoints", &cluster1Client.Fake, c1EndpointWatch)
	RegisterFakeOnUpdate("endpoints", &cluster1Client.Fake, c1EndpointWatch)

	cluster2Client := &fakekubeclientset.Clientset{}
	RegisterFakeList("services", &cluster2Client.Fake, &v1.ServiceList{Items: []v1.Service{}})
	c2ServiceWatch := RegisterFakeWatch("services", &cluster2Client.Fake)
	RegisterFakeList("endpoints", &cluster2Client.Fake, &v1.EndpointsList{Items: []v1.Endpoints{}})
	c2EndpointWatch := RegisterFakeWatch("endpoints", &cluster2Client.Fake)
	RegisterFakeOnCreate("services", &cluster2Client.Fake, c2ServiceWatch)
	RegisterFakeOnUpdate("services", &cluster2Client.Fake, c2ServiceWatch)
	RegisterFakeOnCreate("endpoints", &cluster2Client.Fake, c2EndpointWatch)
	RegisterFakeOnUpdate("endpoints", &cluster2Client.Fake, c2EndpointWatch)

	cluster1 := NewClusterWithRegionZone("cluster1", v1.ConditionTrue, "region1", "zone1")
	cluster2 := NewClusterWithRegionZone("cluster2", v1.ConditionTrue, "region2", "zone2")

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
	sc.smallDelay = 20 * time.Millisecond
	sc.updateTimeout = 5 * time.Second

	stop := make(chan struct{})
	glog.Infof("Running Service Controller")
	sc.Run(5, stop)

	glog.Infof("Adding cluster 1")
	fedclusterWatch.Add(cluster1)

	service := NewService("test-service-1", 80)

	// Test add federated service.
	glog.Infof("Adding federated service")
	fedServiceWatch.Add(service)
	key := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}.String()

	glog.Infof("Test service was correctly created in cluster 1")
	assert.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster1.Name,
		key, service, wait.ForeverTestTimeout))

	glog.Infof("Adding cluster 2")
	fedclusterWatch.Add(cluster2)

	glog.Infof("Test service was correctly created in cluster 2")
	assert.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster2.Name,
		key, service, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster1 service status is updated")
	service.Status = v1.ServiceStatus{
		LoadBalancer: v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{IP: "10.20.30.40"},
			}}}

	desiredStatus := service.Status

	c1ServiceWatch.Modify(service)
	assert.NoError(t, WaitForFedServiceStatusUpdate(t, sc.serviceStore,
		key, desiredStatus.LoadBalancer, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster1 endpoint for the service is created")
	expectedIngressAnnotation := si.NewServiceIngressAnnotation(si.NewServiceIngress().
		AddEndpoints("region1", "zone1", "cluster1", []string{"10.20.30.40"}, true).
		GetJSONMarshalledBytes())
	c1EndpointWatch.Add(NewEndpoint("test-service-1", "192.168.0.1"))
	assert.NoError(t, WaitForFedServiceIngressUpdate(t, sc.serviceStore,
		key, expectedIngressAnnotation, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster2 service status is updated")
	service.Status = v1.ServiceStatus{
		LoadBalancer: v1.LoadBalancerStatus{
			Ingress: []v1.LoadBalancerIngress{
				{IP: "10.20.30.50"},
			}}}
	desiredStatus.LoadBalancer.Ingress = append(desiredStatus.LoadBalancer.Ingress, v1.LoadBalancerIngress{IP: "10.20.30.50"})

	c2ServiceWatch.Modify(service)
	assert.NoError(t, WaitForFedServiceStatusUpdate(t, sc.serviceStore,
		key, desiredStatus.LoadBalancer, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster2 endpoint for the service is created")
	expectedIngressAnnotation = si.NewServiceIngressAnnotation(si.NewServiceIngress().
		AddEndpoints("region1", "zone1", "cluster1", []string{"10.20.30.40"}, true).
		AddEndpoints("region2", "zone2", "cluster2", []string{"10.20.30.50"}, true).
		GetJSONMarshalledBytes())
	c2EndpointWatch.Add(NewEndpoint("test-service-1", "192.168.1.1"))
	assert.NoError(t, WaitForFedServiceIngressUpdate(t, sc.serviceStore,
		key, expectedIngressAnnotation, wait.ForeverTestTimeout))

	glog.Infof("Test federation service is updated when cluster1 endpoint for the service is deleted")
	expectedIngressAnnotation = si.NewServiceIngressAnnotation(si.NewServiceIngress().
		AddEndpoints("region1", "zone1", "cluster1", []string{"10.20.30.40"}, false).
		AddEndpoints("region2", "zone2", "cluster2", []string{"10.20.30.50"}, true).
		GetJSONMarshalledBytes())
	c1EndpointWatch.Delete(NewEndpoint("test-service-1", "192.168.0.1"))
	assert.NoError(t, WaitForFedServiceIngressUpdate(t, sc.serviceStore,
		key, expectedIngressAnnotation, wait.ForeverTestTimeout))

	// Test update federated service.
	glog.Infof("Test modifying federated service by changing the port")
	service.Spec.Ports[0].Port = 9090
	fedServiceWatch.Modify(service)
	assert.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster1.Name,
		key, service, wait.ForeverTestTimeout))

	// Test cluster service is recreated when deleted.
	glog.Infof("Test cluster service is recreated when deleted")
	c1ServiceWatch.Delete(service)
	assert.NoError(t, WaitForClusterService(t, sc.federatedInformer.GetTargetStore(), cluster1.Name,
		key, service, wait.ForeverTestTimeout))

	close(stop)
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

// NewCluster builds a new cluster object.
func NewClusterWithRegionZone(name string, readyStatus v1.ConditionStatus, region, zone string) *v1beta1.Cluster {
	return &v1beta1.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{},
		},
		Status: v1beta1.ClusterStatus{
			Conditions: []v1beta1.ClusterCondition{
				{Type: v1beta1.ClusterReady, Status: readyStatus},
			},
			Region: region,
			Zones:  []string{zone},
		},
	}
}

// Wait for federation service status to be updated to match the desiredStatus.
func WaitForClusterService(t *testing.T, store fedutil.FederatedReadOnlyStore, clusterName, key string, desiredService *v1.Service, timeout time.Duration) error {
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		obj, found, err := store.GetByKey(clusterName, key)
		if !found || err != nil {
			return false, err
		}
		service := obj.(*v1.Service)
		if !Equivalent(service, desiredService) {
			return false, nil
		}
		return true, nil
	})
	return err
}

// Wait for federation service status to be updated to match the desiredStatus.
func WaitForFedServiceStatusUpdate(t *testing.T, store listers.StoreToServiceLister, key string, desiredStatus v1.LoadBalancerStatus, timeout time.Duration) error {
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		obj, found, err := store.Indexer.GetByKey(key)
		if !found || err != nil {
			return false, err
		}
		service := obj.(*v1.Service)
		if !reflect.DeepEqual(service.Status.LoadBalancer, desiredStatus) {
			glog.V(5).Infof("DesiredStatus: %v, ExistingStatus: %v", desiredStatus, service.Status.LoadBalancer)
			return false, nil
		}
		return true, nil
	})
	return err
}

// Wait for federation service ingress to be updated to match the desiredIngress.
func WaitForFedServiceIngressUpdate(t *testing.T, store listers.StoreToServiceLister, key string, desiredIngress map[string]string, timeout time.Duration) error {
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		obj, found, err := store.Indexer.GetByKey(key)
		if !found || err != nil {
			return false, err
		}
		service := obj.(*v1.Service)
		if !reflect.DeepEqual(service.Annotations, desiredIngress) {
			glog.V(5).Infof("DesiredIngress: %v, ExistingIngress: %v", desiredIngress, service.Annotations)
			return false, nil
		}
		return true, nil
	})
	return err
}
