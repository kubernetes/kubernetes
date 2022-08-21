/*
Copyright 2022 The Kubernetes Authors.

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

package servicecidrs

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingapiv1alpha1 "k8s.io/api/networking/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller"
)

type testController struct {
	*Controller
	servicesStore     cache.Store
	servicecidrsStore cache.Store
	ipaddressesStore  cache.Store
}

func newController() (*fake.Clientset, *testController) {
	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())

	serviceInformer := informerFactory.Core().V1().Services()
	serviceCIDRInformer := informerFactory.Networking().V1alpha1().ServiceCIDRs()
	ipAddressInformer := informerFactory.Networking().V1alpha1().IPAddresses()
	controller := NewController(
		serviceInformer,
		serviceCIDRInformer,
		ipAddressInformer,
		client)

	var alwaysReady = func() bool { return true }
	controller.servicesSynced = alwaysReady
	controller.serviceCIDRsSynced = alwaysReady
	controller.ipAddressSynced = alwaysReady

	return client, &testController{
		controller,
		serviceInformer.Informer().GetStore(),
		serviceCIDRInformer.Informer().GetStore(),
		ipAddressInformer.Informer().GetStore(),
	}
}

func TestController_enqueue(t *testing.T) {

	_, controller := newController()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go controller.Run(1, ctx.Done())
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
		Spec: v1.ServiceSpec{
			Selector:   map[string]string{"foo": "bar"},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
			Ports:      []v1.ServicePort{{Port: 80}},
		},
	}
	controller.enqueue(svc)
	time.Sleep(1 * time.Second)

}

func TestControllerServiceCIDRAddFinalizer(t *testing.T) {
	client, controller := newController()
	serviceCIDR := &networkingapiv1alpha1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: "cidr-nofinalizer",
			Labels: map[string]string{
				networkingapiv1alpha1.LabelServiceCIDRFromFlags: "true",
			},
		},
		Spec: networkingapiv1alpha1.ServiceCIDRSpec{
			IPv4: "192.168.0.0/24",
			IPv6: "2001:db2::/64",
		},
	}

	controller.servicecidrsStore.Add(serviceCIDR)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go controller.Run(1, ctx.Done())
	_, err := client.NetworkingV1alpha1().ServiceCIDRs().Create(ctx, serviceCIDR, metav1.CreateOptions{})
	if err != nil {
		t.Fatal((err))
	}
	controller.addServiceCIDR(serviceCIDR)

	// wait until it adds the finalizer
	err = wait.PollImmediate(100*time.Millisecond, 5*time.Second, func() (bool, error) {
		cidr, err := client.NetworkingV1alpha1().ServiceCIDRs().Get(ctx, serviceCIDR.Name, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		if needToAddFinalizer(cidr, networkingapiv1alpha1.ServiceCIDRProtectionFinalizer) {
			return false, nil
		}
		return true, nil

	})
	if err != nil {
		t.Fatal((err))
	}
}

func Test_needToAddFinalizer(t *testing.T) {
	now := metav1.Now()
	tests := []struct {
		name      string
		obj       metav1.Object
		finalizer string
		want      bool
	}{
		{
			name: "service without finalizer",
			obj: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
			},
			finalizer: "finalizer",
			want:      true,
		},
		{
			name: "service with finalizer",
			obj: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "test",
					Finalizers: []string{"finalizer"},
				},
			},
			finalizer: "finalizer",
			want:      false,
		},
		{
			name: "service being deleted",
			obj: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: &now,
					Name:              "test",
					Finalizers:        []string{"finalizer"},
				},
			},
			finalizer: "finalizer",
			want:      false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := needToAddFinalizer(tt.obj, tt.finalizer); got != tt.want {
				t.Errorf("needToAddFinalizer() = %v, want %v", got, tt.want)
			}
		})
	}
}
