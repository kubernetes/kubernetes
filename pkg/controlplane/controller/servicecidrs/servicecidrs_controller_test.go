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
	"fmt"
	"testing"
	"time"

	networkingapiv1alpha1 "k8s.io/api/networking/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	netutils "k8s.io/utils/net"
)

var alwaysReady = func() bool { return true }

const (
	defaultIPv4CIDR = "10.16.0.0/16"
	defaultIPv6CIDR = "2001:db8::/64"
)

func newController() (*fake.Clientset, *Controller) {
	client := fake.NewSimpleClientset()
	_, primaryRange, err := netutils.ParseCIDRSloppy(defaultIPv4CIDR)
	if err != nil {
		panic(err)
	}

	_, secondaryRange, err := netutils.ParseCIDRSloppy(defaultIPv6CIDR)
	if err != nil {
		panic(err)
	}

	controller := NewController(
		*primaryRange,
		*secondaryRange,
		netutils.ParseIPSloppy("192.168.0.1"),
		443,
		6443,
		0,
		nil,
		client)

	controller.servicesSynced = controller.serviceInformer.GetController().HasSynced
	controller.serviceCIDRsSynced = controller.serviceCIDRInformer.GetController().HasSynced

	return client, controller

}

// Bootstrap test cases
// Single apiserver:
// - no serviceCIDR -> create serviceCIDR from flags
// - existing serviceCIDR match flags -> noop
// - ServiceCIDR contains flags  -> noop
// - ServiceCIDR is contained within flags
// - ServiceDIR does not match flags
func TestController_FromFlags(t *testing.T) {
	// logs.GlogSetter("4")
	type cidr struct {
		v4 string
		v6 string
	}
	testCases := []struct {
		name         string
		cidrs        []cidr
		serviceCIDRs int
		actions      int
	}{
		{
			name:         "no existing service CIDRs",
			serviceCIDRs: 1,
			actions:      6, // 2xList , 2xWatch, 2xCreate
		},
		{
			name:         "existing service CIDRs match flags",
			cidrs:        []cidr{{defaultIPv4CIDR, defaultIPv6CIDR}},
			serviceCIDRs: 1,
			actions:      6, // 1xCreate (test) 2xList , 2xWatch, 1xCreate
		},
		{
			name:         "existing service CIDRs contain flags",
			cidrs:        []cidr{{"10.0.0.0/8", "2001:db8::/48"}},
			serviceCIDRs: 1,
			actions:      6, // 1xCreate (test) 2xList , 2xWatch, 1xCreate
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client, controller := newController()
			for i, cidr := range tc.cidrs {
				s := &networkingapiv1alpha1.ServiceCIDR{
					ObjectMeta: metav1.ObjectMeta{
						Name: fmt.Sprintf("kubernetes-default-existing-%d", i),
						Labels: map[string]string{
							networkingapiv1alpha1.LabelServiceCIDRFromFlags: "true",
						},
						Finalizers: []string{networkingapiv1alpha1.ServiceCIDRProtectionFinalizer},
					},
					Spec: networkingapiv1alpha1.ServiceCIDRSpec{
						IPv4: cidr.v4,
						IPv6: cidr.v6,
					},
				}
				err := controller.serviceCIDRInformer.GetStore().Add(s)
				if err != nil {
					t.Fatalf("Expected no error adding ServiceCIDR: %v", err)
				}
				_, err = client.NetworkingV1alpha1().ServiceCIDRs().Create(context.TODO(), s, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Expected no error creating EndpointSlice: %v", err)
				}
			}

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			go controller.Run(ctx.Done())

			t.Log("wait until bootstrap happens")

			time.Sleep(1 * time.Second)
			if len(client.Actions()) != tc.actions {
				t.Errorf("Expected %d got %d", tc.actions, len(client.Actions()))
			}

			cidrs, err := client.NetworkingV1alpha1().ServiceCIDRs().List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if len(cidrs.Items) != tc.serviceCIDRs {
				t.Errorf("Expected %d got %d", tc.serviceCIDRs, len(cidrs.Items))
			}

			svc, err := client.CoreV1().Services("default").Get(ctx, "kubernetes", metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if svc.Spec.ClusterIP != "10.16.0.1" {
				t.Errorf("Expected IP 10.16.0.1 Received IP %s", svc.Spec.ClusterIP)
			}
		})
	}
}

func TestController_DefaultService(t *testing.T) {

	client, controller := newController()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go controller.Run(ctx.Done())
	time.Sleep(1 * time.Second)
	svc, err := client.CoreV1().Services("default").Get(ctx, "kubernetes", metav1.GetOptions{})
	if err != nil {
		t.Fatal("err")
	}
	if svc.Spec.ClusterIP != "10.16.0.1" {
		t.Errorf("Expected IP 10.16.0.1 Received IP %s", svc.Spec.ClusterIP)
	}

}

func Test_subnetContains(t *testing.T) {
	tests := []struct {
		name string
		a    string
		b    string
		want bool
	}{
		{
			name: "same IPv4",
			a:    "192.168.0.0/24",
			b:    "192.168.0.0/24",
			want: true,
		},
		{
			name: "contains IPv4",
			a:    "192.168.0.0/23",
			b:    "192.168.0.0/24",
			want: true,
		},
		{
			name: "contains IPv4 large",
			a:    "192.168.0.0/2",
			b:    "222.168.0.0/24",
			want: true,
		},
		{
			name: "contained IPv4",
			a:    "192.168.1.0/25",
			b:    "192.168.1.0/24",
			want: false,
		},
		{
			name: "not contains IPv4",
			a:    "10.1.1.0/24",
			b:    "192.168.1.0/24",
			want: false,
		},
		{
			name: "same IPv4",
			a:    "2001:db2::/48",
			b:    "2001:db2::/48",
			want: true,
		},
		{
			name: "contains IPv4",
			a:    "2001:db2::/48",
			b:    "2001:db2::/64",
			want: true,
		},
		{
			name: "contained IPv4",
			a:    "2001:db2::/65",
			b:    "2001:db2::/64",
			want: false,
		},
		{
			name: "not contains IPv4",
			a:    "fd00:1::/64",
			b:    "2001:db2::/64",
			want: false,
		},
		{
			name: "different family",
			a:    "10.1.1.0/24",
			b:    "2001:db2::/64",
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, a, err := netutils.ParseCIDRSloppy(tt.a)
			if err != nil {
				t.Fatal(err)
			}
			_, b, err := netutils.ParseCIDRSloppy(tt.b)
			if err != nil {
				t.Fatal(err)
			}
			if got := subnetContains(a, b); got != tt.want {
				t.Errorf("subnetContains() = %v, want %v", got, tt.want)
			}
		})
	}
}
