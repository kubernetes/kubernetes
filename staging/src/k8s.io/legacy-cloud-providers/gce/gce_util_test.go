// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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

package gce

import (
	"context"
	"net"
	"reflect"
	"testing"

	compute "google.golang.org/api/compute/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestLastIPInRange(t *testing.T) {
	for _, tc := range []struct {
		cidr string
		want string
	}{
		{"10.1.2.3/32", "10.1.2.3"},
		{"10.1.2.0/31", "10.1.2.1"},
		{"10.1.0.0/30", "10.1.0.3"},
		{"10.0.0.0/29", "10.0.0.7"},
		{"::0/128", "::"},
		{"::0/127", "::1"},
		{"::0/126", "::3"},
		{"::0/120", "::ff"},
	} {
		_, c, err := net.ParseCIDR(tc.cidr)
		if err != nil {
			t.Errorf("net.ParseCIDR(%v) = _, %v, %v; want nil", tc.cidr, c, err)
			continue
		}

		if lastIP := lastIPInRange(c); lastIP.String() != tc.want {
			t.Errorf("LastIPInRange(%v) = %v; want %v", tc.cidr, lastIP, tc.want)
		}
	}
}

func TestSubnetsInCIDR(t *testing.T) {
	subnets := []*compute.Subnetwork{
		{
			Name:        "A",
			IpCidrRange: "10.0.0.0/20",
		},
		{
			Name:        "B",
			IpCidrRange: "10.0.16.0/20",
		},
		{
			Name:        "C",
			IpCidrRange: "10.132.0.0/20",
		},
		{
			Name:        "D",
			IpCidrRange: "10.0.32.0/20",
		},
		{
			Name:        "E",
			IpCidrRange: "10.134.0.0/20",
		},
	}
	expectedNames := []string{"C", "E"}

	gotSubs, err := subnetsInCIDR(subnets, autoSubnetIPRange)
	if err != nil {
		t.Errorf("autoSubnetInList() = _, %v", err)
	}

	var gotNames []string
	for _, v := range gotSubs {
		gotNames = append(gotNames, v.Name)
	}
	if !reflect.DeepEqual(gotNames, expectedNames) {
		t.Errorf("autoSubnetInList() = %v, expected: %v", gotNames, expectedNames)
	}
}

func TestFirewallToGcloudArgs(t *testing.T) {
	firewall := compute.Firewall{
		Description:  "Last Line of Defense",
		TargetTags:   []string{"jock-nodes", "band-nodes"},
		SourceRanges: []string{"3.3.3.3/20", "1.1.1.1/20", "2.2.2.2/20"},
		Allowed: []*compute.FirewallAllowed{
			{
				IPProtocol: "udp",
				Ports:      []string{"321", "123-456", "123"},
			},
			{
				IPProtocol: "tcp",
				Ports:      []string{"321", "123-456", "123"},
			},
			{
				IPProtocol: "sctp",
				Ports:      []string{"321", "123-456", "123"},
			},
		},
	}
	got := firewallToGcloudArgs(&firewall, "my-project")

	var e = `--description "Last Line of Defense" --allow sctp:123,sctp:123-456,sctp:321,tcp:123,tcp:123-456,tcp:321,udp:123,udp:123-456,udp:321 --source-ranges 1.1.1.1/20,2.2.2.2/20,3.3.3.3/20 --target-tags band-nodes,jock-nodes --project my-project`
	if got != e {
		t.Errorf("%q does not equal %q", got, e)
	}
}

// TestAddRemoveFinalizer tests the add/remove and hasFinalizer methods.
func TestAddRemoveFinalizer(t *testing.T) {
	svc := fakeLoadbalancerService(string(LBTypeInternal))
	gce, err := fakeGCECloud(vals)
	if err != nil {
		t.Fatalf("Failed to get GCE client, err %v", err)
	}
	svc, err = gce.client.CoreV1().Services(svc.Namespace).Create(context.TODO(), svc, metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Failed to create service %s, err %v", svc.Name, err)
	}

	err = addFinalizer(svc, gce.client.CoreV1(), ILBFinalizerV1)
	if err != nil {
		t.Fatalf("Failed to add finalizer, err %v", err)
	}
	svc, err = gce.client.CoreV1().Services(svc.Namespace).Get(context.TODO(), svc.Name, metav1.GetOptions{})
	if err != nil {
		t.Errorf("Failed to get service, err %v", err)
	}
	if !hasFinalizer(svc, ILBFinalizerV1) {
		t.Errorf("Unable to find finalizer '%s' in service %s", ILBFinalizerV1, svc.Name)
	}
	err = removeFinalizer(svc, gce.client.CoreV1(), ILBFinalizerV1)
	if err != nil {
		t.Fatalf("Failed to remove finalizer, err %v", err)
	}
	svc, err = gce.client.CoreV1().Services(svc.Namespace).Get(context.TODO(), svc.Name, metav1.GetOptions{})
	if err != nil {
		t.Errorf("Failed to get service, err %v", err)
	}
	if hasFinalizer(svc, ILBFinalizerV1) {
		t.Errorf("Failed to remove finalizer '%s' in service %s", ILBFinalizerV1, svc.Name)
	}
}
