/*
Copyright 2024 The Kubernetes Authors.

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

package servicecidr

import (
	"net/netip"
	"reflect"
	"sort"
	"testing"

	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	networkinglisters "k8s.io/client-go/listers/networking/v1alpha1"
	"k8s.io/client-go/tools/cache"
	netutils "k8s.io/utils/net"
)

func newServiceCIDR(name, primary, secondary string) *networkingv1alpha1.ServiceCIDR {
	serviceCIDR := &networkingv1alpha1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1alpha1.ServiceCIDRSpec{},
	}
	serviceCIDR.Spec.CIDRs = append(serviceCIDR.Spec.CIDRs, primary)
	if secondary != "" {
		serviceCIDR.Spec.CIDRs = append(serviceCIDR.Spec.CIDRs, secondary)
	}
	return serviceCIDR
}

func TestOverlapsPrefix(t *testing.T) {
	tests := []struct {
		name         string
		serviceCIDRs []*networkingv1alpha1.ServiceCIDR
		prefix       netip.Prefix
		want         []string
	}{
		{
			name: "only one ServiceCIDR and IPv4 prefix contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/26"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and same IPv4 prefix",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/24"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and larger IPv4 prefix",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/16"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and non contained IPv4 prefix",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			want:   []string{},
		},
		{
			name: "only one ServiceCIDR and IPv6 prefix contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/112"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and same IPv6 prefix",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/96"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and IPv6 larger",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/64"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and IPv6 prefix out of range",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("2001:db2::/112"),
			want:   []string{},
		},
		{
			name: "two ServiceCIDR and IPv4 prefix contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/24"),
			want:   []string{"kubernetes", "secondary"},
		},
		{
			name: "two overlapping ServiceCIDR and IPv4 prefix only contained in one",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/18"),
			want:   []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and IPv4 larger",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/8"),
			want:   []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and IPv4 prefix not contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			want:   []string{},
		},
		{
			name: "two ServiceCIDR and IPv6 prefix contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/96"),
			want:   []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and IPv6 prefix contained in one",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/72"),
			want:   []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and aprefix larger",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/52"),
			want:   []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and prefix out of range",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("2001:db2::/64"),
			want:   []string{},
		},
		{
			name: "multiple ServiceCIDR match with overlap contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("kubernetes2", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/28"),
			want:   []string{"kubernetes", "kubernetes2", "secondary"},
		},
		{
			name: "multiple ServiceCIDR match with overlap contains",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("kubernetes2", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/8"),
			want:   []string{"kubernetes", "kubernetes2", "secondary"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			for _, serviceCIDR := range tt.serviceCIDRs {
				err := indexer.Add(serviceCIDR)
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
			}
			lister := networkinglisters.NewServiceCIDRLister(indexer)
			got := []string{}
			for _, serviceCIDR := range OverlapsPrefix(lister, tt.prefix) {
				got = append(got, serviceCIDR.Name)
			}
			// sort slices to make the order predictable and avoid flakiness
			sort.Strings(got)
			sort.Strings(tt.want)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("OverlapsAddress() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestContainsPrefix(t *testing.T) {
	tests := []struct {
		name         string
		serviceCIDRs []*networkingv1alpha1.ServiceCIDR
		prefix       netip.Prefix
		want         []string
	}{
		{
			name: "only one ServiceCIDR and IPv4 prefix contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/26"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and same IPv4 prefix",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/24"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and larger IPv4 prefix",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/16"),
			want:   []string{},
		},
		{
			name: "only one ServiceCIDR and non containerd IPv4 prefix",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			want:   []string{},
		},
		{
			name: "only one ServiceCIDR and IPv6 prefix contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/112"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and same IPv6 prefix",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/96"),
			want:   []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and IPv6 larger",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/64"),
			want:   []string{},
		},
		{
			name: "only one ServiceCIDR and IPv6 prefix out of range",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			prefix: netip.MustParsePrefix("2001:db2::/112"),
			want:   []string{},
		},
		{
			name: "two ServiceCIDR and IPv4 prefix contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/24"),
			want:   []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and IPv4 prefix only contained in one",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/18"),
			want:   []string{"secondary"},
		},
		{
			name: "two ServiceCIDR and IPv4 larger",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/8"),
			want:   []string{},
		},
		{
			name: "two ServiceCIDR and IPv4 prefix not contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			want:   []string{},
		},
		{
			name: "two ServiceCIDR and IPv6 prefix contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/96"),
			want:   []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and IPv6 prefix contained in one",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/72"),
			want:   []string{"secondary"},
		},
		{
			name: "two ServiceCIDR and aprefix larger",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("2001:db8::/52"),
			want:   []string{},
		},
		{
			name: "two ServiceCIDR and prefix out of range",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("2001:db2::/64"),
			want:   []string{},
		},
		{
			name: "multiple ServiceCIDR match with overlap",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("kubernetes2", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			prefix: netip.MustParsePrefix("10.0.0.0/28"),
			want:   []string{"kubernetes", "kubernetes2", "secondary"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			for _, serviceCIDR := range tt.serviceCIDRs {
				err := indexer.Add(serviceCIDR)
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
			}
			lister := networkinglisters.NewServiceCIDRLister(indexer)
			got := []string{}
			for _, serviceCIDR := range ContainsPrefix(lister, tt.prefix) {
				got = append(got, serviceCIDR.Name)
			}
			// sort slices to make the order predictable and avoid flakiness
			sort.Strings(got)
			sort.Strings(tt.want)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ContainsAddress() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestContainsAddress(t *testing.T) {
	tests := []struct {
		name         string
		serviceCIDRs []*networkingv1alpha1.ServiceCIDR
		address      netip.Addr
		want         []string
	}{
		{
			name: "only one ServiceCIDR and IPv4 address contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			address: netip.MustParseAddr("10.0.0.1"),
			want:    []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and IPv4 address broadcast",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			address: netip.MustParseAddr("10.0.0.255"),
			want:    []string{},
		},
		{
			name: "only one ServiceCIDR and IPv4 address base",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			address: netip.MustParseAddr("10.0.0.0"),
			want:    []string{},
		},
		{
			name: "only one ServiceCIDR and IPv4 address out of range",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			address: netip.MustParseAddr("192.0.0.1"),
			want:    []string{},
		},
		{
			name: "only one ServiceCIDR and IPv6 address contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			address: netip.MustParseAddr("2001:db8::2:3"),
			want:    []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and IPv6 address broadcast",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			address: netip.MustParseAddr("2001:db8::ffff:ffff"),
			want:    []string{"kubernetes"},
		},
		{
			name: "only one ServiceCIDR and IPv6 address base",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			address: netip.MustParseAddr("2001:db8::"),
			want:    []string{},
		},
		{
			name: "only one ServiceCIDR and IPv6 address out of range",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
			},
			address: netip.MustParseAddr("2002:1:2:3::2"),
			want:    []string{},
		},
		{
			name: "two ServiceCIDR and IPv4 address contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			address: netip.MustParseAddr("10.0.0.1"),
			want:    []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and IPv4 address broadcast",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			address: netip.MustParseAddr("10.0.0.255"),
			want:    []string{"secondary"},
		},
		{
			name: "two ServiceCIDR and IPv4 address base",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			address: netip.MustParseAddr("10.0.0.0"),
			want:    []string{},
		},
		{
			name: "two ServiceCIDR and IPv4 address out of range",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			address: netip.MustParseAddr("192.0.0.1"),
			want:    []string{},
		},
		{
			name: "two ServiceCIDR and IPv6 address contained",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			address: netip.MustParseAddr("2001:db8::2:3"),
			want:    []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and address broadcast",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			address: netip.MustParseAddr("2001:db8::ffff:ffff"),
			want:    []string{"kubernetes", "secondary"},
		},
		{
			name: "two ServiceCIDR and address base",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			address: netip.MustParseAddr("2001:db8::"),
			want:    []string{},
		},
		{
			name: "two ServiceCIDR and address out of range",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			address: netip.MustParseAddr("2002:1:2:3::2"),
			want:    []string{},
		},
		{
			name: "multiple ServiceCIDR match with overlap",
			serviceCIDRs: []*networkingv1alpha1.ServiceCIDR{
				newServiceCIDR("kubernetes", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("kubernetes2", "10.0.0.0/24", "2001:db8::/96"),
				newServiceCIDR("secondary", "10.0.0.0/16", "2001:db8::/64"),
			},
			address: netip.MustParseAddr("10.0.0.2"),
			want:    []string{"kubernetes", "kubernetes2", "secondary"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			for _, serviceCIDR := range tt.serviceCIDRs {
				err := indexer.Add(serviceCIDR)
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
			}
			lister := networkinglisters.NewServiceCIDRLister(indexer)
			got := []string{}
			for _, serviceCIDR := range ContainsAddress(lister, tt.address) {
				got = append(got, serviceCIDR.Name)
			}
			// sort slices to make the order predictable and avoid flakiness
			sort.Strings(got)
			sort.Strings(tt.want)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ContainsAddress() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_PrefixContainIP(t *testing.T) {
	tests := []struct {
		name   string
		prefix netip.Prefix
		ip     netip.Addr
		want   bool
	}{
		{
			name:   "IPv4 contains",
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			ip:     netip.MustParseAddr("192.168.0.1"),
			want:   true,
		},
		{
			name:   "IPv4 network address",
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			ip:     netip.MustParseAddr("192.168.0.0"),
		},
		{
			name:   "IPv4 broadcast address",
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			ip:     netip.MustParseAddr("192.168.0.255"),
		},
		{
			name:   "IPv4 does not contain",
			prefix: netip.MustParsePrefix("192.168.0.0/24"),
			ip:     netip.MustParseAddr("192.168.1.2"),
		},
		{
			name:   "IPv6 contains",
			prefix: netip.MustParsePrefix("2001:db2::/96"),
			ip:     netip.MustParseAddr("2001:db2::1"),
			want:   true,
		},
		{
			name:   "IPv6 network address",
			prefix: netip.MustParsePrefix("2001:db2::/96"),
			ip:     netip.MustParseAddr("2001:db2::"),
		},
		{
			name:   "IPv6 broadcast address",
			prefix: netip.MustParsePrefix("2001:db2::/96"),
			ip:     netip.MustParseAddr("2001:db2::ffff:ffff"),
			want:   true,
		},
		{
			name:   "IPv6 does not contain",
			prefix: netip.MustParsePrefix("2001:db2::/96"),
			ip:     netip.MustParseAddr("2001:db2:1:2:3::1"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := prefixContainsIP(tt.prefix, tt.ip); got != tt.want {
				t.Errorf("prefixContainIP() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIPToAddr(t *testing.T) {
	tests := []struct {
		name string
		ip   string
		want netip.Addr
	}{
		{
			name: "IPv4",
			ip:   "192.168.2.2",
			want: netip.MustParseAddr("192.168.2.2"),
		},
		{
			name: "IPv6",
			ip:   "2001:db8::2",
			want: netip.MustParseAddr("2001:db8::2"),
		},
		{
			name: "IPv4 in IPv6",
			ip:   "::ffff:192.168.0.1",
			want: netip.MustParseAddr("192.168.0.1"),
		},
		{
			name: "invalid",
			ip:   "invalid_ip",
			want: netip.Addr{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ip := netutils.ParseIPSloppy(tt.ip)
			if got := IPToAddr(ip); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("IPToAddr() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBroadcastAddress(t *testing.T) {
	tests := []struct {
		name    string
		subnet  netip.Prefix
		want    netip.Addr
		wantErr bool
	}{
		{
			name:    "emty subnet",
			wantErr: true,
		},
		{
			name:   "IPv4 even mask",
			subnet: netip.MustParsePrefix("192.168.0.0/24"),
			want:   netip.MustParseAddr("192.168.0.255"),
		},
		{
			name:   "IPv4 odd mask",
			subnet: netip.MustParsePrefix("192.168.0.0/23"),
			want:   netip.MustParseAddr("192.168.1.255"),
		},
		{
			name:   "IPv6 even mask",
			subnet: netip.MustParsePrefix("fd00:1:2:3::/64"),
			want:   netip.MustParseAddr("fd00:1:2:3:ffff:ffff:ffff:ffff"),
		},
		{
			name:   "IPv6 odd mask",
			subnet: netip.MustParsePrefix("fd00:1:2:3::/57"),
			want:   netip.MustParseAddr("fd00:1:2:007f:ffff:ffff:ffff:ffff"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := broadcastAddress(tt.subnet)
			if (err != nil) != tt.wantErr {
				t.Errorf("BroadcastAddress() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BroadcastAddress() = %v, want %v", got, tt.want)
			}
		})
	}
}
