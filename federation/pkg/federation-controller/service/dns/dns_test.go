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

package dns

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns" // Only for unit testing purposes.
	si "k8s.io/kubernetes/federation/pkg/federation-controller/service/ingress"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestServiceController_ensureDnsRecords(t *testing.T) {
	tests := []struct {
		name     string
		service  v1.Service
		expected []string
	}{
		{
			name: "ServiceWithSingleIpEndpoint",
			service: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "servicename",
					Namespace: "servicenamespace",
					Annotations: si.NewServiceIngressAnnotation(si.NewServiceIngress().
						AddEndpoints("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						GetJSONMarshalledBytes()),
				},
			},
			expected: []string{
				"example.com:servicename.servicenamespace.myfederation.svc.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.fooregion.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.foozone.fooregion.federation.example.com:A:180:[198.51.100.1]",
			},
		},
		{
			name: "ServiceWithMultipleIpEndpoints",
			service: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "servicename",
					Namespace: "servicenamespace",
					Annotations: si.NewServiceIngressAnnotation(si.NewServiceIngress().
						AddEndpoints("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						AddEndpoints("barregion", "barzone", "c2", []string{"198.51.200.1"}, true).
						GetJSONMarshalledBytes()),
				},
			},
			expected: []string{
				"example.com:servicename.servicenamespace.myfederation.svc.federation.example.com:A:180:[198.51.100.1 198.51.200.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.fooregion.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.foozone.fooregion.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.barregion.federation.example.com:A:180:[198.51.200.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.barzone.barregion.federation.example.com:A:180:[198.51.200.1]",
			},
		},
		{
			name: "ServiceWithNoEndpoints",
			service: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "servicename",
					Namespace: "servicenamespace",
				},
			},
			expected: []string{},
		},
		{
			name: "ServiceWithEndpointAndServiceDeleted",
			service: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "servicename",
					Namespace:         "servicenamespace",
					DeletionTimestamp: &metav1.Time{Time: time.Now()},
					Annotations: si.NewServiceIngressAnnotation(si.NewServiceIngress().
						AddEndpoints("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						GetJSONMarshalledBytes()),
				},
			},
			expected: []string{},
		},
		{
			name: "ServiceWithMultipleIpEndpointsAndOneEndpointGettingRemoved",
			service: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "servicename",
					Namespace: "servicenamespace",
					Annotations: si.NewServiceIngressAnnotation(si.NewServiceIngress().
						AddEndpoints("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						AddEndpoints("barregion", "barzone", "c2", []string{"198.51.200.1"}, true).
						RemoveEndpoint("barregion", "barzone", "c2", "198.51.200.1").
						GetJSONMarshalledBytes()),
				},
			},
			expected: []string{
				"example.com:servicename.servicenamespace.myfederation.svc.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.fooregion.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.foozone.fooregion.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.barregion.federation.example.com:CNAME:180:[servicename.servicenamespace.myfederation.svc.federation.example.com]",
				"example.com:servicename.servicenamespace.myfederation.svc.barzone.barregion.federation.example.com:CNAME:180:[servicename.servicenamespace.myfederation.svc.barregion.federation.example.com]",
			},
		},
		{
			name: "ServiceWithMultipleIpEndpointsAndOneEndpointIsUnhealthy",
			service: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "servicename",
					Namespace: "servicenamespace",
					Annotations: si.NewServiceIngressAnnotation(si.NewServiceIngress().
						AddEndpoints("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						AddEndpoints("barregion", "barzone", "c2", []string{"198.51.200.1"}, false).
						GetJSONMarshalledBytes()),
				},
			},
			expected: []string{
				"example.com:servicename.servicenamespace.myfederation.svc.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.fooregion.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.foozone.fooregion.federation.example.com:A:180:[198.51.100.1]",
				"example.com:servicename.servicenamespace.myfederation.svc.barregion.federation.example.com:CNAME:180:[servicename.servicenamespace.myfederation.svc.federation.example.com]",
				"example.com:servicename.servicenamespace.myfederation.svc.barzone.barregion.federation.example.com:CNAME:180:[servicename.servicenamespace.myfederation.svc.barregion.federation.example.com]",
			},
		},
		{
			name: "ServiceWithMultipleIpEndpointsAndAllEndpointGettingRemoved",
			service: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "servicename",
					Namespace: "servicenamespace",
					Annotations: si.NewServiceIngressAnnotation(si.NewServiceIngress().
						AddEndpoints("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						AddEndpoints("barregion", "barzone", "c2", []string{"198.51.200.1"}, true).
						RemoveEndpoint("fooregion", "foozone", "c1", "198.51.100.1").
						RemoveEndpoint("barregion", "barzone", "c2", "198.51.200.1").
						GetJSONMarshalledBytes()),
				},
			},
			expected: []string{
				"example.com:servicename.servicenamespace.myfederation.svc.fooregion.federation.example.com:CNAME:180:[servicename.servicenamespace.myfederation.svc.federation.example.com]",
				"example.com:servicename.servicenamespace.myfederation.svc.foozone.fooregion.federation.example.com:CNAME:180:[servicename.servicenamespace.myfederation.svc.fooregion.federation.example.com]",
				"example.com:servicename.servicenamespace.myfederation.svc.barregion.federation.example.com:CNAME:180:[servicename.servicenamespace.myfederation.svc.federation.example.com]",
				"example.com:servicename.servicenamespace.myfederation.svc.barzone.barregion.federation.example.com:CNAME:180:[servicename.servicenamespace.myfederation.svc.barregion.federation.example.com]",
			},
		},
	}
	for _, test := range tests {
		fakedns, _ := clouddns.NewFakeInterface()
		fakednsZones, ok := fakedns.Zones()
		if !ok {
			t.Error("Unable to fetch zones")
		}
		d := ServiceDNSController{
			dns:              fakedns,
			dnsZones:         fakednsZones,
			serviceDnsSuffix: "federation.example.com",
			zoneName:         "example.com",
			federationName:   "myfederation",
		}

		dnsZone, err := getDnsZone(d.zoneName, d.zoneID, d.dnsZones)
		if err != nil {
			t.Errorf("Test failed for %s, Get DNS Zone failed: %v", test.name, err)
		}
		d.dnsZone = dnsZone

		d.ensureDnsRecords(&test.service)

		zones, err := fakednsZones.List()
		if err != nil {
			t.Errorf("error querying zones: %v", err)
		}

		// Dump every record to a testable-by-string-comparison form
		records := []string{}
		for _, z := range zones {
			zoneName := z.Name()

			rrs, ok := z.ResourceRecordSets()
			if !ok {
				t.Errorf("cannot get rrs for zone %q", zoneName)
			}

			rrList, err := rrs.List()
			if err != nil {
				t.Errorf("error querying rr for zone %q: %v", zoneName, err)
			}
			for _, rr := range rrList {
				rrdatas := rr.Rrdatas()

				// Put in consistent (testable-by-string-comparison) order
				sort.Strings(rrdatas)
				records = append(records, fmt.Sprintf("%s:%s:%s:%d:%s", zoneName, rr.Name(), rr.Type(), rr.Ttl(), rrdatas))
			}
		}

		// Ignore order of records
		sort.Strings(records)
		sort.Strings(test.expected)

		if !reflect.DeepEqual(records, test.expected) {
			t.Errorf("Test %q failed.  Actual=%v, Expected=%v", test.name, records, test.expected)
		}

	}
}
