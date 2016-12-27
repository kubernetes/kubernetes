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
	"testing"

	"fmt"
	"reflect"
	"sort"

	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns" // Only for unit testing purposes.
	sc "k8s.io/kubernetes/federation/pkg/federation-controller/service"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
)

func TestServiceController_ensureDnsRecords(t *testing.T) {
	tests := []struct {
		name     string
		service  apiv1.Service
		exist    bool
		expected []string
	}{
		{
			name: "ServiceWithSingleIpEndpoint",
			service: apiv1.Service{
				ObjectMeta: apiv1.ObjectMeta{
					Name:      "nginx",
					Namespace: "ns",
					Annotations: sc.NewEndpointAnnotation(sc.NewEpMap().
						AddEndpoint("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						GetJSONMarshalledBytes()),
				},
			},
			exist: true,
			expected: []string{
				"example.com:nginx.ns.myfed.svc.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.fooregion.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.foozone.fooregion.fed.example.com:A:180:[198.51.100.1]",
			},
		},
		{
			name: "ServiceWithMultipleIpEndpoints",
			service: apiv1.Service{
				ObjectMeta: apiv1.ObjectMeta{
					Name:      "nginx",
					Namespace: "ns",
					Annotations: sc.NewEndpointAnnotation(sc.NewEpMap().
						AddEndpoint("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						AddEndpoint("barregion", "barzone", "c2", []string{"198.51.200.1"}, true).
						GetJSONMarshalledBytes()),
				},
			},
			exist: true,
			expected: []string{
				"example.com:nginx.ns.myfed.svc.fed.example.com:A:180:[198.51.100.1 198.51.200.1]",
				"example.com:nginx.ns.myfed.svc.fooregion.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.foozone.fooregion.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.barregion.fed.example.com:A:180:[198.51.200.1]",
				"example.com:nginx.ns.myfed.svc.barzone.barregion.fed.example.com:A:180:[198.51.200.1]",
			},
		},
		{
			name: "ServiceWithNoEndpoints",
			service: apiv1.Service{
				ObjectMeta: apiv1.ObjectMeta{
					Name:      "nginx",
					Namespace: "ns",
				},
			},
			exist:    true,
			expected: []string{},
		},
		{
			name: "ServiceWithEndpointAndServiceDeleted",
			service: apiv1.Service{
				ObjectMeta: apiv1.ObjectMeta{
					Name:      "nginx",
					Namespace: "ns",
					Annotations: sc.NewEndpointAnnotation(sc.NewEpMap().
						AddEndpoint("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						GetJSONMarshalledBytes()),
				},
			},
			exist:    false,
			expected: []string{},
		},
		{
			name: "ServiceWithMultipleIpEndpointsAndOneEndpointGettingRemoved",
			service: apiv1.Service{
				ObjectMeta: apiv1.ObjectMeta{
					Name:      "nginx",
					Namespace: "ns",
					Annotations: sc.NewEndpointAnnotation(sc.NewEpMap().
						AddEndpoint("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						AddEndpoint("barregion", "barzone", "c2", []string{"198.51.200.1"}, true).
						RemoveEndpoint("barregion", "barzone", "c2", "198.51.200.1").
						GetJSONMarshalledBytes()),
				},
			},
			exist: true,
			expected: []string{
				"example.com:nginx.ns.myfed.svc.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.fooregion.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.foozone.fooregion.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.barregion.fed.example.com:CNAME:180:[nginx.ns.myfed.svc.fed.example.com]",
				"example.com:nginx.ns.myfed.svc.barzone.barregion.fed.example.com:CNAME:180:[nginx.ns.myfed.svc.barregion.fed.example.com]",
			},
		},
		{
			name: "ServiceWithMultipleIpEndpointsAndOneEndpointIsUnhealthy",
			service: apiv1.Service{
				ObjectMeta: apiv1.ObjectMeta{
					Name:      "nginx",
					Namespace: "ns",
					Annotations: sc.NewEndpointAnnotation(sc.NewEpMap().
						AddEndpoint("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						AddEndpoint("barregion", "barzone", "c2", []string{"198.51.200.1"}, false).
						GetJSONMarshalledBytes()),
				},
			},
			exist: true,
			expected: []string{
				"example.com:nginx.ns.myfed.svc.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.fooregion.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.foozone.fooregion.fed.example.com:A:180:[198.51.100.1]",
				"example.com:nginx.ns.myfed.svc.barregion.fed.example.com:CNAME:180:[nginx.ns.myfed.svc.fed.example.com]",
				"example.com:nginx.ns.myfed.svc.barzone.barregion.fed.example.com:CNAME:180:[nginx.ns.myfed.svc.barregion.fed.example.com]",
			},
		},
		{
			name: "ServiceWithMultipleIpEndpointsAndAllEndpointGettingRemoved",
			service: apiv1.Service{
				ObjectMeta: apiv1.ObjectMeta{
					Name:      "nginx",
					Namespace: "ns",
					Annotations: sc.NewEndpointAnnotation(sc.NewEpMap().
						AddEndpoint("fooregion", "foozone", "c1", []string{"198.51.100.1"}, true).
						AddEndpoint("barregion", "barzone", "c2", []string{"198.51.200.1"}, true).
						RemoveEndpoint("fooregion", "foozone", "c1", "198.51.100.1").
						RemoveEndpoint("barregion", "barzone", "c2", "198.51.200.1").
						GetJSONMarshalledBytes()),
				},
			},
			exist: true,
			expected: []string{
				"example.com:nginx.ns.myfed.svc.fooregion.fed.example.com:CNAME:180:[nginx.ns.myfed.svc.fed.example.com]",
				"example.com:nginx.ns.myfed.svc.foozone.fooregion.fed.example.com:CNAME:180:[nginx.ns.myfed.svc.fooregion.fed.example.com]",
				"example.com:nginx.ns.myfed.svc.barregion.fed.example.com:CNAME:180:[nginx.ns.myfed.svc.fed.example.com]",
				"example.com:nginx.ns.myfed.svc.barzone.barregion.fed.example.com:CNAME:180:[nginx.ns.myfed.svc.barregion.fed.example.com]",
			},
		},
	}
	for _, test := range tests {
		fakedns, _ := clouddns.NewFakeInterface()
		fakednsZones, ok := fakedns.Zones()
		if !ok {
			t.Error("Unable to fetch zones")
		}
		dc := DNSController{
			dns:              fakedns,
			dnsZones:         fakednsZones,
			serviceDNSSuffix: "fed.example.com",
			zoneName:         "example.com",
			federationName:   "myfed",
		}

		dnsZone, err := getDNSZone(dc.zoneName, dc.zoneID, dc.dnsZones)
		if err != nil {
			t.Errorf("Test failed for %s, Get DNS Zone failed: %v", test.name, err)
		}

		err = dc.ensureDNSRrsets(dnsZone, &test.service, test.exist)
		if err != nil {
			t.Errorf("Test failed for %s, unexpected error %v", test.name, err)
		}

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
				records = append(records, fmt.Sprintf("%s:%s:%s:%d:%s",
					zoneName, rr.Name(), rr.Type(), rr.Ttl(), rrdatas))
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
