/*
Copyright 2015 The Kubernetes Authors.

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

package endpoints

import (
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
)

func podRef(uid string) *v1.ObjectReference {
	ref := v1.ObjectReference{UID: types.UID(uid)}
	return &ref
}

func TestPackSubsets(t *testing.T) {
	// The downside of table-driven tests is that some things have to live outside the table.
	fooObjRef := v1.ObjectReference{Name: "foo"}
	barObjRef := v1.ObjectReference{Name: "bar"}

	testCases := []struct {
		name   string
		given  []v1.EndpointSubset
		expect []v1.EndpointSubset
	}{
		{
			name:   "empty everything",
			given:  []v1.EndpointSubset{{Addresses: []v1.EndpointAddress{}, Ports: []v1.EndpointPort{}}},
			expect: []v1.EndpointSubset{},
		}, {
			name:   "empty addresses",
			given:  []v1.EndpointSubset{{Addresses: []v1.EndpointAddress{}, Ports: []v1.EndpointPort{{Port: 111}}}},
			expect: []v1.EndpointSubset{},
		}, {
			name:   "empty ports",
			given:  []v1.EndpointSubset{{Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}}, Ports: []v1.EndpointPort{}}},
			expect: []v1.EndpointSubset{},
		}, {
			name:   "empty ports",
			given:  []v1.EndpointSubset{{NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4"}}, Ports: []v1.EndpointPort{}}},
			expect: []v1.EndpointSubset{},
		}, {
			name: "one set, one ip, one port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, one ip, one port (IPv6)",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "beef::1:2:3:4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "beef::1:2:3:4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, one notReady ip, one port",
			given: []v1.EndpointSubset{{
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, one ip, one UID, one port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, one notReady ip, one UID, one port",
			given: []v1.EndpointSubset{{
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, one ip, empty UID, one port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, one notReady ip, empty UID, one port",
			given: []v1.EndpointSubset{{
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("")}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("")}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, two ips, one port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, two mixed ips, one port",
			given: []v1.EndpointSubset{{
				Addresses:         []v1.EndpointAddress{{IP: "1.2.3.4"}},
				NotReadyAddresses: []v1.EndpointAddress{{IP: "5.6.7.8"}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses:         []v1.EndpointAddress{{IP: "1.2.3.4"}},
				NotReadyAddresses: []v1.EndpointAddress{{IP: "5.6.7.8"}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, two duplicate ips, one port, notReady is covered by ready",
			given: []v1.EndpointSubset{{
				Addresses:         []v1.EndpointAddress{{IP: "1.2.3.4"}},
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, one ip, two ports",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}, {Port: 222}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}, {Port: 222}},
			}},
		}, {
			name: "one set, dup ips, one port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}, {IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, dup ips, one port (IPv6)",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "beef::1"}, {IP: "beef::1"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "beef::1"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, dup ips with target-refs, one port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{
					{IP: "1.2.3.4", TargetRef: &fooObjRef},
					{IP: "1.2.3.4", TargetRef: &barObjRef},
				},
				Ports: []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: &fooObjRef}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, dup mixed ips with target-refs, one port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{
					{IP: "1.2.3.4", TargetRef: &fooObjRef},
				},
				NotReadyAddresses: []v1.EndpointAddress{
					{IP: "1.2.3.4", TargetRef: &barObjRef},
				},
				Ports: []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				// finding the same address twice is considered an error on input, only the first address+port
				// reference is preserved
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: &fooObjRef}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "one set, one ip, dup ports",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}, {Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "two sets, dup ip, dup port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "two sets, dup mixed ip, dup port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "two sets, dup ip, two ports",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 222}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}, {Port: 222}},
			}},
		}, {
			name: "two sets, dup ip, dup uids, two ports",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 222}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 111}, {Port: 222}},
			}},
		}, {
			name: "two sets, dup mixed ip, dup uids, two ports",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:             []v1.EndpointPort{{Port: 222}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:             []v1.EndpointPort{{Port: 222}},
			}},
		}, {
			name: "two sets, two ips, dup port",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "5.6.7.8"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}, {IP: "5.6.7.8"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "two set, dup ip, two uids, dup ports",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-2")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{
					{IP: "1.2.3.4", TargetRef: podRef("uid-1")},
					{IP: "1.2.3.4", TargetRef: podRef("uid-2")},
				},
				Ports: []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "two set, dup ip, with and without uid, dup ports",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-2")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{
					{IP: "1.2.3.4"},
					{IP: "1.2.3.4", TargetRef: podRef("uid-2")},
				},
				Ports: []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "two sets, two ips, two dup ip with uid, dup port, wrong order",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "5.6.7.8"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "5.6.7.8", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{
					{IP: "1.2.3.4"},
					{IP: "1.2.3.4", TargetRef: podRef("uid-1")},
					{IP: "5.6.7.8"},
					{IP: "5.6.7.8", TargetRef: podRef("uid-1")},
				},
				Ports: []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "two sets, two mixed ips, two dup ip with uid, dup port, wrong order, ends up with split addresses",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "5.6.7.8"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				NotReadyAddresses: []v1.EndpointAddress{{IP: "5.6.7.8", TargetRef: podRef("uid-1")}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}, {
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4", TargetRef: podRef("uid-1")}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}, {
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:             []v1.EndpointPort{{Port: 111}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{
					{IP: "5.6.7.8"},
				},
				NotReadyAddresses: []v1.EndpointAddress{
					{IP: "1.2.3.4"},
					{IP: "1.2.3.4", TargetRef: podRef("uid-1")},
					{IP: "5.6.7.8", TargetRef: podRef("uid-1")},
				},
				Ports: []v1.EndpointPort{{Port: 111}},
			}},
		}, {
			name: "two sets, two ips, two ports",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "5.6.7.8"}},
				Ports:     []v1.EndpointPort{{Port: 222}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "5.6.7.8"}},
				Ports:     []v1.EndpointPort{{Port: 222}},
			}},
		}, {
			name: "four sets, three ips, three ports, jumbled",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.5"}},
				Ports:     []v1.EndpointPort{{Port: 222}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.6"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.5"}},
				Ports:     []v1.EndpointPort{{Port: 333}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}, {IP: "1.2.3.6"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.5"}},
				Ports:     []v1.EndpointPort{{Port: 222}, {Port: 333}},
			}},
		}, {
			name: "four sets, three mixed ips, three ports, jumbled",
			given: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.5"}},
				Ports:             []v1.EndpointPort{{Port: 222}},
			}, {
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.6"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.5"}},
				Ports:             []v1.EndpointPort{{Port: 333}},
			}},
			expect: []v1.EndpointSubset{{
				Addresses: []v1.EndpointAddress{{IP: "1.2.3.4"}, {IP: "1.2.3.6"}},
				Ports:     []v1.EndpointPort{{Port: 111}},
			}, {
				NotReadyAddresses: []v1.EndpointAddress{{IP: "1.2.3.5"}},
				Ports:             []v1.EndpointPort{{Port: 222}, {Port: 333}},
			}},
		},
	}

	for _, tc := range testCases {
		result := RepackSubsets(tc.given)
		if !reflect.DeepEqual(result, SortSubsets(tc.expect)) {
			t.Errorf("case %q: expected %s, got %s", tc.name, spew.Sprintf("%#v", SortSubsets(tc.expect)), spew.Sprintf("%#v", result))
		}
	}
}
