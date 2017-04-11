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

package dnsprovider

import (
	"testing"

	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time interface check
var _ ResourceRecordSet = record{}

type record struct {
	name    string
	rrdatas []string
	ttl     int64
	type_   string
}

func (r record) Name() string {
	return r.name
}

func (r record) Ttl() int64 {
	return r.ttl
}

func (r record) Rrdatas() []string {
	return r.rrdatas
}

func (r record) Type() rrstype.RrsType {
	return rrstype.RrsType(r.type_)
}

const testDNSZone string = "foo.com"

var testData = []struct {
	inputs         [2]record
	expectedOutput bool
}{
	{
		[2]record{
			{"foo", []string{"1.2.3.4", "5,6,7,8"}, 180, "A"}, // Identical
			{"foo", []string{"1.2.3.4", "5,6,7,8"}, 180, "A"}}, true,
	},
	{
		[2]record{
			{"foo", []string{"1.2.3.4", "5,6,7,8"}, 180, "A"}, // Identical except Name
			{"bar", []string{"1.2.3.4", "5,6,7,8"}, 180, "A"}}, false,
	},
	{
		[2]record{
			{"foo", []string{"1.2.3.4", "5,6,7,8"}, 180, "A"}, // Identical except Rrdata
			{"foo", []string{"1.2.3.4", "5,6,7,9"}, 180, "A"}}, false,
	},
	{
		[2]record{
			{"foo", []string{"1.2.3.4", "5,6,7,8"}, 180, "A"}, // Identical except Rrdata ordering reversed
			{"foo", []string{"5,6,7,8", "1.2.3.4"}, 180, "A"}}, false,
	},
	{
		[2]record{
			{"foo", []string{"1.2.3.4", "5,6,7,8"}, 180, "A"}, // Identical except TTL
			{"foo", []string{"1.2.3.4", "5,6,7,8"}, 150, "A"}}, false,
	},
	{
		[2]record{
			{"foo", []string{"1.2.3.4", "5,6,7,8"}, 180, "A"}, // Identical except Type
			{"foo", []string{"1.2.3.4", "5,6,7,8"}, 180, "CNAME"}}, false,
	},
}

func TestEquivalent(t *testing.T) {
	for _, test := range testData {
		output := ResourceRecordSetsEquivalent(test.inputs[0], test.inputs[1])
		if output != test.expectedOutput {
			t.Errorf("Expected equivalence comparison of %q and %q to yield %v, but it vielded %v", test.inputs[0], test.inputs[1], test.expectedOutput, output)
		}
	}
}
