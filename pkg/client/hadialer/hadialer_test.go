/*
Copyright 2014 The Kubernetes Authors.

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

package hadialer

import (
	"slices"
	"testing"
)

func TestInterleaveIPV4AndV6(t *testing.T) {
	addrs := []string{
		"192.168.200.1",
		"10.111.3.1",
		"192.168.200.2",
		"fd11::102",
		"fd11::101",
	}

	interleaveIPV4AndV6(&addrs)

	expected := []string{
		"192.168.200.1",
		"fd11::102",
		"10.111.3.1",
		"fd11::101",
		"192.168.200.2",
	}

	if !slices.Equal(addrs, expected) {
		t.Errorf("expected %v, got %v", expected, addrs)
	}
}
