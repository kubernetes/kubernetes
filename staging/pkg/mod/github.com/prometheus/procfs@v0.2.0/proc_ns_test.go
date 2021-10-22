// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"testing"
)

func TestNewNamespaces(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	namespaces, err := p.Namespaces()
	if err != nil {
		t.Fatal(err)
	}

	expectedNamespaces := map[string]Namespace{
		"mnt": {"mnt", 4026531840},
		"net": {"net", 4026531993},
	}

	if want, have := len(expectedNamespaces), len(namespaces); want != have {
		t.Errorf("want %d parsed namespaces, have %d", want, have)
	}
	for _, ns := range namespaces {
		if want, have := expectedNamespaces[ns.Type], ns; want != have {
			t.Errorf("%s: want %v, have %v", ns.Type, want, have)
		}
	}
}
