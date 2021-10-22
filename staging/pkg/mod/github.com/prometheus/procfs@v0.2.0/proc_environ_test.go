// Copyright 2019 The Prometheus Authors
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

import "testing"

func TestProcEnviron(t *testing.T) {
	p, err := getProcFixtures(t).Proc(26231)
	if err != nil {
		t.Fatal(err)
	}

	environments, err := p.Environ()
	if err != nil {
		t.Fatal(err)
	}

	expectedEnvironments := []string{
		"PATH=/go/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
		"HOSTNAME=cd24e11f73a5",
		"TERM=xterm",
		"GOLANG_VERSION=1.12.5",
		"GOPATH=/go",
		"HOME=/root",
	}

	if want, have := len(expectedEnvironments), len(environments); want != have {
		t.Errorf("want %d parsed environments, have %d", want, have)
	}

	for i, environment := range environments {
		if want, have := expectedEnvironments[i], environment; want != have {
			t.Errorf("%d: want %v, have %v", i, want, have)
		}
	}
}
