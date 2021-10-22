// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package iam

import (
	"fmt"
	"sort"
	"testing"

	"cloud.google.com/go/internal/testutil"
)

func TestPolicy(t *testing.T) {
	p := &Policy{}

	add := func(member string, role RoleName) {
		p.Add(member, role)
	}
	remove := func(member string, role RoleName) {
		p.Remove(member, role)
	}

	if msg, ok := checkMembers(p, Owner, nil); !ok {
		t.Fatal(msg)
	}
	add("m1", Owner)
	if msg, ok := checkMembers(p, Owner, []string{"m1"}); !ok {
		t.Fatal(msg)
	}
	add("m2", Owner)
	if msg, ok := checkMembers(p, Owner, []string{"m1", "m2"}); !ok {
		t.Fatal(msg)
	}
	add("m1", Owner) // duplicate adds ignored
	if msg, ok := checkMembers(p, Owner, []string{"m1", "m2"}); !ok {
		t.Fatal(msg)
	}
	// No other roles populated yet.
	if msg, ok := checkMembers(p, Viewer, nil); !ok {
		t.Fatal(msg)
	}
	remove("m1", Owner)
	if msg, ok := checkMembers(p, Owner, []string{"m2"}); !ok {
		t.Fatal(msg)
	}
	if msg, ok := checkMembers(p, Viewer, nil); !ok {
		t.Fatal(msg)
	}
	remove("m3", Owner) // OK to remove non-existent member.
	if msg, ok := checkMembers(p, Owner, []string{"m2"}); !ok {
		t.Fatal(msg)
	}
	remove("m2", Owner)
	if msg, ok := checkMembers(p, Owner, nil); !ok {
		t.Fatal(msg)
	}
	if got, want := p.Roles(), []RoleName(nil); !testutil.Equal(got, want) {
		t.Fatalf("roles: got %v, want %v", got, want)
	}
}

func checkMembers(p *Policy, role RoleName, wantMembers []string) (string, bool) {
	gotMembers := p.Members(role)
	sort.Strings(gotMembers)
	sort.Strings(wantMembers)
	if !testutil.Equal(gotMembers, wantMembers) {
		return fmt.Sprintf("got %v, want %v", gotMembers, wantMembers), false
	}
	for _, m := range wantMembers {
		if !p.HasRole(m, role) {
			return fmt.Sprintf("member %q should have role %s but does not", m, role), false
		}
	}
	return "", true
}
