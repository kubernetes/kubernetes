/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"context"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

func TestUserDirectory(t *testing.T) {
	s := New(NewServiceInstance(esx.ServiceContent, esx.RootFolder))

	ts := s.NewServer()
	defer ts.Close()

	ctx := context.Background()

	c, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	var tests = []struct {
		findUsers  bool
		findGroups bool
	}{
		{true, true},
		{true, false},
		{false, true},
		{false, false},
	}

	ref := *c.ServiceContent.UserDirectory

	for _, test := range tests {
		req := types.RetrieveUserGroups{
			This:       ref,
			SearchStr:  "root",
			ExactMatch: true,
			FindUsers:  test.findUsers,
			FindGroups: test.findGroups,
		}

		result, err := methods.RetrieveUserGroups(ctx, c.Client, &req)
		if err != nil {
			t.Fatal(err)
		}

		expectedSize := 0
		if test.findGroups {
			expectedSize++
		}
		if test.findUsers {
			expectedSize++
		}

		if len(result.Returnval) != expectedSize {
			t.Fatalf("expect search result for root is %d; got %d", expectedSize, len(result.Returnval))
		}

		for _, u := range result.Returnval {
			if u.GetUserSearchResult().Principal != "root" {
				t.Fatalf("expect principal to be root; got %s", u.GetUserSearchResult().Principal)
			}
			if !test.findGroups && u.GetUserSearchResult().Group {
				t.Fatal("expect search result is non-group; got group")
			}
			if !test.findUsers && !u.GetUserSearchResult().Group {
				t.Fatal("expect search result is non-user; got user")
			}
		}
	}
}

func TestUserDirectoryExactlyMatch(t *testing.T) {
	s := New(NewServiceInstance(esx.ServiceContent, esx.RootFolder))

	ts := s.NewServer()
	defer ts.Close()

	ctx := context.Background()

	c, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	var tests = []struct {
		exactly      bool
		search       string
		expectedSize int
	}{
		{true, "root", 2},
		{false, "ROO", 2},
	}

	ref := *c.ServiceContent.UserDirectory

	for _, test := range tests {
		req := types.RetrieveUserGroups{
			This:       ref,
			SearchStr:  test.search,
			ExactMatch: test.exactly,
			FindUsers:  true,
			FindGroups: true,
		}

		result, err := methods.RetrieveUserGroups(ctx, c.Client, &req)
		if err != nil {
			t.Fatal(err)
		}

		if err != nil {
			t.Fatal(err)
		}

		if len(result.Returnval) != test.expectedSize {
			t.Fatalf("expect result contains %d results; got %d", test.expectedSize, len(result.Returnval))
		}
	}
}
