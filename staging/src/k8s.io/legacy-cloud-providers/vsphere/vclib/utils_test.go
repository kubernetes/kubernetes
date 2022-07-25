/*
Copyright 2018 The Kubernetes Authors.

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

package vclib

import (
	"context"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/simulator"
)

func TestUtils(t *testing.T) {
	ctx := context.Background()

	model := simulator.VPX()
	// Child folder "F0" will be created under the root folder and datacenter folders,
	// and all resources are created within the "F0" child folders.
	model.Folder = 1

	defer model.Remove()
	err := model.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := model.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	vc := &VSphereConnection{Client: c.Client}

	dc, err := GetDatacenter(ctx, vc, TestDefaultDatacenter)
	if err != nil {
		t.Error(err)
	}

	finder := getFinder(dc)
	datastores, err := finder.DatastoreList(ctx, "*")
	if err != nil {
		t.Fatal(err)
	}

	count := model.Count()
	if count.Datastore != len(datastores) {
		t.Errorf("got %d Datastores, expected: %d", len(datastores), count.Datastore)
	}

	_, err = finder.Datastore(ctx, testNameNotFound)
	if !IsNotFound(err) {
		t.Errorf("unexpected error: %s", err)
	}
}

func TestIsvCenterDeprecated(t *testing.T) {
	type testsData struct {
		vcVersion    string
		vcAPIVersion string
		isDeprecated bool
	}
	testdataArray := []testsData{
		{"8.0.0", "8.0.0.0", false},
		{"7.0.3", "7.0.3.0", false},
		{"7.0.2", "7.0.2.0", false},
		{"7.0.1", "7.0.1.1", true},
		{"7.0.0", "7.0.0.0", true},
		{"6.7.0", "6.7.3", true},
		{"6.7.0", "6.7", true},
		{"6.7.0", "6.7.2", true},
		{"6.7.0", "6.7.1", true},
		{"6.5.0", "6.5", true},
	}

	for _, test := range testdataArray {
		deprecated, err := isvCenterDeprecated(test.vcVersion, test.vcAPIVersion)
		if err != nil {
			t.Fatal(err)
		}
		if deprecated != test.isDeprecated {
			t.Fatalf("deprecation test failed for vc version: %q and vc API version: %q",
				test.vcVersion, test.vcAPIVersion)
		} else {
			t.Logf("deprecation test for vc version: %q and vc API version: %q passed. Is Deprecated : %v",
				test.vcAPIVersion, test.vcAPIVersion, deprecated)
		}
	}
}
