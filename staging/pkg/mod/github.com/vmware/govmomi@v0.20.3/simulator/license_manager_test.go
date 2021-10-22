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
	"reflect"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/license"
)

func TestLicenseManagerVPX(t *testing.T) {
	ctx := context.Background()
	m := VPX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	lm := license.NewManager(c.Client)
	am, err := lm.AssignmentManager(ctx)
	if err != nil {
		t.Fatal(err)
	}

	la, err := am.QueryAssigned(ctx, "enoent")
	if err != nil {
		t.Fatal(err)
	}

	if len(la) != 0 {
		t.Errorf("unexpected license")
	}

	finder := find.NewFinder(c.Client, false)
	hosts, err := finder.HostSystemList(ctx, "/...")
	if err != nil {
		t.Fatal(err)
	}

	host := hosts[0].Reference().Value
	vcid := c.Client.ServiceContent.About.InstanceUuid

	for _, name := range []string{"", host, vcid} {
		la, err = am.QueryAssigned(ctx, name)
		if err != nil {
			t.Fatal(err)
		}

		if len(la) != 1 {
			t.Fatal("no licenses")
		}

		if !reflect.DeepEqual(la[0].AssignedLicense, EvalLicense) {
			t.Fatal("invalid license")
		}
	}
}

func TestLicenseManagerESX(t *testing.T) {
	ctx := context.Background()
	m := ESX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	lm := license.NewManager(c.Client)
	_, err = lm.AssignmentManager(ctx)
	if err == nil {
		t.Fatal("expected error")
	}

	la, err := lm.List(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if len(la) != 1 {
		t.Fatal("no licenses")
	}

	if !reflect.DeepEqual(la[0], EvalLicense) {
		t.Fatal("invalid license")
	}
}

func TestAddRemoveLicense(t *testing.T) {
	ctx := context.Background()
	m := ESX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	lm := license.NewManager(c.Client)
	key := "00000-00000-00000-00000-11111"
	labels := map[string]string{"key": "value"}

	info, err := lm.Add(ctx, key, labels)
	if err != nil {
		t.Fatal(err)
	}

	if info.LicenseKey != key {
		t.Fatalf("expect info.LicenseKey equal to %q; got %q", key, info.LicenseKey)
	}

	if len(info.Labels) != len(labels) {
		t.Fatalf("expect len(info.Labels) eqaul to %d; got %d",
			len(labels), len(info.Labels))
	}

	if info.Labels[0].Key != "key" || info.Labels[0].Value != "value" {
		t.Fatalf("expect label to be {key:value}; got {%s:%s}",
			info.Labels[0].Key, info.Labels[0].Value)
	}

	la, err := lm.List(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if len(la) != 2 {
		t.Fatal("no licenses")
	}

	if la[1].LicenseKey != key {
		t.Fatalf("expect info.LicenseKey equal to %q; got %q",
			key, la[1].LicenseKey)
	}

	if len(la[1].Labels) != len(labels) {
		t.Fatalf("expect len(info.Labels) eqaul to %d; got %d",
			len(labels), len(la[1].Labels))
	}

	if la[1].Labels[0].Key != "key" || la[1].Labels[0].Value != "value" {
		t.Fatalf("expect label to be {key:value}; got {%s:%s}",
			la[1].Labels[0].Key, la[1].Labels[0].Value)
	}

	err = lm.Remove(ctx, key)
	if err != nil {
		t.Fatal(err)
	}

	la, err = lm.List(ctx)
	if err != nil {
		t.Fatal(err)
	}

	if len(la) != 1 {
		t.Fatal("no licenses")
	}
}
