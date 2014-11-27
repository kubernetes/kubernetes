/*
Copyright 2014 Google Inc. All rights reserved.

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

package v1beta1_test

import (
	"reflect"
	"testing"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
)

var Convert = newer.Scheme.Convert

func TestEnvConversion(t *testing.T) {
	nonCanonical := []v1beta1.EnvVar{
		{Key: "EV"},
		{Key: "EV", Name: "EX"},
	}
	canonical := []newer.EnvVar{
		{Name: "EV"},
		{Name: "EX"},
	}
	for i := range nonCanonical {
		var got newer.EnvVar
		err := Convert(&nonCanonical[i], &got)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if e, a := canonical[i], got; !reflect.DeepEqual(e, a) {
			t.Errorf("expected %v, got %v", e, a)
		}
	}

	// Test conversion the other way, too.
	for i := range canonical {
		var got v1beta1.EnvVar
		err := Convert(&canonical[i], &got)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if e, a := canonical[i].Name, got.Key; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
		if e, a := canonical[i].Name, got.Name; e != a {
			t.Errorf("expected %v, got %v", e, a)
		}
	}
}

func TestVolumeMountConversionToOld(t *testing.T) {
	table := []struct {
		in  newer.VolumeMount
		out v1beta1.VolumeMount
	}{
		{
			in:  newer.VolumeMount{Name: "foo", MountPath: "/dev/foo", ReadOnly: true},
			out: v1beta1.VolumeMount{Name: "foo", MountPath: "/dev/foo", Path: "/dev/foo", ReadOnly: true},
		},
	}
	for _, item := range table {
		got := v1beta1.VolumeMount{}
		err := Convert(&item.in, &got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := item.out, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected: %#v, got %#v", e, a)
		}
	}
}

func TestVolumeMountConversionToNew(t *testing.T) {
	table := []struct {
		in  v1beta1.VolumeMount
		out newer.VolumeMount
	}{
		{
			in:  v1beta1.VolumeMount{Name: "foo", MountPath: "/dev/foo", ReadOnly: true},
			out: newer.VolumeMount{Name: "foo", MountPath: "/dev/foo", ReadOnly: true},
		}, {
			in:  v1beta1.VolumeMount{Name: "foo", MountPath: "/dev/foo", Path: "/dev/bar", ReadOnly: true},
			out: newer.VolumeMount{Name: "foo", MountPath: "/dev/foo", ReadOnly: true},
		}, {
			in:  v1beta1.VolumeMount{Name: "foo", Path: "/dev/bar", ReadOnly: true},
			out: newer.VolumeMount{Name: "foo", MountPath: "/dev/bar", ReadOnly: true},
		},
	}
	for _, item := range table {
		got := newer.VolumeMount{}
		err := Convert(&item.in, &got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		if e, a := item.out, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected: %#v, got %#v", e, a)
		}
	}
}

func TestMinionListConversionToNew(t *testing.T) {
	oldMinion := func(id string) v1beta1.Minion {
		return v1beta1.Minion{TypeMeta: v1beta1.TypeMeta{ID: id}}
	}
	newMinion := func(id string) newer.Minion {
		return newer.Minion{ObjectMeta: newer.ObjectMeta{Name: id}}
	}
	oldMinions := []v1beta1.Minion{
		oldMinion("foo"),
		oldMinion("bar"),
	}
	newMinions := []newer.Minion{
		newMinion("foo"),
		newMinion("bar"),
	}

	table := []struct {
		oldML *v1beta1.MinionList
		newML *newer.MinionList
	}{
		{
			oldML: &v1beta1.MinionList{Items: oldMinions},
			newML: &newer.MinionList{Items: newMinions},
		}, {
			oldML: &v1beta1.MinionList{Minions: oldMinions},
			newML: &newer.MinionList{Items: newMinions},
		}, {
			oldML: &v1beta1.MinionList{
				Items:   oldMinions,
				Minions: []v1beta1.Minion{oldMinion("baz")},
			},
			newML: &newer.MinionList{Items: newMinions},
		},
	}

	for _, item := range table {
		got := &newer.MinionList{}
		err := Convert(item.oldML, got)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if e, a := item.newML, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected: %#v, got %#v", e, a)
		}
	}
}

func TestMinionListConversionToOld(t *testing.T) {
	oldMinion := func(id string) v1beta1.Minion {
		return v1beta1.Minion{TypeMeta: v1beta1.TypeMeta{ID: id}}
	}
	newMinion := func(id string) newer.Minion {
		return newer.Minion{ObjectMeta: newer.ObjectMeta{Name: id}}
	}
	oldMinions := []v1beta1.Minion{
		oldMinion("foo"),
		oldMinion("bar"),
	}
	newMinions := []newer.Minion{
		newMinion("foo"),
		newMinion("bar"),
	}

	newML := &newer.MinionList{Items: newMinions}
	oldML := &v1beta1.MinionList{
		Items:   oldMinions,
		Minions: oldMinions,
	}

	got := &v1beta1.MinionList{}
	err := Convert(newML, got)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := oldML, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %#v, got %#v", e, a)
	}
}
