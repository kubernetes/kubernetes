/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"math/rand"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

func expectPrefix(t *testing.T, prefix string, errs field.ErrorList) {
	for i := range errs {
		if f, p := errs[i].Field, prefix; !strings.HasPrefix(f, p) {
			t.Errorf("expected prefix '%s' for field '%s' (%v)", p, f, errs[i])
		}
	}
}

// Ensure custom name functions are allowed
func TestValidateObjectMetaCustomName(t *testing.T) {
	errs := ValidateObjectMeta(
		&api.ObjectMeta{Name: "test", GenerateName: "foo"},
		false,
		func(s string, prefix bool) (bool, string) {
			if s == "test" {
				return true, ""
			}
			return false, "name-gen"
		},
		field.NewPath("field"))
	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), "name-gen") {
		t.Errorf("unexpected error message: %v", errs)
	}
}

// Ensure namespace names follow dns label format
func TestValidateObjectMetaNamespaces(t *testing.T) {
	errs := ValidateObjectMeta(
		&api.ObjectMeta{Name: "test", Namespace: "foo.bar"},
		true,
		func(s string, prefix bool) (bool, string) {
			return true, ""
		},
		field.NewPath("field"))
	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), `Invalid value: "foo.bar"`) {
		t.Errorf("unexpected error message: %v", errs)
	}
	maxLength := 63
	letters := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
	b := make([]rune, maxLength+1)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	errs = ValidateObjectMeta(
		&api.ObjectMeta{Name: "test", Namespace: string(b)},
		true,
		func(s string, prefix bool) (bool, string) {
			return true, ""
		},
		field.NewPath("field"))
	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), "Invalid value") {
		t.Errorf("unexpected error message: %v", errs)
	}
}

func TestValidateObjectMetaOwnerReferences(t *testing.T) {
	testCases := []struct {
		ownerReferences []api.OwnerReference
		expectError     bool
	}{
		{
			[]api.OwnerReference{
				{
					APIVersion: "thirdpartyVersion",
					Kind:       "thirdpartyKind",
					Name:       "name",
					UID:        "1",
				},
			},
			false,
		},
		{
			// event shouldn't be set as an owner
			[]api.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "Event",
					Name:       "name",
					UID:        "1",
				},
			},
			true,
		},
	}

	for _, tc := range testCases {
		errs := ValidateObjectMeta(
			&api.ObjectMeta{Name: "test", Namespace: "test", OwnerReferences: tc.ownerReferences},
			true,
			func(s string, prefix bool) (bool, string) {
				return true, ""
			},
			field.NewPath("field"))
		if len(errs) != 0 && !tc.expectError {
			t.Errorf("unexpected error: %v", errs)
		}
		if len(errs) == 0 && tc.expectError {
			t.Errorf("expect error")
		}
		if len(errs) != 0 && !strings.Contains(errs[0].Error(), "is disallowed from being an owner") {
			t.Errorf("unexpected error message: %v", errs)
		}
	}
}

func TestValidateObjectMetaUpdateIgnoresCreationTimestamp(t *testing.T) {
	if errs := ValidateObjectMetaUpdate(
		&api.ObjectMeta{Name: "test", ResourceVersion: "1"},
		&api.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: unversioned.NewTime(time.Unix(10, 0))},
		field.NewPath("field"),
	); len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if errs := ValidateObjectMetaUpdate(
		&api.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: unversioned.NewTime(time.Unix(10, 0))},
		&api.ObjectMeta{Name: "test", ResourceVersion: "1"},
		field.NewPath("field"),
	); len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if errs := ValidateObjectMetaUpdate(
		&api.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: unversioned.NewTime(time.Unix(10, 0))},
		&api.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: unversioned.NewTime(time.Unix(11, 0))},
		field.NewPath("field"),
	); len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
}

func TestValidateObjectMetaUpdatePreventsDeletionFieldMutation(t *testing.T) {
	now := unversioned.NewTime(time.Unix(1000, 0).UTC())
	later := unversioned.NewTime(time.Unix(2000, 0).UTC())
	gracePeriodShort := int64(30)
	gracePeriodLong := int64(40)

	testcases := map[string]struct {
		Old          api.ObjectMeta
		New          api.ObjectMeta
		ExpectedNew  api.ObjectMeta
		ExpectedErrs []string
	}{
		"valid without deletion fields": {
			Old:          api.ObjectMeta{Name: "test", ResourceVersion: "1"},
			New:          api.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedNew:  api.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedErrs: []string{},
		},
		"valid with deletion fields": {
			Old:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &gracePeriodShort},
			New:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedNew:  api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedErrs: []string{},
		},

		"invalid set deletionTimestamp": {
			Old:          api.ObjectMeta{Name: "test", ResourceVersion: "1"},
			New:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			ExpectedNew:  api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			ExpectedErrs: []string{"field.deletionTimestamp: Invalid value: \"1970-01-01T00:16:40Z\": field is immutable; may only be changed via deletion"},
		},
		"invalid clear deletionTimestamp": {
			Old:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			New:          api.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedNew:  api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			ExpectedErrs: []string{}, // no errors, validation copies the old value
		},
		"invalid change deletionTimestamp": {
			Old:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			New:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &later},
			ExpectedNew:  api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionTimestamp: &now},
			ExpectedErrs: []string{}, // no errors, validation copies the old value
		},

		"invalid set deletionGracePeriodSeconds": {
			Old:          api.ObjectMeta{Name: "test", ResourceVersion: "1"},
			New:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedNew:  api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedErrs: []string{"field.deletionGracePeriodSeconds: Invalid value: 30: field is immutable; may only be changed via deletion"},
		},
		"invalid clear deletionGracePeriodSeconds": {
			Old:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			New:          api.ObjectMeta{Name: "test", ResourceVersion: "1"},
			ExpectedNew:  api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			ExpectedErrs: []string{}, // no errors, validation copies the old value
		},
		"invalid change deletionGracePeriodSeconds": {
			Old:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodShort},
			New:          api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodLong},
			ExpectedNew:  api.ObjectMeta{Name: "test", ResourceVersion: "1", DeletionGracePeriodSeconds: &gracePeriodLong},
			ExpectedErrs: []string{"field.deletionGracePeriodSeconds: Invalid value: 40: field is immutable; may only be changed via deletion"},
		},
	}

	for k, tc := range testcases {
		errs := ValidateObjectMetaUpdate(&tc.New, &tc.Old, field.NewPath("field"))
		if len(errs) != len(tc.ExpectedErrs) {
			t.Logf("%s: Expected: %#v", k, tc.ExpectedErrs)
			t.Logf("%s: Got: %#v", k, errs)
			t.Errorf("%s: expected %d errors, got %d", k, len(tc.ExpectedErrs), len(errs))
			continue
		}
		for i := range errs {
			if errs[i].Error() != tc.ExpectedErrs[i] {
				t.Errorf("%s: error #%d: expected %q, got %q", k, i, tc.ExpectedErrs[i], errs[i].Error())
			}
		}
		if !reflect.DeepEqual(tc.New, tc.ExpectedNew) {
			t.Errorf("%s: Expected after validation:\n%#v\ngot\n%#v", k, tc.ExpectedNew, tc.New)
		}
	}
}

// Ensure trailing slash is allowed in generate name
func TestValidateObjectMetaTrimsTrailingSlash(t *testing.T) {
	errs := ValidateObjectMeta(
		&api.ObjectMeta{Name: "test", GenerateName: "foo-"},
		false,
		NameIsDNSSubdomain,
		field.NewPath("field"))
	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
}

// Ensure updating finalizers is disallowed
func TestValidateObjectMetaUpdateDisallowsUpdatingFinalizers(t *testing.T) {
	errs := ValidateObjectMetaUpdate(
		&api.ObjectMeta{Name: "test", ResourceVersion: "1", Finalizers: []string{"orphaning"}},
		&api.ObjectMeta{Name: "test", ResourceVersion: "1"},
		field.NewPath("field"),
	)
	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), "field is immutable") {
		t.Errorf("unexpected error message: %v", errs)
	}
}

func TestValidateLabels(t *testing.T) {
	successCases := []map[string]string{
		{"simple": "bar"},
		{"now-with-dashes": "bar"},
		{"1-starts-with-num": "bar"},
		{"1234": "bar"},
		{"simple/simple": "bar"},
		{"now-with-dashes/simple": "bar"},
		{"now-with-dashes/now-with-dashes": "bar"},
		{"now.with.dots/simple": "bar"},
		{"now-with.dashes-and.dots/simple": "bar"},
		{"1-num.2-num/3-num": "bar"},
		{"1234/5678": "bar"},
		{"1.2.3.4/5678": "bar"},
		{"UpperCaseAreOK123": "bar"},
		{"goodvalue": "123_-.BaR"},
	}
	for i := range successCases {
		errs := ValidateLabels(successCases[i], field.NewPath("field"))
		if len(errs) != 0 {
			t.Errorf("case[%d] expected success, got %#v", i, errs)
		}
	}

	labelNameErrorCases := []map[string]string{
		{"nospecialchars^=@": "bar"},
		{"cantendwithadash-": "bar"},
		{"only/one/slash": "bar"},
		{strings.Repeat("a", 254): "bar"},
	}
	for i := range labelNameErrorCases {
		errs := ValidateLabels(labelNameErrorCases[i], field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		} else {
			detail := errs[0].Detail
			if detail != qualifiedNameErrorMsg {
				t.Errorf("error detail %s should be equal %s", detail, qualifiedNameErrorMsg)
			}
		}
	}

	labelValueErrorCases := []map[string]string{
		{"toolongvalue": strings.Repeat("a", 64)},
		{"backslashesinvalue": "some\\bad\\value"},
		{"nocommasallowed": "bad,value"},
		{"strangecharsinvalue": "?#$notsogood"},
	}
	for i := range labelValueErrorCases {
		errs := ValidateLabels(labelValueErrorCases[i], field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		} else {
			detail := errs[0].Detail
			if detail != labelValueErrorMsg {
				t.Errorf("error detail %s should be equal %s", detail, labelValueErrorMsg)
			}
		}
	}
}

func TestValidateAnnotations(t *testing.T) {
	successCases := []map[string]string{
		{"simple": "bar"},
		{"now-with-dashes": "bar"},
		{"1-starts-with-num": "bar"},
		{"1234": "bar"},
		{"simple/simple": "bar"},
		{"now-with-dashes/simple": "bar"},
		{"now-with-dashes/now-with-dashes": "bar"},
		{"now.with.dots/simple": "bar"},
		{"now-with.dashes-and.dots/simple": "bar"},
		{"1-num.2-num/3-num": "bar"},
		{"1234/5678": "bar"},
		{"1.2.3.4/5678": "bar"},
		{"UpperCase123": "bar"},
		{"a": strings.Repeat("b", totalAnnotationSizeLimitB-1)},
		{
			"a": strings.Repeat("b", totalAnnotationSizeLimitB/2-1),
			"c": strings.Repeat("d", totalAnnotationSizeLimitB/2-1),
		},
	}
	for i := range successCases {
		errs := ValidateAnnotations(successCases[i], field.NewPath("field"))
		if len(errs) != 0 {
			t.Errorf("case[%d] expected success, got %#v", i, errs)
		}
	}

	nameErrorCases := []map[string]string{
		{"nospecialchars^=@": "bar"},
		{"cantendwithadash-": "bar"},
		{"only/one/slash": "bar"},
		{strings.Repeat("a", 254): "bar"},
	}
	for i := range nameErrorCases {
		errs := ValidateAnnotations(nameErrorCases[i], field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		}
		detail := errs[0].Detail
		if detail != qualifiedNameErrorMsg {
			t.Errorf("error detail %s should be equal %s", detail, qualifiedNameErrorMsg)
		}
	}
	totalSizeErrorCases := []map[string]string{
		{"a": strings.Repeat("b", totalAnnotationSizeLimitB)},
		{
			"a": strings.Repeat("b", totalAnnotationSizeLimitB/2),
			"c": strings.Repeat("d", totalAnnotationSizeLimitB/2),
		},
	}
	for i := range totalSizeErrorCases {
		errs := ValidateAnnotations(totalSizeErrorCases[i], field.NewPath("field"))
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		}
	}
}

func testVolume(name string, namespace string, spec api.PersistentVolumeSpec) *api.PersistentVolume {
	objMeta := api.ObjectMeta{Name: name}
	if namespace != "" {
		objMeta.Namespace = namespace
	}

	return &api.PersistentVolume{
		ObjectMeta: objMeta,
		Spec:       spec,
	}
}

func TestValidatePersistentVolumes(t *testing.T) {
	scenarios := map[string]struct {
		isExpectedFailure bool
		volume            *api.PersistentVolume
	}{
		"good-volume": {
			isExpectedFailure: false,
			volume: testVolume("foo", "", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: "/foo"},
				},
			}),
		},
		"invalid-accessmode": {
			isExpectedFailure: true,
			volume: testVolume("foo", "", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{"fakemode"},
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: "/foo"},
				},
			}),
		},
		"unexpected-namespace": {
			isExpectedFailure: true,
			volume: testVolume("foo", "unexpected-namespace", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: "/foo"},
				},
			}),
		},
		"bad-name": {
			isExpectedFailure: true,
			volume: testVolume("123*Bad(Name", "unexpected-namespace", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: "/foo"},
				},
			}),
		},
		"missing-name": {
			isExpectedFailure: true,
			volume: testVolume("", "", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
			}),
		},
		"missing-capacity": {
			isExpectedFailure: true,
			volume:            testVolume("foo", "", api.PersistentVolumeSpec{}),
		},
		"missing-accessmodes": {
			isExpectedFailure: true,
			volume: testVolume("goodname", "missing-accessmodes", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: "/foo"},
				},
			}),
		},
		"too-many-sources": {
			isExpectedFailure: true,
			volume: testVolume("", "", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("5G"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath:          &api.HostPathVolumeSource{Path: "/foo"},
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{PDName: "foo", FSType: "ext4"},
				},
			}),
		},
	}

	for name, scenario := range scenarios {
		errs := ValidatePersistentVolume(scenario.volume)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}

}

func testVolumeClaim(name string, namespace string, spec api.PersistentVolumeClaimSpec) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       spec,
	}
}

func TestValidatePersistentVolumeClaim(t *testing.T) {
	scenarios := map[string]struct {
		isExpectedFailure bool
		claim             *api.PersistentVolumeClaim
	}{
		"good-claim": {
			isExpectedFailure: false,
			claim: testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
				},
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
					},
				},
			}),
		},
		"invalid-accessmode": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
				AccessModes: []api.PersistentVolumeAccessMode{"fakemode"},
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
					},
				},
			}),
		},
		"missing-namespace": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "", api.PersistentVolumeClaimSpec{
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadOnlyMany,
				},
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
					},
				},
			}),
		},
		"no-access-modes": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
					},
				},
			}),
		},
		"no-resource-requests": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
			}),
		},
		"invalid-resource-requests": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
					},
				},
			}),
		},
	}

	for name, scenario := range scenarios {
		errs := ValidatePersistentVolumeClaim(scenario.claim)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func TestValidatePersistentVolumeClaimUpdate(t *testing.T) {
	validClaim := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
		AccessModes: []api.PersistentVolumeAccessMode{
			api.ReadWriteOnce,
			api.ReadOnlyMany,
		},
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
		},
	})
	validUpdateClaim := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
		AccessModes: []api.PersistentVolumeAccessMode{
			api.ReadWriteOnce,
			api.ReadOnlyMany,
		},
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	invalidUpdateClaimResources := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
		AccessModes: []api.PersistentVolumeAccessMode{
			api.ReadWriteOnce,
			api.ReadOnlyMany,
		},
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("20G"),
			},
		},
		VolumeName: "volume",
	})
	invalidUpdateClaimAccessModes := testVolumeClaim("foo", "ns", api.PersistentVolumeClaimSpec{
		AccessModes: []api.PersistentVolumeAccessMode{
			api.ReadWriteOnce,
		},
		Resources: api.ResourceRequirements{
			Requests: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	scenarios := map[string]struct {
		isExpectedFailure bool
		oldClaim          *api.PersistentVolumeClaim
		newClaim          *api.PersistentVolumeClaim
	}{
		"valid-update": {
			isExpectedFailure: false,
			oldClaim:          validClaim,
			newClaim:          validUpdateClaim,
		},
		"invalid-update-change-resources-on-bound-claim": {
			isExpectedFailure: true,
			oldClaim:          validUpdateClaim,
			newClaim:          invalidUpdateClaimResources,
		},
		"invalid-update-change-access-modes-on-bound-claim": {
			isExpectedFailure: true,
			oldClaim:          validUpdateClaim,
			newClaim:          invalidUpdateClaimAccessModes,
		},
	}

	for name, scenario := range scenarios {
		// ensure we have a resource version specified for updates
		scenario.oldClaim.ResourceVersion = "1"
		scenario.newClaim.ResourceVersion = "1"
		errs := ValidatePersistentVolumeClaimUpdate(scenario.newClaim, scenario.oldClaim)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func TestValidateVolumes(t *testing.T) {
	lun := int32(1)
	successCase := []api.Volume{
		{Name: "abc", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/mnt/path1"}}},
		{Name: "123", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/mnt/path2"}}},
		{Name: "abc-123", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/mnt/path3"}}},
		{Name: "empty", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
		{Name: "gcepd", VolumeSource: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}},
		{Name: "awsebs", VolumeSource: api.VolumeSource{AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{VolumeID: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}},
		{Name: "gitrepo", VolumeSource: api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{Repository: "my-repo", Revision: "hashstring", Directory: "target"}}},
		{Name: "gitrepodot", VolumeSource: api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{Repository: "my-repo", Directory: "."}}},
		{Name: "iscsidisk", VolumeSource: api.VolumeSource{ISCSI: &api.ISCSIVolumeSource{TargetPortal: "127.0.0.1", IQN: "iqn.2015-02.example.com:test", Lun: 1, FSType: "ext4", ReadOnly: false}}},
		{Name: "secret", VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "my-secret"}}},
		{Name: "glusterfs", VolumeSource: api.VolumeSource{Glusterfs: &api.GlusterfsVolumeSource{EndpointsName: "host1", Path: "path", ReadOnly: false}}},
		{Name: "flocker", VolumeSource: api.VolumeSource{Flocker: &api.FlockerVolumeSource{DatasetName: "datasetName"}}},
		{Name: "rbd", VolumeSource: api.VolumeSource{RBD: &api.RBDVolumeSource{CephMonitors: []string{"foo"}, RBDImage: "bar", FSType: "ext4"}}},
		{Name: "cinder", VolumeSource: api.VolumeSource{Cinder: &api.CinderVolumeSource{VolumeID: "29ea5088-4f60-4757-962e-dba678767887", FSType: "ext4", ReadOnly: false}}},
		{Name: "cephfs", VolumeSource: api.VolumeSource{CephFS: &api.CephFSVolumeSource{Monitors: []string{"foo"}}}},
		{Name: "downwardapi", VolumeSource: api.VolumeSource{DownwardAPI: &api.DownwardAPIVolumeSource{Items: []api.DownwardAPIVolumeFile{
			{Path: "labels", FieldRef: api.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.labels"}},
			{Path: "annotations", FieldRef: api.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.annotations"}},
			{Path: "namespace", FieldRef: api.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.namespace"}},
			{Path: "name", FieldRef: api.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.name"}},
			{Path: "path/withslash/andslash", FieldRef: api.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.labels"}},
			{Path: "path/./withdot", FieldRef: api.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.labels"}},
			{Path: "path/with..dot", FieldRef: api.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.labels"}},
			{Path: "second-level-dirent-can-have/..dot", FieldRef: api.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.labels"}},
		}}}},
		{Name: "fc", VolumeSource: api.VolumeSource{FC: &api.FCVolumeSource{TargetWWNs: []string{"some_wwn"}, Lun: &lun, FSType: "ext4", ReadOnly: false}}},
		{Name: "flexvolume", VolumeSource: api.VolumeSource{FlexVolume: &api.FlexVolumeSource{Driver: "kubernetes.io/blue", FSType: "ext4"}}},
		{Name: "azure", VolumeSource: api.VolumeSource{AzureFile: &api.AzureFileVolumeSource{SecretName: "key", ShareName: "share", ReadOnly: false}}},
	}
	names, errs := validateVolumes(successCase, field.NewPath("field"))
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	if len(names) != len(successCase) || !names.HasAll("abc", "123", "abc-123", "empty", "gcepd", "gitrepo", "secret", "iscsidisk", "cinder", "cephfs", "flexvolume", "fc") {
		t.Errorf("wrong names result: %v", names)
	}
	emptyVS := api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}
	emptyPortal := api.VolumeSource{ISCSI: &api.ISCSIVolumeSource{TargetPortal: "", IQN: "iqn.2015-02.example.com:test", Lun: 1, FSType: "ext4", ReadOnly: false}}
	emptyIQN := api.VolumeSource{ISCSI: &api.ISCSIVolumeSource{TargetPortal: "127.0.0.1", IQN: "", Lun: 1, FSType: "ext4", ReadOnly: false}}
	emptyHosts := api.VolumeSource{Glusterfs: &api.GlusterfsVolumeSource{EndpointsName: "", Path: "path", ReadOnly: false}}
	emptyPath := api.VolumeSource{Glusterfs: &api.GlusterfsVolumeSource{EndpointsName: "host", Path: "", ReadOnly: false}}
	emptyName := api.VolumeSource{Flocker: &api.FlockerVolumeSource{DatasetName: ""}}
	emptyMon := api.VolumeSource{RBD: &api.RBDVolumeSource{CephMonitors: []string{}, RBDImage: "bar", FSType: "ext4"}}
	emptyImage := api.VolumeSource{RBD: &api.RBDVolumeSource{CephMonitors: []string{"foo"}, RBDImage: "", FSType: "ext4"}}
	emptyCephFSMon := api.VolumeSource{CephFS: &api.CephFSVolumeSource{Monitors: []string{}}}
	startsWithDots := api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{Repository: "foo", Directory: "..dots/bar"}}
	containsDots := api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{Repository: "foo", Directory: "dots/../bar"}}
	absPath := api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{Repository: "foo", Directory: "/abstarget"}}
	emptyPathName := api.VolumeSource{DownwardAPI: &api.DownwardAPIVolumeSource{Items: []api.DownwardAPIVolumeFile{{Path: "",
		FieldRef: api.ObjectFieldSelector{
			APIVersion: "v1",
			FieldPath:  "metadata.labels"}}},
	}}
	absolutePathName := api.VolumeSource{DownwardAPI: &api.DownwardAPIVolumeSource{Items: []api.DownwardAPIVolumeFile{{Path: "/absolutepath",
		FieldRef: api.ObjectFieldSelector{
			APIVersion: "v1",
			FieldPath:  "metadata.labels"}}},
	}}
	dotDotInPath := api.VolumeSource{DownwardAPI: &api.DownwardAPIVolumeSource{Items: []api.DownwardAPIVolumeFile{{Path: "../../passwd",
		FieldRef: api.ObjectFieldSelector{
			APIVersion: "v1",
			FieldPath:  "metadata.labels"}}},
	}}
	dotDotPathName := api.VolumeSource{DownwardAPI: &api.DownwardAPIVolumeSource{Items: []api.DownwardAPIVolumeFile{{Path: "..badFileName",
		FieldRef: api.ObjectFieldSelector{
			APIVersion: "v1",
			FieldPath:  "metadata.labels"}}},
	}}
	dotDotFirstLevelDirent := api.VolumeSource{DownwardAPI: &api.DownwardAPIVolumeSource{Items: []api.DownwardAPIVolumeFile{{Path: "..badDirName/goodFileName",
		FieldRef: api.ObjectFieldSelector{
			APIVersion: "v1",
			FieldPath:  "metadata.labels"}}},
	}}
	zeroWWN := api.VolumeSource{FC: &api.FCVolumeSource{TargetWWNs: []string{}, Lun: &lun, FSType: "ext4", ReadOnly: false}}
	emptyLun := api.VolumeSource{FC: &api.FCVolumeSource{TargetWWNs: []string{"wwn"}, Lun: nil, FSType: "ext4", ReadOnly: false}}
	slashInName := api.VolumeSource{Flocker: &api.FlockerVolumeSource{DatasetName: "foo/bar"}}
	emptyAzureSecret := api.VolumeSource{AzureFile: &api.AzureFileVolumeSource{SecretName: "", ShareName: "share", ReadOnly: false}}
	emptyAzureShare := api.VolumeSource{AzureFile: &api.AzureFileVolumeSource{SecretName: "name", ShareName: "", ReadOnly: false}}
	errorCases := map[string]struct {
		V []api.Volume
		T field.ErrorType
		F string
		D string
	}{
		"zero-length name": {
			[]api.Volume{{Name: "", VolumeSource: emptyVS}},
			field.ErrorTypeRequired,
			"name", "",
		},
		"name > 63 characters": {
			[]api.Volume{{Name: strings.Repeat("a", 64), VolumeSource: emptyVS}},
			field.ErrorTypeInvalid,
			"name", "must be a DNS label",
		},
		"name not a DNS label": {
			[]api.Volume{{Name: "a.b.c", VolumeSource: emptyVS}},
			field.ErrorTypeInvalid,
			"name", "must be a DNS label",
		},
		"name not unique": {
			[]api.Volume{{Name: "abc", VolumeSource: emptyVS}, {Name: "abc", VolumeSource: emptyVS}},
			field.ErrorTypeDuplicate,
			"[1].name", "",
		},
		"empty portal": {
			[]api.Volume{{Name: "badportal", VolumeSource: emptyPortal}},
			field.ErrorTypeRequired,
			"iscsi.targetPortal", "",
		},
		"empty iqn": {
			[]api.Volume{{Name: "badiqn", VolumeSource: emptyIQN}},
			field.ErrorTypeRequired,
			"iscsi.iqn", "",
		},
		"empty hosts": {
			[]api.Volume{{Name: "badhost", VolumeSource: emptyHosts}},
			field.ErrorTypeRequired,
			"glusterfs.endpoints", "",
		},
		"empty path": {
			[]api.Volume{{Name: "badpath", VolumeSource: emptyPath}},
			field.ErrorTypeRequired,
			"glusterfs.path", "",
		},
		"empty datasetName": {
			[]api.Volume{{Name: "badname", VolumeSource: emptyName}},
			field.ErrorTypeRequired,
			"flocker.datasetName", "",
		},
		"empty mon": {
			[]api.Volume{{Name: "badmon", VolumeSource: emptyMon}},
			field.ErrorTypeRequired,
			"rbd.monitors", "",
		},
		"empty image": {
			[]api.Volume{{Name: "badimage", VolumeSource: emptyImage}},
			field.ErrorTypeRequired,
			"rbd.image", "",
		},
		"empty cephfs mon": {
			[]api.Volume{{Name: "badmon", VolumeSource: emptyCephFSMon}},
			field.ErrorTypeRequired,
			"cephfs.monitors", "",
		},
		"empty metatada path": {
			[]api.Volume{{Name: "emptyname", VolumeSource: emptyPathName}},
			field.ErrorTypeRequired,
			"downwardAPI.path", "",
		},
		"absolute path": {
			[]api.Volume{{Name: "absolutepath", VolumeSource: absolutePathName}},
			field.ErrorTypeInvalid,
			"downwardAPI.path", "",
		},
		"dot dot path": {
			[]api.Volume{{Name: "dotdotpath", VolumeSource: dotDotInPath}},
			field.ErrorTypeInvalid,
			"downwardAPI.path", `must not contain '..'`,
		},
		"dot dot file name": {
			[]api.Volume{{Name: "dotdotfilename", VolumeSource: dotDotPathName}},
			field.ErrorTypeInvalid,
			"downwardAPI.path", `must not start with '..'`,
		},
		"dot dot first level dirent": {
			[]api.Volume{{Name: "dotdotdirfilename", VolumeSource: dotDotFirstLevelDirent}},
			field.ErrorTypeInvalid,
			"downwardAPI.path", `must not start with '..'`,
		},
		"empty wwn": {
			[]api.Volume{{Name: "badimage", VolumeSource: zeroWWN}},
			field.ErrorTypeRequired,
			"fc.targetWWNs", "",
		},
		"empty lun": {
			[]api.Volume{{Name: "badimage", VolumeSource: emptyLun}},
			field.ErrorTypeRequired,
			"fc.lun", "",
		},
		"slash in datasetName": {
			[]api.Volume{{Name: "slashinname", VolumeSource: slashInName}},
			field.ErrorTypeInvalid,
			"flocker.datasetName", "must not contain '/'",
		},
		"starts with '..'": {
			[]api.Volume{{Name: "badprefix", VolumeSource: startsWithDots}},
			field.ErrorTypeInvalid,
			"gitRepo.directory", `must not start with '..'`,
		},
		"contains '..'": {
			[]api.Volume{{Name: "containsdots", VolumeSource: containsDots}},
			field.ErrorTypeInvalid,
			"gitRepo.directory", `must not contain '..'`,
		},
		"absolute target": {
			[]api.Volume{{Name: "absolutetarget", VolumeSource: absPath}},
			field.ErrorTypeInvalid,
			"gitRepo.directory", "",
		},
		"empty secret": {
			[]api.Volume{{Name: "emptyaccount", VolumeSource: emptyAzureSecret}},
			field.ErrorTypeRequired,
			"azureFile.secretName", "",
		},
		"empty share": {
			[]api.Volume{{Name: "emptyaccount", VolumeSource: emptyAzureShare}},
			field.ErrorTypeRequired,
			"azureFile.shareName", "",
		},
	}
	for k, v := range errorCases {
		_, errs := validateVolumes(v.V, field.NewPath("field"))
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.V)
			continue
		}
		for i := range errs {
			if errs[i].Type != v.T {
				t.Errorf("%s: expected error to have type %q: %q", k, v.T, errs[i].Type)
			}
			if !strings.Contains(errs[i].Field, v.F) {
				t.Errorf("%s: expected error field %q: %q", k, v.F, errs[i].Field)
			}
			if !strings.Contains(errs[i].Detail, v.D) {
				t.Errorf("%s: expected error detail %q, got %q", k, v.D, errs[i].Detail)
			}
		}
	}
}

func TestValidatePorts(t *testing.T) {
	successCase := []api.ContainerPort{
		{Name: "abc", ContainerPort: 80, HostPort: 80, Protocol: "TCP"},
		{Name: "easy", ContainerPort: 82, Protocol: "TCP"},
		{Name: "as", ContainerPort: 83, Protocol: "UDP"},
		{Name: "do-re-me", ContainerPort: 84, Protocol: "UDP"},
		{ContainerPort: 85, Protocol: "TCP"},
	}
	if errs := validateContainerPorts(successCase, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	nonCanonicalCase := []api.ContainerPort{
		{ContainerPort: 80, Protocol: "TCP"},
	}
	if errs := validateContainerPorts(nonCanonicalCase, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		P []api.ContainerPort
		T field.ErrorType
		F string
		D string
	}{
		"name > 15 characters": {
			[]api.ContainerPort{{Name: strings.Repeat("a", 16), ContainerPort: 80, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"name", PortNameErrorMsg,
		},
		"name not a IANA svc name ": {
			[]api.ContainerPort{{Name: "a.b.c", ContainerPort: 80, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"name", PortNameErrorMsg,
		},
		"name not a IANA svc name (i.e. a number)": {
			[]api.ContainerPort{{Name: "80", ContainerPort: 80, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"name", PortNameErrorMsg,
		},
		"name not unique": {
			[]api.ContainerPort{
				{Name: "abc", ContainerPort: 80, Protocol: "TCP"},
				{Name: "abc", ContainerPort: 81, Protocol: "TCP"},
			},
			field.ErrorTypeDuplicate,
			"[1].name", "",
		},
		"zero container port": {
			[]api.ContainerPort{{ContainerPort: 0, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"containerPort", PortRangeErrorMsg,
		},
		"invalid container port": {
			[]api.ContainerPort{{ContainerPort: 65536, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"containerPort", PortRangeErrorMsg,
		},
		"invalid host port": {
			[]api.ContainerPort{{ContainerPort: 80, HostPort: 65536, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"hostPort", PortRangeErrorMsg,
		},
		"invalid protocol case": {
			[]api.ContainerPort{{ContainerPort: 80, Protocol: "tcp"}},
			field.ErrorTypeNotSupported,
			"protocol", "supported values: TCP, UDP",
		},
		"invalid protocol": {
			[]api.ContainerPort{{ContainerPort: 80, Protocol: "ICMP"}},
			field.ErrorTypeNotSupported,
			"protocol", "supported values: TCP, UDP",
		},
		"protocol required": {
			[]api.ContainerPort{{Name: "abc", ContainerPort: 80}},
			field.ErrorTypeRequired,
			"protocol", "",
		},
	}
	for k, v := range errorCases {
		errs := validateContainerPorts(v.P, field.NewPath("field"))
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			if errs[i].Type != v.T {
				t.Errorf("%s: expected error to have type %q: %q", k, v.T, errs[i].Type)
			}
			if !strings.Contains(errs[i].Field, v.F) {
				t.Errorf("%s: expected error field %q: %q", k, v.F, errs[i].Field)
			}
			if !strings.Contains(errs[i].Detail, v.D) {
				t.Errorf("%s: expected error detail %q, got %q", k, v.D, errs[i].Detail)
			}
		}
	}
}

func TestValidateEnv(t *testing.T) {
	successCase := []api.EnvVar{
		{Name: "abc", Value: "value"},
		{Name: "ABC", Value: "value"},
		{Name: "AbC_123", Value: "value"},
		{Name: "abc", Value: ""},
		{
			Name: "abc",
			ValueFrom: &api.EnvVarSource{
				FieldRef: &api.ObjectFieldSelector{
					APIVersion: testapi.Default.GroupVersion().String(),
					FieldPath:  "metadata.name",
				},
			},
		},
		{
			Name: "secret_value",
			ValueFrom: &api.EnvVarSource{
				SecretKeyRef: &api.SecretKeySelector{
					LocalObjectReference: api.LocalObjectReference{
						Name: "some-secret",
					},
					Key: "secret-key",
				},
			},
		},
		{
			Name: "ENV_VAR_1",
			ValueFrom: &api.EnvVarSource{
				ConfigMapKeyRef: &api.ConfigMapKeySelector{
					LocalObjectReference: api.LocalObjectReference{
						Name: "some-config-map",
					},
					Key: "some-key",
				},
			},
		},
	}
	if errs := validateEnv(successCase, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := []struct {
		name          string
		envs          []api.EnvVar
		expectedError string
	}{
		{
			name:          "zero-length name",
			envs:          []api.EnvVar{{Name: ""}},
			expectedError: "[0].name: Required value",
		},
		{
			name:          "name not a C identifier",
			envs:          []api.EnvVar{{Name: "a.b.c"}},
			expectedError: `[0].name: Invalid value: "a.b.c": must be a C identifier (matching regex [A-Za-z_][A-Za-z0-9_]*): e.g. "my_name" or "MyName"`,
		},
		{
			name: "value and valueFrom specified",
			envs: []api.EnvVar{{
				Name:  "abc",
				Value: "foo",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: testapi.Default.GroupVersion().String(),
						FieldPath:  "metadata.name",
					},
				},
			}},
			expectedError: "[0].valueFrom: Invalid value: \"\": may not be specified when `value` is not empty",
		},
		{
			name: "valueFrom.fieldRef and valueFrom.secretKeyRef specified",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: testapi.Default.GroupVersion().String(),
						FieldPath:  "metadata.name",
					},
					SecretKeyRef: &api.SecretKeySelector{
						LocalObjectReference: api.LocalObjectReference{
							Name: "a-secret",
						},
						Key: "a-key",
					},
				},
			}},
			expectedError: "[0].valueFrom: Invalid value: \"\": may not have more than one field specified at a time",
		},
		{
			name: "valueFrom.fieldRef and valueFrom.configMapKeyRef set",
			envs: []api.EnvVar{{
				Name: "some_var_name",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: testapi.Default.GroupVersion().String(),
						FieldPath:  "metadata.name",
					},
					ConfigMapKeyRef: &api.ConfigMapKeySelector{
						LocalObjectReference: api.LocalObjectReference{
							Name: "some-config-map",
						},
						Key: "some-key",
					},
				},
			}},
			expectedError: `[0].valueFrom: Invalid value: "": may not have more than one field specified at a time`,
		},
		{
			name: "valueFrom.fieldRef and valueFrom.secretKeyRef specified",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: testapi.Default.GroupVersion().String(),
						FieldPath:  "metadata.name",
					},
					SecretKeyRef: &api.SecretKeySelector{
						LocalObjectReference: api.LocalObjectReference{
							Name: "a-secret",
						},
						Key: "a-key",
					},
					ConfigMapKeyRef: &api.ConfigMapKeySelector{
						LocalObjectReference: api.LocalObjectReference{
							Name: "some-config-map",
						},
						Key: "some-key",
					},
				},
			}},
			expectedError: `[0].valueFrom: Invalid value: "": may not have more than one field specified at a time`,
		},
		{
			name: "missing FieldPath on ObjectFieldSelector",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: testapi.Default.GroupVersion().String(),
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.fieldPath: Required value`,
		},
		{
			name: "missing APIVersion on ObjectFieldSelector",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						FieldPath: "metadata.name",
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.apiVersion: Required value`,
		},
		{
			name: "invalid fieldPath",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						FieldPath:  "metadata.whoops",
						APIVersion: testapi.Default.GroupVersion().String(),
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.fieldPath: Invalid value: "metadata.whoops": error converting fieldPath`,
		},
		{
			name: "invalid fieldPath labels",
			envs: []api.EnvVar{{
				Name: "labels",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						FieldPath:  "metadata.labels",
						APIVersion: "v1",
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.fieldPath: Unsupported value: "metadata.labels": supported values: metadata.name, metadata.namespace, status.podIP`,
		},
		{
			name: "invalid fieldPath annotations",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						FieldPath:  "metadata.annotations",
						APIVersion: "v1",
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.fieldPath: Unsupported value: "metadata.annotations": supported values: metadata.name, metadata.namespace, status.podIP`,
		},
		{
			name: "unsupported fieldPath",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						FieldPath:  "status.phase",
						APIVersion: testapi.Default.GroupVersion().String(),
					},
				},
			}},
			expectedError: `valueFrom.fieldRef.fieldPath: Unsupported value: "status.phase": supported values: metadata.name, metadata.namespace, status.podIP`,
		},
	}
	for _, tc := range errorCases {
		if errs := validateEnv(tc.envs, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %s", tc.name)
		} else {
			for i := range errs {
				str := errs[i].Error()
				if str != "" && !strings.Contains(str, tc.expectedError) {
					t.Errorf("%s: expected error detail either empty or %q, got %q", tc.name, tc.expectedError, str)
				}
			}
		}
	}
}

func TestValidateVolumeMounts(t *testing.T) {
	volumes := sets.NewString("abc", "123", "abc-123")

	successCase := []api.VolumeMount{
		{Name: "abc", MountPath: "/foo"},
		{Name: "123", MountPath: "/bar"},
		{Name: "abc-123", MountPath: "/baz"},
		{Name: "abc-123", MountPath: "/baa", SubPath: ""},
		{Name: "abc-123", MountPath: "/bab", SubPath: "baz"},
		{Name: "abc-123", MountPath: "/bac", SubPath: ".baz"},
		{Name: "abc-123", MountPath: "/bad", SubPath: "..baz"},
	}
	if errs := validateVolumeMounts(successCase, volumes, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string][]api.VolumeMount{
		"empty name":          {{Name: "", MountPath: "/foo"}},
		"name not found":      {{Name: "", MountPath: "/foo"}},
		"empty mountpath":     {{Name: "abc", MountPath: ""}},
		"colon mountpath":     {{Name: "abc", MountPath: "foo:bar"}},
		"mountpath collision": {{Name: "foo", MountPath: "/path/a"}, {Name: "bar", MountPath: "/path/a"}},
		"absolute subpath":    {{Name: "abc", MountPath: "/bar", SubPath: "/baz"}},
		"subpath in ..":       {{Name: "abc", MountPath: "/bar", SubPath: "../baz"}},
		"subpath contains ..": {{Name: "abc", MountPath: "/bar", SubPath: "baz/../bat"}},
		"subpath ends in ..":  {{Name: "abc", MountPath: "/bar", SubPath: "./.."}},
	}
	for k, v := range errorCases {
		if errs := validateVolumeMounts(v, volumes, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateProbe(t *testing.T) {
	handler := api.Handler{Exec: &api.ExecAction{Command: []string{"echo"}}}
	// These fields must be positive.
	positiveFields := [...]string{"InitialDelaySeconds", "TimeoutSeconds", "PeriodSeconds", "SuccessThreshold", "FailureThreshold"}
	successCases := []*api.Probe{nil}
	for _, field := range positiveFields {
		probe := &api.Probe{Handler: handler}
		reflect.ValueOf(probe).Elem().FieldByName(field).SetInt(10)
		successCases = append(successCases, probe)
	}

	for _, p := range successCases {
		if errs := validateProbe(p, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []*api.Probe{{TimeoutSeconds: 10, InitialDelaySeconds: 10}}
	for _, field := range positiveFields {
		probe := &api.Probe{Handler: handler}
		reflect.ValueOf(probe).Elem().FieldByName(field).SetInt(-10)
		errorCases = append(errorCases, probe)
	}
	for _, p := range errorCases {
		if errs := validateProbe(p, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %v", p)
		}
	}
}

func TestValidateHandler(t *testing.T) {
	successCases := []api.Handler{
		{Exec: &api.ExecAction{Command: []string{"echo"}}},
		{HTTPGet: &api.HTTPGetAction{Path: "/", Port: intstr.FromInt(1), Host: "", Scheme: "HTTP"}},
		{HTTPGet: &api.HTTPGetAction{Path: "/foo", Port: intstr.FromInt(65535), Host: "host", Scheme: "HTTP"}},
		{HTTPGet: &api.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP"}},
		{HTTPGet: &api.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP", HTTPHeaders: []api.HTTPHeader{{"Host", "foo.example.com"}}}},
		{HTTPGet: &api.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP", HTTPHeaders: []api.HTTPHeader{{"X-Forwarded-For", "1.2.3.4"}, {"X-Forwarded-For", "5.6.7.8"}}}},
	}
	for _, h := range successCases {
		if errs := validateHandler(&h, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []api.Handler{
		{},
		{Exec: &api.ExecAction{Command: []string{}}},
		{HTTPGet: &api.HTTPGetAction{Path: "", Port: intstr.FromInt(0), Host: ""}},
		{HTTPGet: &api.HTTPGetAction{Path: "/foo", Port: intstr.FromInt(65536), Host: "host"}},
		{HTTPGet: &api.HTTPGetAction{Path: "", Port: intstr.FromString(""), Host: ""}},
		{HTTPGet: &api.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP", HTTPHeaders: []api.HTTPHeader{{"Host:", "foo.example.com"}}}},
		{HTTPGet: &api.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP", HTTPHeaders: []api.HTTPHeader{{"X_Forwarded_For", "foo.example.com"}}}},
	}
	for _, h := range errorCases {
		if errs := validateHandler(&h, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %#v", h)
		}
	}
}

func TestValidatePullPolicy(t *testing.T) {
	type T struct {
		Container      api.Container
		ExpectedPolicy api.PullPolicy
	}
	testCases := map[string]T{
		"NotPresent1": {
			api.Container{Name: "abc", Image: "image:latest", ImagePullPolicy: "IfNotPresent"},
			api.PullIfNotPresent,
		},
		"NotPresent2": {
			api.Container{Name: "abc1", Image: "image", ImagePullPolicy: "IfNotPresent"},
			api.PullIfNotPresent,
		},
		"Always1": {
			api.Container{Name: "123", Image: "image:latest", ImagePullPolicy: "Always"},
			api.PullAlways,
		},
		"Always2": {
			api.Container{Name: "1234", Image: "image", ImagePullPolicy: "Always"},
			api.PullAlways,
		},
		"Never1": {
			api.Container{Name: "abc-123", Image: "image:latest", ImagePullPolicy: "Never"},
			api.PullNever,
		},
		"Never2": {
			api.Container{Name: "abc-1234", Image: "image", ImagePullPolicy: "Never"},
			api.PullNever,
		},
	}
	for k, v := range testCases {
		ctr := &v.Container
		errs := validatePullPolicy(ctr.ImagePullPolicy, field.NewPath("field"))
		if len(errs) != 0 {
			t.Errorf("case[%s] expected success, got %#v", k, errs)
		}
		if ctr.ImagePullPolicy != v.ExpectedPolicy {
			t.Errorf("case[%s] expected policy %v, got %v", k, v.ExpectedPolicy, ctr.ImagePullPolicy)
		}
	}
}

func getResourceLimits(cpu, memory string) api.ResourceList {
	res := api.ResourceList{}
	res[api.ResourceCPU] = resource.MustParse(cpu)
	res[api.ResourceMemory] = resource.MustParse(memory)
	return res
}

func TestValidateContainers(t *testing.T) {
	volumes := sets.String{}
	capabilities.SetForTests(capabilities.Capabilities{
		AllowPrivileged: true,
	})

	successCase := []api.Container{
		{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"},
		{Name: "123", Image: "image", ImagePullPolicy: "IfNotPresent"},
		{Name: "abc-123", Image: "image", ImagePullPolicy: "IfNotPresent"},
		{
			Name:  "life-123",
			Image: "image",
			Lifecycle: &api.Lifecycle{
				PreStop: &api.Handler{
					Exec: &api.ExecAction{Command: []string{"ls", "-l"}},
				},
			},
			ImagePullPolicy: "IfNotPresent",
		},
		{
			Name:  "resources-test",
			Image: "image",
			Resources: api.ResourceRequirements{
				Limits: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
					api.ResourceName("my.org/resource"):  resource.MustParse("10m"),
				},
			},
			ImagePullPolicy: "IfNotPresent",
		},
		{
			Name:  "resources-request-limit-simple",
			Image: "image",
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceCPU): resource.MustParse("8"),
				},
				Limits: api.ResourceList{
					api.ResourceName(api.ResourceCPU): resource.MustParse("10"),
				},
			},
			ImagePullPolicy: "IfNotPresent",
		},
		{
			Name:  "resources-request-limit-edge",
			Image: "image",
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
					api.ResourceName("my.org/resource"):  resource.MustParse("10m"),
				},
				Limits: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
					api.ResourceName("my.org/resource"):  resource.MustParse("10m"),
				},
			},
			ImagePullPolicy: "IfNotPresent",
		},
		{
			Name:  "resources-request-limit-partials",
			Image: "image",
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("9.5"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
				},
				Limits: api.ResourceList{
					api.ResourceName(api.ResourceCPU):   resource.MustParse("10"),
					api.ResourceName("my.org/resource"): resource.MustParse("10m"),
				},
			},
			ImagePullPolicy: "IfNotPresent",
		},
		{
			Name:  "resources-request",
			Image: "image",
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("9.5"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
				},
			},
			ImagePullPolicy: "IfNotPresent",
		},
		{
			Name:  "same-host-port-different-protocol",
			Image: "image",
			Ports: []api.ContainerPort{
				{ContainerPort: 80, HostPort: 80, Protocol: "TCP"},
				{ContainerPort: 80, HostPort: 80, Protocol: "UDP"},
			},
			ImagePullPolicy: "IfNotPresent",
		},
		{Name: "abc-1234", Image: "image", ImagePullPolicy: "IfNotPresent", SecurityContext: fakeValidSecurityContext(true)},
	}
	if errs := validateContainers(successCase, volumes, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	capabilities.SetForTests(capabilities.Capabilities{
		AllowPrivileged: false,
	})
	errorCases := map[string][]api.Container{
		"zero-length name":     {{Name: "", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		"name > 63 characters": {{Name: strings.Repeat("a", 64), Image: "image", ImagePullPolicy: "IfNotPresent"}},
		"name not a DNS label": {{Name: "a.b.c", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		"name not unique": {
			{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"},
			{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"},
		},
		"zero-length image": {{Name: "abc", Image: "", ImagePullPolicy: "IfNotPresent"}},
		"host port not unique": {
			{Name: "abc", Image: "image", Ports: []api.ContainerPort{{ContainerPort: 80, HostPort: 80, Protocol: "TCP"}},
				ImagePullPolicy: "IfNotPresent"},
			{Name: "def", Image: "image", Ports: []api.ContainerPort{{ContainerPort: 81, HostPort: 80, Protocol: "TCP"}},
				ImagePullPolicy: "IfNotPresent"},
		},
		"invalid env var name": {
			{Name: "abc", Image: "image", Env: []api.EnvVar{{Name: "ev.1"}}, ImagePullPolicy: "IfNotPresent"},
		},
		"unknown volume name": {
			{Name: "abc", Image: "image", VolumeMounts: []api.VolumeMount{{Name: "anything", MountPath: "/foo"}},
				ImagePullPolicy: "IfNotPresent"},
		},
		"invalid lifecycle, no exec command.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &api.Lifecycle{
					PreStop: &api.Handler{
						Exec: &api.ExecAction{},
					},
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"invalid lifecycle, no http path.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &api.Lifecycle{
					PreStop: &api.Handler{
						HTTPGet: &api.HTTPGetAction{},
					},
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"invalid lifecycle, no tcp socket port.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &api.Lifecycle{
					PreStop: &api.Handler{
						TCPSocket: &api.TCPSocketAction{},
					},
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"invalid lifecycle, zero tcp socket port.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &api.Lifecycle{
					PreStop: &api.Handler{
						TCPSocket: &api.TCPSocketAction{
							Port: intstr.FromInt(0),
						},
					},
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"invalid lifecycle, no action.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &api.Lifecycle{
					PreStop: &api.Handler{},
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"invalid liveness probe, no tcp socket port.": {
			{
				Name:  "life-123",
				Image: "image",
				LivenessProbe: &api.Probe{
					Handler: api.Handler{
						TCPSocket: &api.TCPSocketAction{},
					},
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"invalid liveness probe, no action.": {
			{
				Name:  "life-123",
				Image: "image",
				LivenessProbe: &api.Probe{
					Handler: api.Handler{},
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"privilege disabled": {
			{Name: "abc", Image: "image", SecurityContext: fakeValidSecurityContext(true)},
		},
		"invalid compute resource": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: api.ResourceRequirements{
					Limits: api.ResourceList{
						"disk": resource.MustParse("10G"),
					},
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"Resource CPU invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: api.ResourceRequirements{
					Limits: getResourceLimits("-10", "0"),
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"Resource Requests CPU invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: api.ResourceRequirements{
					Requests: getResourceLimits("-10", "0"),
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"Resource Memory invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: api.ResourceRequirements{
					Limits: getResourceLimits("0", "-10"),
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"Request limit simple invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: api.ResourceRequirements{
					Limits:   getResourceLimits("5", "3"),
					Requests: getResourceLimits("6", "3"),
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
		"Request limit multiple invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: api.ResourceRequirements{
					Limits:   getResourceLimits("5", "3"),
					Requests: getResourceLimits("6", "4"),
				},
				ImagePullPolicy: "IfNotPresent",
			},
		},
	}
	for k, v := range errorCases {
		if errs := validateContainers(v, volumes, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateRestartPolicy(t *testing.T) {
	successCases := []api.RestartPolicy{
		api.RestartPolicyAlways,
		api.RestartPolicyOnFailure,
		api.RestartPolicyNever,
	}
	for _, policy := range successCases {
		if errs := validateRestartPolicy(&policy, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []api.RestartPolicy{"", "newpolicy"}

	for k, policy := range errorCases {
		if errs := validateRestartPolicy(&policy, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %d", k)
		}
	}
}

func TestValidateDNSPolicy(t *testing.T) {
	successCases := []api.DNSPolicy{api.DNSClusterFirst, api.DNSDefault, api.DNSPolicy(api.DNSClusterFirst)}
	for _, policy := range successCases {
		if errs := validateDNSPolicy(&policy, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []api.DNSPolicy{api.DNSPolicy("invalid")}
	for _, policy := range errorCases {
		if errs := validateDNSPolicy(&policy, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %v", policy)
		}
	}
}

func TestValidatePodSpec(t *testing.T) {
	activeDeadlineSeconds := int64(30)
	minID := int64(0)
	maxID := int64(2147483647)
	successCases := []api.PodSpec{
		{ // Populate basic fields, leave defaults for most.
			Volumes:       []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		{ // Populate all fields.
			Volumes: []api.Volume{
				{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			RestartPolicy: api.RestartPolicyAlways,
			NodeSelector: map[string]string{
				"key": "value",
			},
			NodeName:              "foobar",
			DNSPolicy:             api.DNSClusterFirst,
			ActiveDeadlineSeconds: &activeDeadlineSeconds,
			ServiceAccountName:    "acct",
		},
		{ // Populate HostNetwork.
			Containers: []api.Container{
				{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", Ports: []api.ContainerPort{
					{HostPort: 8080, ContainerPort: 8080, Protocol: "TCP"}},
				},
			},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		{ // Populate RunAsUser SupplementalGroups FSGroup with minID 0
			Containers: []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			SecurityContext: &api.PodSecurityContext{
				SupplementalGroups: []int64{minID},
				RunAsUser:          &minID,
				FSGroup:            &minID,
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		{ // Populate RunAsUser SupplementalGroups FSGroup with maxID 2147483647
			Containers: []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			SecurityContext: &api.PodSecurityContext{
				SupplementalGroups: []int64{maxID},
				RunAsUser:          &maxID,
				FSGroup:            &maxID,
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		{ // Populate HostIPC.
			SecurityContext: &api.PodSecurityContext{
				HostIPC: true,
			},
			Volumes:       []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		{ // Populate HostPID.
			SecurityContext: &api.PodSecurityContext{
				HostPID: true,
			},
			Volumes:       []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		{ // Populate Affinity.
			Volumes:       []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
	}
	for i := range successCases {
		if errs := ValidatePodSpec(&successCases[i], field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	activeDeadlineSeconds = int64(0)
	minID = int64(-1)
	maxID = int64(2147483648)
	failureCases := map[string]api.PodSpec{
		"bad volume": {
			Volumes:       []api.Volume{{}},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
		"no containers": {
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		"bad container": {
			Containers:    []api.Container{{}},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		"bad DNS policy": {
			DNSPolicy:     api.DNSPolicy("invalid"),
			RestartPolicy: api.RestartPolicyAlways,
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
		"bad service account name": {
			Containers:         []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			RestartPolicy:      api.RestartPolicyAlways,
			DNSPolicy:          api.DNSClusterFirst,
			ServiceAccountName: "invalidName",
		},
		"bad restart policy": {
			RestartPolicy: "UnknowPolicy",
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
		"with hostNetwork hostPort not equal to containerPort": {
			Containers: []api.Container{
				{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", Ports: []api.ContainerPort{
					{HostPort: 8080, ContainerPort: 2600, Protocol: "TCP"}},
				},
			},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		"bad supplementalGroups large than math.MaxInt32": {
			Containers: []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork:        false,
				SupplementalGroups: []int64{maxID, 1234},
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		"bad supplementalGroups less than 0": {
			Containers: []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork:        false,
				SupplementalGroups: []int64{minID, 1234},
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		"bad runAsUser large than math.MaxInt32": {
			Containers: []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: false,
				RunAsUser:   &maxID,
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		"bad runAsUser less than 0": {
			Containers: []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: false,
				RunAsUser:   &minID,
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		"bad fsGroup large than math.MaxInt32": {
			Containers: []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: false,
				FSGroup:     &maxID,
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		"bad fsGroup less than 0": {
			Containers: []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: false,
				FSGroup:     &minID,
			},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
		"bad-active-deadline-seconds": {
			Volumes: []api.Volume{
				{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			RestartPolicy: api.RestartPolicyAlways,
			NodeSelector: map[string]string{
				"key": "value",
			},
			NodeName:              "foobar",
			DNSPolicy:             api.DNSClusterFirst,
			ActiveDeadlineSeconds: &activeDeadlineSeconds,
		},
		"bad nodeName": {
			NodeName:      "node name",
			Volumes:       []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
			Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,
		},
	}
	for k, v := range failureCases {
		if errs := ValidatePodSpec(&v, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		}
	}
}

func TestValidatePod(t *testing.T) {
	successCases := []api.Pod{
		{ // Basic fields.
			ObjectMeta: api.ObjectMeta{Name: "123", Namespace: "ns"},
			Spec: api.PodSpec{
				Volumes:       []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
		},
		{ // Just about everything.
			ObjectMeta: api.ObjectMeta{Name: "abc.123.do-re-mi", Namespace: "ns"},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				},
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				NodeSelector: map[string]string{
					"key": "value",
				},
				NodeName: "foobar",
			},
		},
		{ // Serialized affinity requirements in annotations.
			ObjectMeta: api.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				// TODO: Uncomment and move this block into Annotations map once
				// RequiredDuringSchedulingRequiredDuringExecution is implemented
				//					"requiredDuringSchedulingRequiredDuringExecution": {
				//						"nodeSelectorTerms": [{
				//							"matchExpressions": [{
				//								"key": "key1",
				//								"operator": "Exists"
				//							}]
				//						}]
				//					},
				Annotations: map[string]string{
					api.AffinityAnnotationKey: `
					{"nodeAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "key2",
									"operator": "In",
									"values": ["value1", "value2"]
								}]
							}]
						},
						"preferredDuringSchedulingIgnoredDuringExecution": [
							{
								"weight": 10,
								"preference": {"matchExpressions": [
									{
										"key": "foo",
										"operator": "In", "values": ["bar"]
									}
								]}
							}
						]
					}}`,
				},
			},
			Spec: api.PodSpec{
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
		},
	}
	for _, pod := range successCases {
		if errs := ValidatePod(&pod); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]api.Pod{
		"bad name": {
			ObjectMeta: api.ObjectMeta{Name: "", Namespace: "ns"},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
		"bad namespace": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: ""},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
		"bad spec": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "ns"},
			Spec: api.PodSpec{
				Containers: []api.Container{{}},
			},
		},
		"bad label": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc",
				Namespace: "ns",
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
		"invalid json of affinity in pod annotations": {
			ObjectMeta: api.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					api.AffinityAnnotationKey: `
					{"nodeAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
					`,
				},
			},
			Spec: api.PodSpec{
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
		},
		"invalid node selector requirement in affinity in pod annotations, operator can't be null": {
			ObjectMeta: api.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					api.AffinityAnnotationKey: `
					{"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {
						"nodeSelectorTerms": [{
							"matchExpressions": [{
								"key": "key1",
							}]
						}]
					}}}`,
				},
			},
			Spec: api.PodSpec{
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
		},
		"invalid preferredSchedulingTerm in affinity in pod annotations, weight should be in range 1-100": {
			ObjectMeta: api.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					api.AffinityAnnotationKey: `
					{"nodeAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": [
						{
							"weight": 199,
							"preference": {"matchExpressions": [
								{
									"key": "foo",
									"operator": "In",
									"values": ["bar"]
								}
							]}
						}
					]}}`,
				},
			},
			Spec: api.PodSpec{
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
		},
		"invalid requiredDuringSchedulingIgnoredDuringExecution node selector, nodeSelectorTerms must have at least one term": {
			ObjectMeta: api.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					api.AffinityAnnotationKey: `
					{"nodeAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": []
						},
					}}`,
				},
			},
			Spec: api.PodSpec{
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
		},
		"invalid requiredDuringSchedulingIgnoredDuringExecution node selector term, matchExpressions must have at least one node selector requirement": {
			ObjectMeta: api.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					api.AffinityAnnotationKey: `
					{"nodeAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": []
							}]
						},
					}}`,
				},
			},
			Spec: api.PodSpec{
				Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
		},
	}
	for k, v := range errorCases {
		if errs := ValidatePod(&v); len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		}
	}
}

func TestValidatePodUpdate(t *testing.T) {
	var (
		activeDeadlineSecondsZero     = int64(0)
		activeDeadlineSecondsNegative = int64(-30)
		activeDeadlineSecondsPositive = int64(30)
		activeDeadlineSecondsLarger   = int64(31)

		now    = unversioned.Now()
		grace  = int64(30)
		grace2 = int64(31)
	)

	tests := []struct {
		a       api.Pod
		b       api.Pod
		isValid bool
		test    string
	}{
		{api.Pod{}, api.Pod{}, true, "nothing"},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "bar"},
			},
			false,
			"ids",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"bar": "foo",
					},
				},
			},
			true,
			"labels",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Annotations: map[string]string{
						"foo": "bar",
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Annotations: map[string]string{
						"bar": "foo",
					},
				},
			},
			true,
			"annotations",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V1",
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V2",
						},
						{
							Image: "bar:V2",
						},
					},
				},
			},
			false,
			"more containers",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec:       api.PodSpec{Containers: []api.Container{{Image: "foo:V1"}}},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo", DeletionTimestamp: &now},
				Spec:       api.PodSpec{Containers: []api.Container{{Image: "foo:V1"}}},
			},
			true,
			"deletion timestamp filled out",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &grace},
				Spec:       api.PodSpec{Containers: []api.Container{{Image: "foo:V1"}}},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &grace2},
				Spec:       api.PodSpec{Containers: []api.Container{{Image: "foo:V1"}}},
			},
			false,
			"deletion grace period seconds cleared",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V1",
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V2",
						},
					},
				},
			},
			true,
			"image change",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V2",
						},
					},
				},
			},
			false,
			"image change to empty",
		},
		{
			api.Pod{
				Spec: api.PodSpec{},
			},
			api.Pod{
				Spec: api.PodSpec{},
			},
			true,
			"activeDeadlineSeconds no change, nil",
		},
		{
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			true,
			"activeDeadlineSeconds no change, set",
		},
		{
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			api.Pod{},
			true,
			"activeDeadlineSeconds change to positive from nil",
		},
		{
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsLarger,
				},
			},
			true,
			"activeDeadlineSeconds change to smaller positive",
		},
		{
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsLarger,
				},
			},
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			false,
			"activeDeadlineSeconds change to larger positive",
		},

		{
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsNegative,
				},
			},
			api.Pod{},
			false,
			"activeDeadlineSeconds change to negative from nil",
		},
		{
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsNegative,
				},
			},
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			false,
			"activeDeadlineSeconds change to negative from positive",
		},
		{
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsZero,
				},
			},
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			true,
			"activeDeadlineSeconds change to zero from positive",
		},
		{
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsZero,
				},
			},
			api.Pod{},
			true,
			"activeDeadlineSeconds change to zero from nil",
		},
		{
			api.Pod{},
			api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			false,
			"activeDeadlineSeconds change to nil from positive",
		},

		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V1",
							Resources: api.ResourceRequirements{
								Limits: getResourceLimits("100m", "0"),
							},
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V2",
							Resources: api.ResourceRequirements{
								Limits: getResourceLimits("1000m", "0"),
							},
						},
					},
				},
			},
			false,
			"cpu change",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V1",
							Ports: []api.ContainerPort{
								{HostPort: 8080, ContainerPort: 80},
							},
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V2",
							Ports: []api.ContainerPort{
								{HostPort: 8000, ContainerPort: 80},
							},
						},
					},
				},
			},
			false,
			"port change",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"Bar": "foo",
					},
				},
			},
			true,
			"bad label change",
		},
	}

	for _, test := range tests {
		test.a.ObjectMeta.ResourceVersion = "1"
		test.b.ObjectMeta.ResourceVersion = "1"
		errs := ValidatePodUpdate(&test.a, &test.b)
		if test.isValid {
			if len(errs) != 0 {
				t.Errorf("unexpected invalid: %s (%+v)\nA: %+v\nB: %+v", test.test, errs, test.a, test.b)
			}
		} else {
			if len(errs) == 0 {
				t.Errorf("unexpected valid: %s\nA: %+v\nB: %+v", test.test, test.a, test.b)
			}
		}
	}
}

func makeValidService() api.Service {
	return api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:            "valid",
			Namespace:       "valid",
			Labels:          map[string]string{},
			Annotations:     map[string]string{},
			ResourceVersion: "1",
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"key": "val"},
			SessionAffinity: "None",
			Type:            api.ServiceTypeClusterIP,
			Ports:           []api.ServicePort{{Name: "p", Protocol: "TCP", Port: 8675, TargetPort: intstr.FromInt(8675)}},
		},
	}
}

func TestValidateService(t *testing.T) {
	testCases := []struct {
		name     string
		tweakSvc func(svc *api.Service) // given a basic valid service, each test case can customize it
		numErrs  int
	}{
		{
			name: "missing namespace",
			tweakSvc: func(s *api.Service) {
				s.Namespace = ""
			},
			numErrs: 1,
		},
		{
			name: "invalid namespace",
			tweakSvc: func(s *api.Service) {
				s.Namespace = "-123"
			},
			numErrs: 1,
		},
		{
			name: "missing name",
			tweakSvc: func(s *api.Service) {
				s.Name = ""
			},
			numErrs: 1,
		},
		{
			name: "invalid name",
			tweakSvc: func(s *api.Service) {
				s.Name = "-123"
			},
			numErrs: 1,
		},
		{
			name: "too long name",
			tweakSvc: func(s *api.Service) {
				s.Name = strings.Repeat("a", 25)
			},
			numErrs: 1,
		},
		{
			name: "invalid generateName",
			tweakSvc: func(s *api.Service) {
				s.GenerateName = "-123"
			},
			numErrs: 1,
		},
		{
			name: "too long generateName",
			tweakSvc: func(s *api.Service) {
				s.GenerateName = strings.Repeat("a", 25)
			},
			numErrs: 1,
		},
		{
			name: "invalid label",
			tweakSvc: func(s *api.Service) {
				s.Labels["NoUppercaseOrSpecialCharsLike=Equals"] = "bar"
			},
			numErrs: 1,
		},
		{
			name: "invalid annotation",
			tweakSvc: func(s *api.Service) {
				s.Annotations["NoSpecialCharsLike=Equals"] = "bar"
			},
			numErrs: 1,
		},
		{
			name: "nil selector",
			tweakSvc: func(s *api.Service) {
				s.Spec.Selector = nil
			},
			numErrs: 0,
		},
		{
			name: "invalid selector",
			tweakSvc: func(s *api.Service) {
				s.Spec.Selector["NoSpecialCharsLike=Equals"] = "bar"
			},
			numErrs: 1,
		},
		{
			name: "missing session affinity",
			tweakSvc: func(s *api.Service) {
				s.Spec.SessionAffinity = ""
			},
			numErrs: 1,
		},
		{
			name: "missing type",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = ""
			},
			numErrs: 1,
		},
		{
			name: "missing ports",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports = nil
			},
			numErrs: 1,
		},
		{
			name: "missing ports but headless",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports = nil
				s.Spec.ClusterIP = api.ClusterIPNone
			},
			numErrs: 0,
		},
		{
			name: "empty port[0] name",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Name = ""
			},
			numErrs: 0,
		},
		{
			name: "empty port[1] name",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "", Protocol: "TCP", Port: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "empty multi-port port[0] name",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Name = ""
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "p", Protocol: "TCP", Port: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "invalid port name",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Name = "INVALID"
			},
			numErrs: 1,
		},
		{
			name: "missing protocol",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Protocol = ""
			},
			numErrs: 1,
		},
		{
			name: "invalid protocol",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Protocol = "INVALID"
			},
			numErrs: 1,
		},
		{
			name: "invalid cluster ip",
			tweakSvc: func(s *api.Service) {
				s.Spec.ClusterIP = "invalid"
			},
			numErrs: 1,
		},
		{
			name: "missing port",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Port = 0
			},
			numErrs: 1,
		},
		{
			name: "invalid port",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Port = 65536
			},
			numErrs: 1,
		},
		{
			name: "invalid TargetPort int",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].TargetPort = intstr.FromInt(65536)
			},
			numErrs: 1,
		},
		{
			name: "valid port headless",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Port = 11722
				s.Spec.Ports[0].TargetPort = intstr.FromInt(11722)
				s.Spec.ClusterIP = api.ClusterIPNone
			},
			numErrs: 0,
		},
		{
			name: "invalid port headless 1",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Port = 11722
				s.Spec.Ports[0].TargetPort = intstr.FromInt(11721)
				s.Spec.ClusterIP = api.ClusterIPNone
			},
			// in the v1 API, targetPorts on headless services were tolerated.
			// once we have version-specific validation, we can reject this on newer API versions, but until then, we have to tolerate it for compatibility.
			// numErrs: 1,
			numErrs: 0,
		},
		{
			name: "invalid port headless 2",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Port = 11722
				s.Spec.Ports[0].TargetPort = intstr.FromString("target")
				s.Spec.ClusterIP = api.ClusterIPNone
			},
			// in the v1 API, targetPorts on headless services were tolerated.
			// once we have version-specific validation, we can reject this on newer API versions, but until then, we have to tolerate it for compatibility.
			// numErrs: 1,
			numErrs: 0,
		},
		{
			name: "invalid publicIPs localhost",
			tweakSvc: func(s *api.Service) {
				s.Spec.ExternalIPs = []string{"127.0.0.1"}
			},
			numErrs: 1,
		},
		{
			name: "invalid publicIPs",
			tweakSvc: func(s *api.Service) {
				s.Spec.ExternalIPs = []string{"0.0.0.0"}
			},
			numErrs: 1,
		},
		{
			name: "invalid publicIPs host",
			tweakSvc: func(s *api.Service) {
				s.Spec.ExternalIPs = []string{"myhost.mydomain"}
			},
			numErrs: 1,
		},
		{
			name: "dup port name",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Name = "p"
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "p", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "valid load balancer protocol UDP 1",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports[0].Protocol = "UDP"
			},
			numErrs: 0,
		},
		{
			name: "valid load balancer protocol UDP 2",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports[0] = api.ServicePort{Name: "q", Port: 12345, Protocol: "UDP", TargetPort: intstr.FromInt(12345)}
			},
			numErrs: 0,
		},
		{
			name: "invalid load balancer with mix protocol",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "UDP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "valid 1",
			tweakSvc: func(s *api.Service) {
				// do nothing
			},
			numErrs: 0,
		},
		{
			name: "valid 2",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Protocol = "UDP"
				s.Spec.Ports[0].TargetPort = intstr.FromInt(12345)
			},
			numErrs: 0,
		},
		{
			name: "valid 3",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].TargetPort = intstr.FromString("http")
			},
			numErrs: 0,
		},
		{
			name: "valid cluster ip - none ",
			tweakSvc: func(s *api.Service) {
				s.Spec.ClusterIP = "None"
			},
			numErrs: 0,
		},
		{
			name: "valid cluster ip - empty",
			tweakSvc: func(s *api.Service) {
				s.Spec.ClusterIP = ""
				s.Spec.Ports[0].TargetPort = intstr.FromString("http")
			},
			numErrs: 0,
		},
		{
			name: "valid type - cluster",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeClusterIP
			},
			numErrs: 0,
		},
		{
			name: "valid type - loadbalancer",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
			},
			numErrs: 0,
		},
		{
			name: "valid type loadbalancer 2 ports",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid external load balancer 2 ports",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "duplicate nodeports",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(1)})
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "r", Port: 2, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(2)})
			},
			numErrs: 1,
		},
		{
			name: "duplicate nodeports (different protocols)",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(1)})
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "r", Port: 2, Protocol: "UDP", NodePort: 1, TargetPort: intstr.FromInt(2)})
			},
			numErrs: 0,
		},
		{
			name: "valid type - cluster",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeClusterIP
			},
			numErrs: 0,
		},
		{
			name: "valid type - nodeport",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
			},
			numErrs: 0,
		},
		{
			name: "valid type - loadbalancer",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
			},
			numErrs: 0,
		},
		{
			name: "valid type loadbalancer 2 ports",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid type loadbalancer with NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", NodePort: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid type=NodePort service with NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", NodePort: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid type=NodePort service without NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid cluster service without NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeClusterIP
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "invalid cluster service with NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeClusterIP
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", NodePort: 12345, TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "invalid public service with duplicate NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "p1", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(1)})
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "p2", Port: 2, Protocol: "TCP", NodePort: 1, TargetPort: intstr.FromInt(2)})
			},
			numErrs: 1,
		},
		{
			name: "valid type=LoadBalancer",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 0,
		},
		{
			// For now we open firewalls, and its insecure if we open 10250, remove this
			// when we have better protections in place.
			name: "invalid port type=LoadBalancer",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "kubelet", Port: 10250, Protocol: "TCP", TargetPort: intstr.FromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "valid LoadBalancer source range annotation",
			tweakSvc: func(s *api.Service) {
				s.Annotations[service.AnnotationLoadBalancerSourceRangesKey] = "1.2.3.4/8,  5.6.7.8/16"
			},
			numErrs: 0,
		},
		{
			name: "empty LoadBalancer source range annotation",
			tweakSvc: func(s *api.Service) {
				s.Annotations[service.AnnotationLoadBalancerSourceRangesKey] = ""
			},
			numErrs: 0,
		},
		{
			name: "invalid LoadBalancer source range annotation (hostname)",
			tweakSvc: func(s *api.Service) {
				s.Annotations[service.AnnotationLoadBalancerSourceRangesKey] = "foo.bar"
			},
			numErrs: 1,
		},
		{
			name: "invalid LoadBalancer source range annotation (invalid CIDR)",
			tweakSvc: func(s *api.Service) {
				s.Annotations[service.AnnotationLoadBalancerSourceRangesKey] = "1.2.3.4/33"
			},
			numErrs: 1,
		},
	}

	for _, tc := range testCases {
		svc := makeValidService()
		tc.tweakSvc(&svc)
		errs := ValidateService(&svc)
		if len(errs) != tc.numErrs {
			t.Errorf("Unexpected error list for case %q: %v", tc.name, errs.ToAggregate())
		}
	}
}

func TestValidateReplicationControllerStatusUpdate(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	type rcUpdateTest struct {
		old    api.ReplicationController
		update api.ReplicationController
	}
	successCases := []rcUpdateTest{
		{
			old: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
				Status: api.ReplicationControllerStatus{
					Replicas: 2,
				},
			},
			update: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Replicas: 3,
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
				Status: api.ReplicationControllerStatus{
					Replicas: 4,
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateReplicationControllerStatusUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]rcUpdateTest{
		"negative replicas": {
			old: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
				Status: api.ReplicationControllerStatus{
					Replicas: 3,
				},
			},
			update: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
				Status: api.ReplicationControllerStatus{
					Replicas: -3,
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		if errs := ValidateReplicationControllerStatusUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}

}

func TestValidateReplicationControllerUpdate(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	readWriteVolumePodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
				Volumes:       []api.Volume{{Name: "gcepd", VolumeSource: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}}},
			},
		},
	}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
			ObjectMeta: api.ObjectMeta{
				Labels: invalidSelector,
			},
		},
	}
	type rcUpdateTest struct {
		old    api.ReplicationController
		update api.ReplicationController
	}
	successCases := []rcUpdateTest{
		{
			old: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Replicas: 3,
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
		},
		{
			old: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Replicas: 1,
					Selector: validSelector,
					Template: &readWriteVolumePodTemplate.Template,
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateReplicationControllerUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]rcUpdateTest{
		"more than one read/write": {
			old: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Selector: validSelector,
					Template: &readWriteVolumePodTemplate.Template,
				},
			},
		},
		"invalid selector": {
			old: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Selector: invalidSelector,
					Template: &validPodTemplate.Template,
				},
			},
		},
		"invalid pod": {
			old: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Selector: validSelector,
					Template: &invalidPodTemplate.Template,
				},
			},
		},
		"negative replicas": {
			old: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
			update: api.ReplicationController{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: api.ReplicationControllerSpec{
					Replicas: -1,
					Selector: validSelector,
					Template: &validPodTemplate.Template,
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		if errs := ValidateReplicationControllerUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}

func TestValidateReplicationController(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	readWriteVolumePodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				Volumes:       []api.Volume{{Name: "gcepd", VolumeSource: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}}},
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
			ObjectMeta: api.ObjectMeta{
				Labels: invalidSelector,
			},
		},
	}
	successCases := []api.ReplicationController{
		{
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Replicas: 1,
				Selector: validSelector,
				Template: &readWriteVolumePodTemplate.Template,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateReplicationController(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]api.ReplicationController{
		"zero-length ID": {
			ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"missing-namespace": {
			ObjectMeta: api.ObjectMeta{Name: "abc-123"},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"empty selector": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Template: &validPodTemplate.Template,
			},
		},
		"selector_doesnt_match": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Selector: map[string]string{"foo": "bar"},
				Template: &validPodTemplate.Template,
			},
		},
		"invalid manifest": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
			},
		},
		"read-write persistent disk with > 1 pod": {
			ObjectMeta: api.ObjectMeta{Name: "abc"},
			Spec: api.ReplicationControllerSpec{
				Replicas: 2,
				Selector: validSelector,
				Template: &readWriteVolumePodTemplate.Template,
			},
		},
		"negative_replicas": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Replicas: -1,
				Selector: validSelector,
			},
		},
		"invalid_label": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"invalid_label 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: api.ReplicationControllerSpec{
				Template: &invalidPodTemplate.Template,
			},
		},
		"invalid_annotation": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Annotations: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"invalid restart policy 1": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
		"invalid restart policy 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyNever,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateReplicationController(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].Field
			if !strings.HasPrefix(field, "spec.template.") &&
				field != "metadata.name" &&
				field != "metadata.namespace" &&
				field != "spec.selector" &&
				field != "spec.template" &&
				field != "GCEPersistentDisk.ReadOnly" &&
				field != "spec.replicas" &&
				field != "spec.template.labels" &&
				field != "metadata.annotations" &&
				field != "metadata.labels" &&
				field != "status.replicas" {
				t.Errorf("%s: missing prefix for: %v", k, errs[i])
			}
		}
	}
}

func TestValidateNode(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	successCases := []api.Node{
		{
			ObjectMeta: api.ObjectMeta{
				Name:   "abc",
				Labels: validSelector,
			},
			Status: api.NodeStatus{
				Addresses: []api.NodeAddress{
					{Type: api.NodeLegacyHostIP, Address: "something"},
				},
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
					api.ResourceName("my.org/gpu"):       resource.MustParse("10"),
				},
			},
			Spec: api.NodeSpec{
				ExternalID: "external",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name: "abc",
			},
			Status: api.NodeStatus{
				Addresses: []api.NodeAddress{
					{Type: api.NodeLegacyHostIP, Address: "something"},
				},
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("0"),
				},
			},
			Spec: api.NodeSpec{
				ExternalID: "external",
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateNode(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]api.Node{
		"zero-length Name": {
			ObjectMeta: api.ObjectMeta{
				Name:   "",
				Labels: validSelector,
			},
			Status: api.NodeStatus{
				Addresses: []api.NodeAddress{},
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
				},
			},
			Spec: api.NodeSpec{
				ExternalID: "external",
			},
		},
		"invalid-labels": {
			ObjectMeta: api.ObjectMeta{
				Name:   "abc-123",
				Labels: invalidSelector,
			},
			Status: api.NodeStatus{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
				},
			},
			Spec: api.NodeSpec{
				ExternalID: "external",
			},
		},
		"missing-external-id": {
			ObjectMeta: api.ObjectMeta{
				Name:   "abc-123",
				Labels: validSelector,
			},
			Status: api.NodeStatus{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceCPU):    resource.MustParse("10"),
					api.ResourceName(api.ResourceMemory): resource.MustParse("10G"),
				},
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateNode(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].Field
			expectedFields := map[string]bool{
				"metadata.name":        true,
				"metadata.labels":      true,
				"metadata.annotations": true,
				"metadata.namespace":   true,
				"spec.externalID":      true,
			}
			if expectedFields[field] == false {
				t.Errorf("%s: missing prefix for: %v", k, errs[i])
			}
		}
	}
}

func TestValidateNodeUpdate(t *testing.T) {
	tests := []struct {
		oldNode api.Node
		node    api.Node
		valid   bool
	}{
		{api.Node{}, api.Node{}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo"}},
			api.Node{
				ObjectMeta: api.ObjectMeta{
					Name: "bar"},
			}, false},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "bar"},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				PodCIDR: "",
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				PodCIDR: "192.168.0.0/16",
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				PodCIDR: "192.123.0.0/16",
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				PodCIDR: "192.168.0.0/16",
			},
		}, false},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Status: api.NodeStatus{
				Capacity: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("10000"),
					api.ResourceMemory: resource.MustParse("100"),
				},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Status: api.NodeStatus{
				Capacity: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("100"),
					api.ResourceMemory: resource.MustParse("10000"),
				},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
			Status: api.NodeStatus{
				Capacity: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("10000"),
					api.ResourceMemory: resource.MustParse("100"),
				},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "fooobaz"},
			},
			Status: api.NodeStatus{
				Capacity: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("100"),
					api.ResourceMemory: resource.MustParse("10000"),
				},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
			Status: api.NodeStatus{
				Addresses: []api.NodeAddress{
					{Type: api.NodeLegacyHostIP, Address: "1.2.3.4"},
				},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "fooobaz"},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"Foo": "baz"},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				Unschedulable: false,
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				Unschedulable: true,
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				Unschedulable: false,
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Status: api.NodeStatus{
				Addresses: []api.NodeAddress{
					{Type: api.NodeExternalIP, Address: "1.1.1.1"},
					{Type: api.NodeExternalIP, Address: "1.1.1.1"},
				},
			},
		}, false},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				Unschedulable: false,
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Status: api.NodeStatus{
				Addresses: []api.NodeAddress{
					{Type: api.NodeExternalIP, Address: "1.1.1.1"},
					{Type: api.NodeInternalIP, Address: "10.1.1.1"},
				},
			},
		}, true},
	}
	for i, test := range tests {
		test.oldNode.ObjectMeta.ResourceVersion = "1"
		test.node.ObjectMeta.ResourceVersion = "1"
		errs := ValidateNodeUpdate(&test.node, &test.oldNode)
		if test.valid && len(errs) > 0 {
			t.Errorf("%d: Unexpected error: %v", i, errs)
			t.Logf("%#v vs %#v", test.oldNode.ObjectMeta, test.node.ObjectMeta)
		}
		if !test.valid && len(errs) == 0 {
			t.Errorf("%d: Unexpected non-error", i)
		}
	}
}

func TestValidateServiceUpdate(t *testing.T) {
	testCases := []struct {
		name     string
		tweakSvc func(oldSvc, newSvc *api.Service) // given basic valid services, each test case can customize them
		numErrs  int
	}{
		{
			name: "no change",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				// do nothing
			},
			numErrs: 0,
		},
		{
			name: "change name",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Name += "2"
			},
			numErrs: 1,
		},
		{
			name: "change namespace",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Namespace += "2"
			},
			numErrs: 1,
		},
		{
			name: "change label valid",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Labels["key"] = "other-value"
			},
			numErrs: 0,
		},
		{
			name: "add label",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Labels["key2"] = "value2"
			},
			numErrs: 0,
		},
		{
			name: "change cluster IP",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "8.6.7.5"
			},
			numErrs: 1,
		},
		{
			name: "remove cluster IP",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = ""
			},
			numErrs: 1,
		},
		{
			name: "change affinity",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Spec.SessionAffinity = "ClientIP"
			},
			numErrs: 0,
		},
		{
			name: "remove affinity",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Spec.SessionAffinity = ""
			},
			numErrs: 1,
		},
		{
			name: "change type",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Spec.Type = api.ServiceTypeLoadBalancer
			},
			numErrs: 0,
		},
		{
			name: "remove type",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Spec.Type = ""
			},
			numErrs: 1,
		},
		{
			name: "change type -> nodeport",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Spec.Type = api.ServiceTypeNodePort
			},
			numErrs: 0,
		},
	}

	for _, tc := range testCases {
		oldSvc := makeValidService()
		newSvc := makeValidService()
		tc.tweakSvc(&oldSvc, &newSvc)
		errs := ValidateServiceUpdate(&newSvc, &oldSvc)
		if len(errs) != tc.numErrs {
			t.Errorf("Unexpected error list for case %q: %v", tc.name, errs.ToAggregate())
		}
	}
}

func TestValidateResourceNames(t *testing.T) {
	table := []struct {
		input   string
		success bool
	}{
		{"memory", true},
		{"cpu", true},
		{"network", false},
		{"disk", false},
		{"", false},
		{".", false},
		{"..", false},
		{"my.favorite.app.co/12345", true},
		{"my.favorite.app.co/_12345", false},
		{"my.favorite.app.co/12345_", false},
		{"kubernetes.io/..", false},
		{"kubernetes.io/" + strings.Repeat("a", 63), true},
		{"kubernetes.io/" + strings.Repeat("a", 64), false},
		{"kubernetes.io//", false},
		{"kubernetes.io", false},
		{"kubernetes.io/will/not/work/", false},
	}
	for k, item := range table {
		err := validateResourceName(item.input, field.NewPath("field"))
		if len(err) != 0 && item.success {
			t.Errorf("expected no failure for input %q", item.input)
		} else if len(err) == 0 && !item.success {
			t.Errorf("expected failure for input %q", item.input)
			for i := range err {
				detail := err[i].Detail
				if detail != "" && detail != qualifiedNameErrorMsg {
					t.Errorf("%d: expected error detail either empty or %s, got %s", k, qualifiedNameErrorMsg, detail)
				}
			}
		}
	}
}

func getResourceList(cpu, memory string) api.ResourceList {
	res := api.ResourceList{}
	if cpu != "" {
		res[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[api.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func getStorageResourceList(storage string) api.ResourceList {
	res := api.ResourceList{}
	if storage != "" {
		res[api.ResourceStorage] = resource.MustParse(storage)
	}
	return res
}

func TestValidateLimitRange(t *testing.T) {
	successCases := []struct {
		name string
		spec api.LimitRangeSpec
	}{
		{
			name: "all-fields-valid",
			spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:                 api.LimitTypePod,
						Max:                  getResourceList("100m", "10000Mi"),
						Min:                  getResourceList("5m", "100Mi"),
						MaxLimitRequestRatio: getResourceList("10", ""),
					},
					{
						Type:                 api.LimitTypeContainer,
						Max:                  getResourceList("100m", "10000Mi"),
						Min:                  getResourceList("5m", "100Mi"),
						Default:              getResourceList("50m", "500Mi"),
						DefaultRequest:       getResourceList("10m", "200Mi"),
						MaxLimitRequestRatio: getResourceList("10", ""),
					},
				},
			},
		},
		{
			name: "all-fields-valid-big-numbers",
			spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:                 api.LimitTypeContainer,
						Max:                  getResourceList("100m", "10000T"),
						Min:                  getResourceList("5m", "100Mi"),
						Default:              getResourceList("50m", "500Mi"),
						DefaultRequest:       getResourceList("10m", "200Mi"),
						MaxLimitRequestRatio: getResourceList("10", ""),
					},
				},
			},
		},
		{
			name: "thirdparty-fields-all-valid-standard-container-resources",
			spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:                 "thirdparty.com/foo",
						Max:                  getResourceList("100m", "10000T"),
						Min:                  getResourceList("5m", "100Mi"),
						Default:              getResourceList("50m", "500Mi"),
						DefaultRequest:       getResourceList("10m", "200Mi"),
						MaxLimitRequestRatio: getResourceList("10", ""),
					},
				},
			},
		},
		{
			name: "thirdparty-fields-all-valid-storage-resources",
			spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:                 "thirdparty.com/foo",
						Max:                  getStorageResourceList("10000T"),
						Min:                  getStorageResourceList("100Mi"),
						Default:              getStorageResourceList("500Mi"),
						DefaultRequest:       getStorageResourceList("200Mi"),
						MaxLimitRequestRatio: getStorageResourceList(""),
					},
				},
			},
		},
	}

	for _, successCase := range successCases {
		limitRange := &api.LimitRange{ObjectMeta: api.ObjectMeta{Name: successCase.name, Namespace: "foo"}, Spec: successCase.spec}
		if errs := ValidateLimitRange(limitRange); len(errs) != 0 {
			t.Errorf("Case %v, unexpected error: %v", successCase.name, errs)
		}
	}

	errorCases := map[string]struct {
		R api.LimitRange
		D string
	}{
		"zero-length-name": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "", Namespace: "foo"}, Spec: api.LimitRangeSpec{}},
			"name or generateName is required",
		},
		"zero-length-namespace": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: ""}, Spec: api.LimitRangeSpec{}},
			"",
		},
		"invalid-name": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "^Invalid", Namespace: "foo"}, Spec: api.LimitRangeSpec{}},
			DNSSubdomainErrorMsg,
		},
		"invalid-namespace": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "^Invalid"}, Spec: api.LimitRangeSpec{}},
			DNS1123LabelErrorMsg,
		},
		"duplicate-limit-type": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type: api.LimitTypePod,
						Max:  getResourceList("100m", "10000m"),
						Min:  getResourceList("0m", "100m"),
					},
					{
						Type: api.LimitTypePod,
						Min:  getResourceList("0m", "100m"),
					},
				},
			}},
			"",
		},
		"default-limit-type-pod": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:    api.LimitTypePod,
						Max:     getResourceList("100m", "10000m"),
						Min:     getResourceList("0m", "100m"),
						Default: getResourceList("10m", "100m"),
					},
				},
			}},
			"may not be specified when `type` is 'Pod'",
		},
		"default-request-limit-type-pod": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:           api.LimitTypePod,
						Max:            getResourceList("100m", "10000m"),
						Min:            getResourceList("0m", "100m"),
						DefaultRequest: getResourceList("10m", "100m"),
					},
				},
			}},
			"may not be specified when `type` is 'Pod'",
		},
		"min value 100m is greater than max value 10m": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type: api.LimitTypePod,
						Max:  getResourceList("10m", ""),
						Min:  getResourceList("100m", ""),
					},
				},
			}},
			"min value 100m is greater than max value 10m",
		},
		"invalid spec default outside range": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:    api.LimitTypeContainer,
						Max:     getResourceList("1", ""),
						Min:     getResourceList("100m", ""),
						Default: getResourceList("2000m", ""),
					},
				},
			}},
			"default value 2 is greater than max value 1",
		},
		"invalid spec defaultrequest outside range": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:           api.LimitTypeContainer,
						Max:            getResourceList("1", ""),
						Min:            getResourceList("100m", ""),
						DefaultRequest: getResourceList("2000m", ""),
					},
				},
			}},
			"default request value 2 is greater than max value 1",
		},
		"invalid spec defaultrequest more than default": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:           api.LimitTypeContainer,
						Max:            getResourceList("2", ""),
						Min:            getResourceList("100m", ""),
						Default:        getResourceList("500m", ""),
						DefaultRequest: getResourceList("800m", ""),
					},
				},
			}},
			"default request value 800m is greater than default limit value 500m",
		},
		"invalid spec maxLimitRequestRatio less than 1": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:                 api.LimitTypePod,
						MaxLimitRequestRatio: getResourceList("800m", ""),
					},
				},
			}},
			"ratio 800m is less than 1",
		},
		"invalid spec maxLimitRequestRatio greater than max/min": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:                 api.LimitTypeContainer,
						Max:                  getResourceList("", "2Gi"),
						Min:                  getResourceList("", "512Mi"),
						MaxLimitRequestRatio: getResourceList("", "10"),
					},
				},
			}},
			"ratio 10 is greater than max/min = 4.000000",
		},
		"invalid non standard limit type": {
			api.LimitRange{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: api.LimitRangeSpec{
				Limits: []api.LimitRangeItem{
					{
						Type:                 "foo",
						Max:                  getStorageResourceList("10000T"),
						Min:                  getStorageResourceList("100Mi"),
						Default:              getStorageResourceList("500Mi"),
						DefaultRequest:       getStorageResourceList("200Mi"),
						MaxLimitRequestRatio: getStorageResourceList(""),
					},
				},
			}},
			"must be a standard limit type or fully qualified",
		},
	}

	for k, v := range errorCases {
		errs := ValidateLimitRange(&v.R)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			detail := errs[i].Detail
			if detail != v.D {
				t.Errorf("[%s]: expected error detail either empty or %q, got %q", k, v.D, detail)
			}
		}
	}

}

func TestValidateResourceQuota(t *testing.T) {
	spec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceCPU:                    resource.MustParse("100"),
			api.ResourceMemory:                 resource.MustParse("10000"),
			api.ResourceRequestsCPU:            resource.MustParse("100"),
			api.ResourceRequestsMemory:         resource.MustParse("10000"),
			api.ResourceLimitsCPU:              resource.MustParse("100"),
			api.ResourceLimitsMemory:           resource.MustParse("10000"),
			api.ResourcePods:                   resource.MustParse("10"),
			api.ResourceServices:               resource.MustParse("0"),
			api.ResourceReplicationControllers: resource.MustParse("10"),
			api.ResourceQuotas:                 resource.MustParse("10"),
			api.ResourceConfigMaps:             resource.MustParse("10"),
			api.ResourceSecrets:                resource.MustParse("10"),
		},
	}

	terminatingSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceCPU:       resource.MustParse("100"),
			api.ResourceLimitsCPU: resource.MustParse("200"),
		},
		Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeTerminating},
	}

	nonTerminatingSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceCPU: resource.MustParse("100"),
		},
		Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeNotTerminating},
	}

	bestEffortSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourcePods: resource.MustParse("100"),
		},
		Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeBestEffort},
	}

	nonBestEffortSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceCPU: resource.MustParse("100"),
		},
		Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeNotBestEffort},
	}

	// storage is not yet supported as a quota tracked resource
	invalidQuotaResourceSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceStorage: resource.MustParse("10"),
		},
	}

	negativeSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceCPU:                    resource.MustParse("-100"),
			api.ResourceMemory:                 resource.MustParse("-10000"),
			api.ResourcePods:                   resource.MustParse("-10"),
			api.ResourceServices:               resource.MustParse("-10"),
			api.ResourceReplicationControllers: resource.MustParse("-10"),
			api.ResourceQuotas:                 resource.MustParse("-10"),
			api.ResourceConfigMaps:             resource.MustParse("-10"),
			api.ResourceSecrets:                resource.MustParse("-10"),
		},
	}

	fractionalComputeSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceCPU: resource.MustParse("100m"),
		},
	}

	fractionalPodSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourcePods:                   resource.MustParse(".1"),
			api.ResourceServices:               resource.MustParse(".5"),
			api.ResourceReplicationControllers: resource.MustParse("1.25"),
			api.ResourceQuotas:                 resource.MustParse("2.5"),
		},
	}

	invalidTerminatingScopePairsSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceCPU: resource.MustParse("100"),
		},
		Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeTerminating, api.ResourceQuotaScopeNotTerminating},
	}

	invalidBestEffortScopePairsSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourcePods: resource.MustParse("100"),
		},
		Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeBestEffort, api.ResourceQuotaScopeNotBestEffort},
	}

	invalidScopeNameSpec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceCPU: resource.MustParse("100"),
		},
		Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScope("foo")},
	}

	successCases := []api.ResourceQuota{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: spec,
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: fractionalComputeSpec,
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: terminatingSpec,
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: nonTerminatingSpec,
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: bestEffortSpec,
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: nonBestEffortSpec,
		},
	}

	for _, successCase := range successCases {
		if errs := ValidateResourceQuota(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]struct {
		R api.ResourceQuota
		D string
	}{
		"zero-length Name": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "", Namespace: "foo"}, Spec: spec},
			"name or generateName is required",
		},
		"zero-length Namespace": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: ""}, Spec: spec},
			"",
		},
		"invalid Name": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "^Invalid", Namespace: "foo"}, Spec: spec},
			DNSSubdomainErrorMsg,
		},
		"invalid Namespace": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "^Invalid"}, Spec: spec},
			DNS1123LabelErrorMsg,
		},
		"negative-limits": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: negativeSpec},
			isNegativeErrorMsg,
		},
		"fractional-api-resource": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: fractionalPodSpec},
			isNotIntegerErrorMsg,
		},
		"invalid-quota-resource": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: invalidQuotaResourceSpec},
			isInvalidQuotaResource,
		},
		"invalid-quota-terminating-pair": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: invalidTerminatingScopePairsSpec},
			"conflicting scopes",
		},
		"invalid-quota-besteffort-pair": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: invalidBestEffortScopePairsSpec},
			"conflicting scopes",
		},
		"invalid-quota-scope-name": {
			api.ResourceQuota{ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: invalidScopeNameSpec},
			"unsupported scope",
		},
	}
	for k, v := range errorCases {
		errs := ValidateResourceQuota(&v.R)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			if errs[i].Detail != v.D {
				t.Errorf("[%s]: expected error detail either empty or %s, got %s", k, v.D, errs[i].Detail)
			}
		}
	}
}

func TestValidateNamespace(t *testing.T) {
	validLabels := map[string]string{"a": "b"}
	invalidLabels := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	successCases := []api.Namespace{
		{
			ObjectMeta: api.ObjectMeta{Name: "abc", Labels: validLabels},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123"},
			Spec: api.NamespaceSpec{
				Finalizers: []api.FinalizerName{"example.com/something", "example.com/other"},
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateNamespace(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]struct {
		R api.Namespace
		D string
	}{
		"zero-length name": {
			api.Namespace{ObjectMeta: api.ObjectMeta{Name: ""}},
			"",
		},
		"defined-namespace": {
			api.Namespace{ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: "makesnosense"}},
			"",
		},
		"invalid-labels": {
			api.Namespace{ObjectMeta: api.ObjectMeta{Name: "abc", Labels: invalidLabels}},
			"",
		},
	}
	for k, v := range errorCases {
		errs := ValidateNamespace(&v.R)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateNamespaceFinalizeUpdate(t *testing.T) {
	tests := []struct {
		oldNamespace api.Namespace
		namespace    api.Namespace
		valid        bool
	}{
		{api.Namespace{}, api.Namespace{}, true},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name: "foo"}},
			api.Namespace{
				ObjectMeta: api.ObjectMeta{
					Name: "foo"},
				Spec: api.NamespaceSpec{
					Finalizers: []api.FinalizerName{"Foo"},
				},
			}, false},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name: "foo"},
			Spec: api.NamespaceSpec{
				Finalizers: []api.FinalizerName{"foo.com/bar"},
			},
		},
			api.Namespace{
				ObjectMeta: api.ObjectMeta{
					Name: "foo"},
				Spec: api.NamespaceSpec{
					Finalizers: []api.FinalizerName{"foo.com/bar", "what.com/bar"},
				},
			}, true},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name: "fooemptyfinalizer"},
			Spec: api.NamespaceSpec{
				Finalizers: []api.FinalizerName{"foo.com/bar"},
			},
		},
			api.Namespace{
				ObjectMeta: api.ObjectMeta{
					Name: "fooemptyfinalizer"},
				Spec: api.NamespaceSpec{
					Finalizers: []api.FinalizerName{"", "foo.com/bar", "what.com/bar"},
				},
			}, false},
	}
	for i, test := range tests {
		test.namespace.ObjectMeta.ResourceVersion = "1"
		test.oldNamespace.ObjectMeta.ResourceVersion = "1"
		errs := ValidateNamespaceFinalizeUpdate(&test.namespace, &test.oldNamespace)
		if test.valid && len(errs) > 0 {
			t.Errorf("%d: Unexpected error: %v", i, errs)
			t.Logf("%#v vs %#v", test.oldNamespace, test.namespace)
		}
		if !test.valid && len(errs) == 0 {
			t.Errorf("%d: Unexpected non-error", i)
		}
	}
}

func TestValidateNamespaceStatusUpdate(t *testing.T) {
	now := unversioned.Now()

	tests := []struct {
		oldNamespace api.Namespace
		namespace    api.Namespace
		valid        bool
	}{
		{api.Namespace{}, api.Namespace{
			Status: api.NamespaceStatus{
				Phase: api.NamespaceActive,
			},
		}, true},
		// Cannot set deletionTimestamp via status update
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name: "foo"}},
			api.Namespace{
				ObjectMeta: api.ObjectMeta{
					Name:              "foo",
					DeletionTimestamp: &now},
				Status: api.NamespaceStatus{
					Phase: api.NamespaceTerminating,
				},
			}, false},
		// Can update phase via status update
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:              "foo",
				DeletionTimestamp: &now}},
			api.Namespace{
				ObjectMeta: api.ObjectMeta{
					Name:              "foo",
					DeletionTimestamp: &now},
				Status: api.NamespaceStatus{
					Phase: api.NamespaceTerminating,
				},
			}, true},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name: "foo"}},
			api.Namespace{
				ObjectMeta: api.ObjectMeta{
					Name: "foo"},
				Status: api.NamespaceStatus{
					Phase: api.NamespaceTerminating,
				},
			}, false},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name: "foo"}},
			api.Namespace{
				ObjectMeta: api.ObjectMeta{
					Name: "bar"},
				Status: api.NamespaceStatus{
					Phase: api.NamespaceTerminating,
				},
			}, false},
	}
	for i, test := range tests {
		test.namespace.ObjectMeta.ResourceVersion = "1"
		test.oldNamespace.ObjectMeta.ResourceVersion = "1"
		errs := ValidateNamespaceStatusUpdate(&test.namespace, &test.oldNamespace)
		if test.valid && len(errs) > 0 {
			t.Errorf("%d: Unexpected error: %v", i, errs)
			t.Logf("%#v vs %#v", test.oldNamespace.ObjectMeta, test.namespace.ObjectMeta)
		}
		if !test.valid && len(errs) == 0 {
			t.Errorf("%d: Unexpected non-error", i)
		}
	}
}

func TestValidateNamespaceUpdate(t *testing.T) {
	tests := []struct {
		oldNamespace api.Namespace
		namespace    api.Namespace
		valid        bool
	}{
		{api.Namespace{}, api.Namespace{}, true},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name: "foo1"}},
			api.Namespace{
				ObjectMeta: api.ObjectMeta{
					Name: "bar1"},
			}, false},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo2",
				Labels: map[string]string{"foo": "bar"},
			},
		}, api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo2",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name: "foo3",
			},
		}, api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo3",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo4",
				Labels: map[string]string{"bar": "foo"},
			},
		}, api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo4",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo5",
				Labels: map[string]string{"foo": "baz"},
			},
		}, api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo5",
				Labels: map[string]string{"Foo": "baz"},
			},
		}, true},
		{api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo6",
				Labels: map[string]string{"foo": "baz"},
			},
		}, api.Namespace{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo6",
				Labels: map[string]string{"Foo": "baz"},
			},
			Spec: api.NamespaceSpec{
				Finalizers: []api.FinalizerName{"kubernetes"},
			},
			Status: api.NamespaceStatus{
				Phase: api.NamespaceTerminating,
			},
		}, true},
	}
	for i, test := range tests {
		test.namespace.ObjectMeta.ResourceVersion = "1"
		test.oldNamespace.ObjectMeta.ResourceVersion = "1"
		errs := ValidateNamespaceUpdate(&test.namespace, &test.oldNamespace)
		if test.valid && len(errs) > 0 {
			t.Errorf("%d: Unexpected error: %v", i, errs)
			t.Logf("%#v vs %#v", test.oldNamespace.ObjectMeta, test.namespace.ObjectMeta)
		}
		if !test.valid && len(errs) == 0 {
			t.Errorf("%d: Unexpected non-error", i)
		}
	}
}

func TestValidateSecret(t *testing.T) {
	// Opaque secret validation
	validSecret := func() api.Secret {
		return api.Secret{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Data: map[string][]byte{
				"data-1": []byte("bar"),
			},
		}
	}

	var (
		emptyName     = validSecret()
		invalidName   = validSecret()
		emptyNs       = validSecret()
		invalidNs     = validSecret()
		overMaxSize   = validSecret()
		invalidKey    = validSecret()
		leadingDotKey = validSecret()
		dotKey        = validSecret()
		doubleDotKey  = validSecret()
	)

	emptyName.Name = ""
	invalidName.Name = "NoUppercaseOrSpecialCharsLike=Equals"
	emptyNs.Namespace = ""
	invalidNs.Namespace = "NoUppercaseOrSpecialCharsLike=Equals"
	overMaxSize.Data = map[string][]byte{
		"over": make([]byte, api.MaxSecretSize+1),
	}
	invalidKey.Data["a..b"] = []byte("whoops")
	leadingDotKey.Data[".key"] = []byte("bar")
	dotKey.Data["."] = []byte("bar")
	doubleDotKey.Data[".."] = []byte("bar")

	// kubernetes.io/service-account-token secret validation
	validServiceAccountTokenSecret := func() api.Secret {
		return api.Secret{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: "bar",
				Annotations: map[string]string{
					api.ServiceAccountNameKey: "foo",
				},
			},
			Type: api.SecretTypeServiceAccountToken,
			Data: map[string][]byte{
				"data-1": []byte("bar"),
			},
		}
	}

	var (
		emptyTokenAnnotation    = validServiceAccountTokenSecret()
		missingTokenAnnotation  = validServiceAccountTokenSecret()
		missingTokenAnnotations = validServiceAccountTokenSecret()
	)
	emptyTokenAnnotation.Annotations[api.ServiceAccountNameKey] = ""
	delete(missingTokenAnnotation.Annotations, api.ServiceAccountNameKey)
	missingTokenAnnotations.Annotations = nil

	tests := map[string]struct {
		secret api.Secret
		valid  bool
	}{
		"valid":                                     {validSecret(), true},
		"empty name":                                {emptyName, false},
		"invalid name":                              {invalidName, false},
		"empty namespace":                           {emptyNs, false},
		"invalid namespace":                         {invalidNs, false},
		"over max size":                             {overMaxSize, false},
		"invalid key":                               {invalidKey, false},
		"valid service-account-token secret":        {validServiceAccountTokenSecret(), true},
		"empty service-account-token annotation":    {emptyTokenAnnotation, false},
		"missing service-account-token annotation":  {missingTokenAnnotation, false},
		"missing service-account-token annotations": {missingTokenAnnotations, false},
		"leading dot key":                           {leadingDotKey, true},
		"dot key":                                   {dotKey, false},
		"double dot key":                            {doubleDotKey, false},
	}

	for name, tc := range tests {
		errs := ValidateSecret(&tc.secret)
		if tc.valid && len(errs) > 0 {
			t.Errorf("%v: Unexpected error: %v", name, errs)
		}
		if !tc.valid && len(errs) == 0 {
			t.Errorf("%v: Unexpected non-error", name)
		}
	}
}

func TestValidateDockerConfigSecret(t *testing.T) {
	validDockerSecret := func() api.Secret {
		return api.Secret{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Type:       api.SecretTypeDockercfg,
			Data: map[string][]byte{
				api.DockerConfigKey: []byte(`{"https://index.docker.io/v1/": {"auth": "Y2x1ZWRyb29sZXIwMDAxOnBhc3N3b3Jk","email": "fake@example.com"}}`),
			},
		}
	}
	validDockerSecret2 := func() api.Secret {
		return api.Secret{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Type:       api.SecretTypeDockerConfigJson,
			Data: map[string][]byte{
				api.DockerConfigJsonKey: []byte(`{"auths":{"https://index.docker.io/v1/": {"auth": "Y2x1ZWRyb29sZXIwMDAxOnBhc3N3b3Jk","email": "fake@example.com"}}}`),
			},
		}
	}

	var (
		missingDockerConfigKey  = validDockerSecret()
		emptyDockerConfigKey    = validDockerSecret()
		invalidDockerConfigKey  = validDockerSecret()
		missingDockerConfigKey2 = validDockerSecret2()
		emptyDockerConfigKey2   = validDockerSecret2()
		invalidDockerConfigKey2 = validDockerSecret2()
	)

	delete(missingDockerConfigKey.Data, api.DockerConfigKey)
	emptyDockerConfigKey.Data[api.DockerConfigKey] = []byte("")
	invalidDockerConfigKey.Data[api.DockerConfigKey] = []byte("bad")
	delete(missingDockerConfigKey2.Data, api.DockerConfigJsonKey)
	emptyDockerConfigKey2.Data[api.DockerConfigJsonKey] = []byte("")
	invalidDockerConfigKey2.Data[api.DockerConfigJsonKey] = []byte("bad")

	tests := map[string]struct {
		secret api.Secret
		valid  bool
	}{
		"valid dockercfg":     {validDockerSecret(), true},
		"missing dockercfg":   {missingDockerConfigKey, false},
		"empty dockercfg":     {emptyDockerConfigKey, false},
		"invalid dockercfg":   {invalidDockerConfigKey, false},
		"valid config.json":   {validDockerSecret2(), true},
		"missing config.json": {missingDockerConfigKey2, false},
		"empty config.json":   {emptyDockerConfigKey2, false},
		"invalid config.json": {invalidDockerConfigKey2, false},
	}

	for name, tc := range tests {
		errs := ValidateSecret(&tc.secret)
		if tc.valid && len(errs) > 0 {
			t.Errorf("%v: Unexpected error: %v", name, errs)
		}
		if !tc.valid && len(errs) == 0 {
			t.Errorf("%v: Unexpected non-error", name)
		}
	}
}

func TestValidateBasicAuthSecret(t *testing.T) {
	validBasicAuthSecret := func() api.Secret {
		return api.Secret{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Type:       api.SecretTypeBasicAuth,
			Data: map[string][]byte{
				api.BasicAuthUsernameKey: []byte("username"),
				api.BasicAuthPasswordKey: []byte("password"),
			},
		}
	}

	var (
		missingBasicAuthUsernamePasswordKeys = validBasicAuthSecret()
		// invalidBasicAuthUsernamePasswordKey  = validBasicAuthSecret()
		// emptyBasicAuthUsernameKey            = validBasicAuthSecret()
		// emptyBasicAuthPasswordKey            = validBasicAuthSecret()
	)

	delete(missingBasicAuthUsernamePasswordKeys.Data, api.BasicAuthUsernameKey)
	delete(missingBasicAuthUsernamePasswordKeys.Data, api.BasicAuthPasswordKey)

	// invalidBasicAuthUsernamePasswordKey.Data[api.BasicAuthUsernameKey] = []byte("bad")
	// invalidBasicAuthUsernamePasswordKey.Data[api.BasicAuthPasswordKey] = []byte("bad")

	// emptyBasicAuthUsernameKey.Data[api.BasicAuthUsernameKey] = []byte("")
	// emptyBasicAuthPasswordKey.Data[api.BasicAuthPasswordKey] = []byte("")

	tests := map[string]struct {
		secret api.Secret
		valid  bool
	}{
		"valid": {validBasicAuthSecret(), true},
		"missing username and password": {missingBasicAuthUsernamePasswordKeys, false},
		// "invalid username and password": {invalidBasicAuthUsernamePasswordKey, false},
		// "empty username":   {emptyBasicAuthUsernameKey, false},
		// "empty password":   {emptyBasicAuthPasswordKey, false},
	}

	for name, tc := range tests {
		errs := ValidateSecret(&tc.secret)
		if tc.valid && len(errs) > 0 {
			t.Errorf("%v: Unexpected error: %v", name, errs)
		}
		if !tc.valid && len(errs) == 0 {
			t.Errorf("%v: Unexpected non-error", name)
		}
	}
}

func TestValidateSSHAuthSecret(t *testing.T) {
	validSSHAuthSecret := func() api.Secret {
		return api.Secret{
			ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "bar"},
			Type:       api.SecretTypeSSHAuth,
			Data: map[string][]byte{
				api.SSHAuthPrivateKey: []byte("foo-bar-baz"),
			},
		}
	}

	missingSSHAuthPrivateKey := validSSHAuthSecret()

	delete(missingSSHAuthPrivateKey.Data, api.SSHAuthPrivateKey)

	tests := map[string]struct {
		secret api.Secret
		valid  bool
	}{
		"valid":               {validSSHAuthSecret(), true},
		"missing private key": {missingSSHAuthPrivateKey, false},
	}

	for name, tc := range tests {
		errs := ValidateSecret(&tc.secret)
		if tc.valid && len(errs) > 0 {
			t.Errorf("%v: Unexpected error: %v", name, errs)
		}
		if !tc.valid && len(errs) == 0 {
			t.Errorf("%v: Unexpected non-error", name)
		}
	}
}

func TestValidateEndpoints(t *testing.T) {
	successCases := map[string]api.Endpoints{
		"simple endpoint": {
			ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
			Subsets: []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{IP: "10.10.1.1"}, {IP: "10.10.2.2"}},
					Ports:     []api.EndpointPort{{Name: "a", Port: 8675, Protocol: "TCP"}, {Name: "b", Port: 309, Protocol: "TCP"}},
				},
				{
					Addresses: []api.EndpointAddress{{IP: "10.10.3.3"}},
					Ports:     []api.EndpointPort{{Name: "a", Port: 93, Protocol: "TCP"}, {Name: "b", Port: 76, Protocol: "TCP"}},
				},
			},
		},
		"empty subsets": {
			ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
		},
		"no name required for singleton port": {
			ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
			Subsets: []api.EndpointSubset{
				{
					Addresses: []api.EndpointAddress{{IP: "10.10.1.1"}},
					Ports:     []api.EndpointPort{{Port: 8675, Protocol: "TCP"}},
				},
			},
		},
	}

	for k, v := range successCases {
		if errs := ValidateEndpoints(&v); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}
	}

	errorCases := map[string]struct {
		endpoints   api.Endpoints
		errorType   field.ErrorType
		errorDetail string
	}{
		"missing namespace": {
			endpoints: api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "mysvc"}},
			errorType: "FieldValueRequired",
		},
		"missing name": {
			endpoints: api.Endpoints{ObjectMeta: api.ObjectMeta{Namespace: "namespace"}},
			errorType: "FieldValueRequired",
		},
		"invalid namespace": {
			endpoints:   api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "no@#invalid.;chars\"allowed"}},
			errorType:   "FieldValueInvalid",
			errorDetail: DNS1123LabelErrorMsg,
		},
		"invalid name": {
			endpoints:   api.Endpoints{ObjectMeta: api.ObjectMeta{Name: "-_Invliad^&Characters", Namespace: "namespace"}},
			errorType:   "FieldValueInvalid",
			errorDetail: DNSSubdomainErrorMsg,
		},
		"empty addresses": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Ports: []api.EndpointPort{{Name: "a", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType: "FieldValueRequired",
		},
		"empty ports": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "10.10.3.3"}},
					},
				},
			},
			errorType: "FieldValueRequired",
		},
		"invalid IP": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "[2001:0db8:85a3:0042:1000:8a2e:0370:7334]"}},
						Ports:     []api.EndpointPort{{Name: "a", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "must be a valid IP address",
		},
		"Multiple ports, one without name": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []api.EndpointPort{{Port: 8675, Protocol: "TCP"}, {Name: "b", Port: 309, Protocol: "TCP"}},
					},
				},
			},
			errorType: "FieldValueRequired",
		},
		"Invalid port number": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []api.EndpointPort{{Name: "a", Port: 66000, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: PortRangeErrorMsg,
		},
		"Invalid protocol": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []api.EndpointPort{{Name: "a", Port: 93, Protocol: "Protocol"}},
					},
				},
			},
			errorType: "FieldValueNotSupported",
		},
		"Address missing IP": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{}},
						Ports:     []api.EndpointPort{{Name: "a", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "must be a valid IP address",
		},
		"Port missing number": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []api.EndpointPort{{Name: "a", Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: PortRangeErrorMsg,
		},
		"Port missing protocol": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "10.10.1.1"}},
						Ports:     []api.EndpointPort{{Name: "a", Port: 93}},
					},
				},
			},
			errorType: "FieldValueRequired",
		},
		"Address is loopback": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "127.0.0.1"}},
						Ports:     []api.EndpointPort{{Name: "p", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "loopback",
		},
		"Address is link-local": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "169.254.169.254"}},
						Ports:     []api.EndpointPort{{Name: "p", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "link-local",
		},
		"Address is link-local multicast": {
			endpoints: api.Endpoints{
				ObjectMeta: api.ObjectMeta{Name: "mysvc", Namespace: "namespace"},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{{IP: "224.0.0.1"}},
						Ports:     []api.EndpointPort{{Name: "p", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "link-local multicast",
		},
	}

	for k, v := range errorCases {
		if errs := ValidateEndpoints(&v.endpoints); len(errs) == 0 || errs[0].Type != v.errorType || !strings.Contains(errs[0].Detail, v.errorDetail) {
			t.Errorf("[%s] Expected error type %s with detail %q, got %v", k, v.errorType, v.errorDetail, errs)
		}
	}
}

func TestValidateTLSSecret(t *testing.T) {
	successCases := map[string]api.Secret{
		"emtpy certificate chain": {
			ObjectMeta: api.ObjectMeta{Name: "tls-cert", Namespace: "namespace"},
			Data: map[string][]byte{
				api.TLSCertKey:       []byte("public key"),
				api.TLSPrivateKeyKey: []byte("private key"),
			},
		},
	}
	for k, v := range successCases {
		if errs := ValidateSecret(&v); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}
	}
	errorCases := map[string]struct {
		secrets     api.Secret
		errorType   field.ErrorType
		errorDetail string
	}{
		"missing public key": {
			secrets: api.Secret{
				ObjectMeta: api.ObjectMeta{Name: "tls-cert"},
				Data: map[string][]byte{
					api.TLSCertKey: []byte("public key"),
				},
			},
			errorType: "FieldValueRequired",
		},
		"missing private key": {
			secrets: api.Secret{
				ObjectMeta: api.ObjectMeta{Name: "tls-cert"},
				Data: map[string][]byte{
					api.TLSCertKey: []byte("public key"),
				},
			},
			errorType: "FieldValueRequired",
		},
	}
	for k, v := range errorCases {
		if errs := ValidateSecret(&v.secrets); len(errs) == 0 || errs[0].Type != v.errorType || !strings.Contains(errs[0].Detail, v.errorDetail) {
			t.Errorf("[%s] Expected error type %s with detail %q, got %v", k, v.errorType, v.errorDetail, errs)
		}
	}
}

func TestValidateSecurityContext(t *testing.T) {
	priv := false
	var runAsUser int64 = 1
	fullValidSC := func() *api.SecurityContext {
		return &api.SecurityContext{
			Privileged: &priv,
			Capabilities: &api.Capabilities{
				Add:  []api.Capability{"foo"},
				Drop: []api.Capability{"bar"},
			},
			SELinuxOptions: &api.SELinuxOptions{
				User:  "user",
				Role:  "role",
				Type:  "type",
				Level: "level",
			},
			RunAsUser: &runAsUser,
		}
	}

	//setup data
	allSettings := fullValidSC()
	noCaps := fullValidSC()
	noCaps.Capabilities = nil

	noSELinux := fullValidSC()
	noSELinux.SELinuxOptions = nil

	noPrivRequest := fullValidSC()
	noPrivRequest.Privileged = nil

	noRunAsUser := fullValidSC()
	noRunAsUser.RunAsUser = nil

	successCases := map[string]struct {
		sc *api.SecurityContext
	}{
		"all settings":    {allSettings},
		"no capabilities": {noCaps},
		"no selinux":      {noSELinux},
		"no priv request": {noPrivRequest},
		"no run as user":  {noRunAsUser},
	}
	for k, v := range successCases {
		if errs := ValidateSecurityContext(v.sc, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("[%s] Expected success, got %v", k, errs)
		}
	}

	privRequestWithGlobalDeny := fullValidSC()
	requestPrivileged := true
	privRequestWithGlobalDeny.Privileged = &requestPrivileged

	negativeRunAsUser := fullValidSC()
	var negativeUser int64 = -1
	negativeRunAsUser.RunAsUser = &negativeUser

	errorCases := map[string]struct {
		sc          *api.SecurityContext
		errorType   field.ErrorType
		errorDetail string
	}{
		"request privileged when capabilities forbids": {
			sc:          privRequestWithGlobalDeny,
			errorType:   "FieldValueForbidden",
			errorDetail: "disallowed by policy",
		},
		"negative RunAsUser": {
			sc:          negativeRunAsUser,
			errorType:   "FieldValueInvalid",
			errorDetail: isNegativeErrorMsg,
		},
	}
	for k, v := range errorCases {
		if errs := ValidateSecurityContext(v.sc, field.NewPath("field")); len(errs) == 0 || errs[0].Type != v.errorType || !strings.Contains(errs[0].Detail, v.errorDetail) {
			t.Errorf("[%s] Expected error type %q with detail %q, got %v", k, v.errorType, v.errorDetail, errs)
		}
	}
}

func fakeValidSecurityContext(priv bool) *api.SecurityContext {
	return &api.SecurityContext{
		Privileged: &priv,
	}
}

func TestValidPodLogOptions(t *testing.T) {
	now := unversioned.Now()
	negative := int64(-1)
	zero := int64(0)
	positive := int64(1)
	tests := []struct {
		opt  api.PodLogOptions
		errs int
	}{
		{api.PodLogOptions{}, 0},
		{api.PodLogOptions{Previous: true}, 0},
		{api.PodLogOptions{Follow: true}, 0},
		{api.PodLogOptions{TailLines: &zero}, 0},
		{api.PodLogOptions{TailLines: &negative}, 1},
		{api.PodLogOptions{TailLines: &positive}, 0},
		{api.PodLogOptions{LimitBytes: &zero}, 1},
		{api.PodLogOptions{LimitBytes: &negative}, 1},
		{api.PodLogOptions{LimitBytes: &positive}, 0},
		{api.PodLogOptions{SinceSeconds: &negative}, 1},
		{api.PodLogOptions{SinceSeconds: &positive}, 0},
		{api.PodLogOptions{SinceSeconds: &zero}, 1},
		{api.PodLogOptions{SinceTime: &now}, 0},
	}
	for i, test := range tests {
		errs := ValidatePodLogOptions(&test.opt)
		if test.errs != len(errs) {
			t.Errorf("%d: Unexpected errors: %v", i, errs)
		}
	}
}

func TestValidateConfigMap(t *testing.T) {
	newConfigMap := func(name, namespace string, data map[string]string) api.ConfigMap {
		return api.ConfigMap{
			ObjectMeta: api.ObjectMeta{
				Name:      name,
				Namespace: namespace,
			},
			Data: data,
		}
	}

	var (
		validConfigMap = newConfigMap("validname", "validns", map[string]string{"key": "value"})
		maxKeyLength   = newConfigMap("validname", "validns", map[string]string{strings.Repeat("a", 253): "value"})

		emptyName        = newConfigMap("", "validns", nil)
		invalidName      = newConfigMap("NoUppercaseOrSpecialCharsLike=Equals", "validns", nil)
		emptyNs          = newConfigMap("validname", "", nil)
		invalidNs        = newConfigMap("validname", "NoUppercaseOrSpecialCharsLike=Equals", nil)
		invalidKey       = newConfigMap("validname", "validns", map[string]string{"a..b": "value"})
		leadingDotKey    = newConfigMap("validname", "validns", map[string]string{".ab": "value"})
		dotKey           = newConfigMap("validname", "validns", map[string]string{".": "value"})
		doubleDotKey     = newConfigMap("validname", "validns", map[string]string{"..": "value"})
		overMaxKeyLength = newConfigMap("validname", "validns", map[string]string{strings.Repeat("a", 254): "value"})
		overMaxSize      = newConfigMap("validname", "validns", map[string]string{"key": strings.Repeat("a", api.MaxSecretSize+1)})
	)

	tests := map[string]struct {
		cfg     api.ConfigMap
		isValid bool
	}{
		"valid":               {validConfigMap, true},
		"max key length":      {maxKeyLength, true},
		"leading dot key":     {leadingDotKey, true},
		"empty name":          {emptyName, false},
		"invalid name":        {invalidName, false},
		"invalid key":         {invalidKey, false},
		"empty namespace":     {emptyNs, false},
		"invalid namespace":   {invalidNs, false},
		"dot key":             {dotKey, false},
		"double dot key":      {doubleDotKey, false},
		"over max key length": {overMaxKeyLength, false},
		"over max size":       {overMaxSize, false},
	}

	for name, tc := range tests {
		errs := ValidateConfigMap(&tc.cfg)
		if tc.isValid && len(errs) > 0 {
			t.Errorf("%v: unexpected error: %v", name, errs)
		}
		if !tc.isValid && len(errs) == 0 {
			t.Errorf("%v: unexpected non-error", name)
		}
	}
}

func TestValidateConfigMapUpdate(t *testing.T) {
	newConfigMap := func(version, name, namespace string, data map[string]string) api.ConfigMap {
		return api.ConfigMap{
			ObjectMeta: api.ObjectMeta{
				Name:            name,
				Namespace:       namespace,
				ResourceVersion: version,
			},
			Data: data,
		}
	}

	var (
		validConfigMap = newConfigMap("1", "validname", "validns", map[string]string{"key": "value"})
		noVersion      = newConfigMap("", "validname", "validns", map[string]string{"key": "value"})
	)

	cases := []struct {
		name    string
		newCfg  api.ConfigMap
		oldCfg  api.ConfigMap
		isValid bool
	}{
		{
			name:    "valid",
			newCfg:  validConfigMap,
			oldCfg:  validConfigMap,
			isValid: true,
		},
		{
			name:    "invalid",
			newCfg:  noVersion,
			oldCfg:  validConfigMap,
			isValid: false,
		},
	}

	for _, tc := range cases {
		errs := ValidateConfigMapUpdate(&tc.newCfg, &tc.oldCfg)
		if tc.isValid && len(errs) > 0 {
			t.Errorf("%v: unexpected error: %v", tc.name, errs)
		}
		if !tc.isValid && len(errs) == 0 {
			t.Errorf("%v: unexpected non-error", tc.name)
		}
	}
}

func TestValidateHasLabel(t *testing.T) {
	successCase := api.ObjectMeta{
		Name:      "123",
		Namespace: "ns",
		Labels: map[string]string{
			"other": "blah",
			"foo":   "bar",
		},
	}
	if errs := ValidateHasLabel(successCase, field.NewPath("field"), "foo", "bar"); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	missingCase := api.ObjectMeta{
		Name:      "123",
		Namespace: "ns",
		Labels: map[string]string{
			"other": "blah",
		},
	}
	if errs := ValidateHasLabel(missingCase, field.NewPath("field"), "foo", "bar"); len(errs) == 0 {
		t.Errorf("expected failure")
	}

	wrongValueCase := api.ObjectMeta{
		Name:      "123",
		Namespace: "ns",
		Labels: map[string]string{
			"other": "blah",
			"foo":   "notbar",
		},
	}
	if errs := ValidateHasLabel(wrongValueCase, field.NewPath("field"), "foo", "bar"); len(errs) == 0 {
		t.Errorf("expected failure")
	}
}
