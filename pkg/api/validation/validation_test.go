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
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/util"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/fielderrors"
	errors "k8s.io/kubernetes/pkg/util/fielderrors"
	"k8s.io/kubernetes/pkg/util/sets"
)

func expectPrefix(t *testing.T, prefix string, errs fielderrors.ValidationErrorList) {
	for i := range errs {
		if f, p := errs[i].(*errors.ValidationError).Field, prefix; !strings.HasPrefix(f, p) {
			t.Errorf("expected prefix '%s' for field '%s' (%v)", p, f, errs[i])
		}
	}
}

// Ensure custom name functions are allowed
func TestValidateObjectMetaCustomName(t *testing.T) {
	errs := ValidateObjectMeta(&api.ObjectMeta{Name: "test", GenerateName: "foo"}, false, func(s string, prefix bool) (bool, string) {
		if s == "test" {
			return true, ""
		}
		return false, "name-gen"
	})
	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), "name-gen") {
		t.Errorf("unexpected error message: %v", errs)
	}
}

// Ensure namespace names follow dns label format
func TestValidateObjectMetaNamespaces(t *testing.T) {
	errs := ValidateObjectMeta(&api.ObjectMeta{Name: "test", Namespace: "foo.bar"}, false, func(s string, prefix bool) (bool, string) {
		return true, ""
	})
	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), "invalid value 'foo.bar'") {
		t.Errorf("unexpected error message: %v", errs)
	}
	maxLength := 63
	letters := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
	b := make([]rune, maxLength+1)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	errs = ValidateObjectMeta(&api.ObjectMeta{Name: "test", Namespace: string(b)}, false, func(s string, prefix bool) (bool, string) {
		return true, ""
	})
	if len(errs) != 1 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if !strings.Contains(errs[0].Error(), "invalid value") {
		t.Errorf("unexpected error message: %v", errs)
	}
}

func TestValidateObjectMetaUpdateIgnoresCreationTimestamp(t *testing.T) {
	if errs := ValidateObjectMetaUpdate(
		&api.ObjectMeta{Name: "test", ResourceVersion: "1"},
		&api.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: unversioned.NewTime(time.Unix(10, 0))},
	); len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if errs := ValidateObjectMetaUpdate(
		&api.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: unversioned.NewTime(time.Unix(10, 0))},
		&api.ObjectMeta{Name: "test", ResourceVersion: "1"},
	); len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if errs := ValidateObjectMetaUpdate(
		&api.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: unversioned.NewTime(time.Unix(10, 0))},
		&api.ObjectMeta{Name: "test", ResourceVersion: "1", CreationTimestamp: unversioned.NewTime(time.Unix(11, 0))},
	); len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
}

// Ensure trailing slash is allowed in generate name
func TestValidateObjectMetaTrimsTrailingSlash(t *testing.T) {
	errs := ValidateObjectMeta(&api.ObjectMeta{Name: "test", GenerateName: "foo-"}, false, NameIsDNSSubdomain)
	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
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
		errs := ValidateLabels(successCases[i], "field")
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
		errs := ValidateLabels(labelNameErrorCases[i], "field")
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		} else {
			detail := errs[0].(*errors.ValidationError).Detail
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
		errs := ValidateLabels(labelValueErrorCases[i], "field")
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		} else {
			detail := errs[0].(*errors.ValidationError).Detail
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
		{"a": strings.Repeat("b", (64*1024)-1)},
		{
			"a": strings.Repeat("b", (32*1024)-1),
			"c": strings.Repeat("d", (32*1024)-1),
		},
	}
	for i := range successCases {
		errs := ValidateAnnotations(successCases[i], "field")
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
		errs := ValidateAnnotations(nameErrorCases[i], "field")
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		}
		detail := errs[0].(*errors.ValidationError).Detail
		if detail != qualifiedNameErrorMsg {
			t.Errorf("error detail %s should be equal %s", detail, qualifiedNameErrorMsg)
		}
	}
	totalSizeErrorCases := []map[string]string{
		{"a": strings.Repeat("b", 64*1024)},
		{
			"a": strings.Repeat("b", 32*1024),
			"c": strings.Repeat("d", 32*1024),
		},
	}
	for i := range totalSizeErrorCases {
		errs := ValidateAnnotations(totalSizeErrorCases[i], "field")
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

func TestValidateVolumes(t *testing.T) {
	lun := 1
	successCase := []api.Volume{
		{Name: "abc", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/mnt/path1"}}},
		{Name: "123", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/mnt/path2"}}},
		{Name: "abc-123", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/mnt/path3"}}},
		{Name: "empty", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
		{Name: "gcepd", VolumeSource: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}},
		{Name: "awsebs", VolumeSource: api.VolumeSource{AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{VolumeID: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}},
		{Name: "gitrepo", VolumeSource: api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{Repository: "my-repo", Revision: "hashstring"}}},
		{Name: "iscsidisk", VolumeSource: api.VolumeSource{ISCSI: &api.ISCSIVolumeSource{TargetPortal: "127.0.0.1", IQN: "iqn.2015-02.example.com:test", Lun: 1, FSType: "ext4", ReadOnly: false}}},
		{Name: "secret", VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "my-secret"}}},
		{Name: "glusterfs", VolumeSource: api.VolumeSource{Glusterfs: &api.GlusterfsVolumeSource{EndpointsName: "host1", Path: "path", ReadOnly: false}}},
		{Name: "flocker", VolumeSource: api.VolumeSource{Flocker: &api.FlockerVolumeSource{DatasetName: "datasetName"}}},
		{Name: "rbd", VolumeSource: api.VolumeSource{RBD: &api.RBDVolumeSource{CephMonitors: []string{"foo"}, RBDImage: "bar", FSType: "ext4"}}},
		{Name: "cinder", VolumeSource: api.VolumeSource{Cinder: &api.CinderVolumeSource{"29ea5088-4f60-4757-962e-dba678767887", "ext4", false}}},
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
		{Name: "fc", VolumeSource: api.VolumeSource{FC: &api.FCVolumeSource{[]string{"some_wwn"}, &lun, "ext4", false}}},
	}
	names, errs := validateVolumes(successCase)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	if len(names) != len(successCase) || !names.HasAll("abc", "123", "abc-123", "empty", "gcepd", "gitrepo", "secret", "iscsidisk", "cinder", "cephfs", "fc") {
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
	zeroWWN := api.VolumeSource{FC: &api.FCVolumeSource{[]string{}, &lun, "ext4", false}}
	emptyLun := api.VolumeSource{FC: &api.FCVolumeSource{[]string{"wwn"}, nil, "ext4", false}}
	slashInName := api.VolumeSource{Flocker: &api.FlockerVolumeSource{DatasetName: "foo/bar"}}
	errorCases := map[string]struct {
		V []api.Volume
		T errors.ValidationErrorType
		F string
		D string
	}{
		"zero-length name":           {[]api.Volume{{Name: "", VolumeSource: emptyVS}}, errors.ValidationErrorTypeRequired, "[0].name", ""},
		"name > 63 characters":       {[]api.Volume{{Name: strings.Repeat("a", 64), VolumeSource: emptyVS}}, errors.ValidationErrorTypeInvalid, "[0].name", "must be a DNS label (at most 63 characters, matching regex [a-z0-9]([-a-z0-9]*[a-z0-9])?): e.g. \"my-name\""},
		"name not a DNS label":       {[]api.Volume{{Name: "a.b.c", VolumeSource: emptyVS}}, errors.ValidationErrorTypeInvalid, "[0].name", "must be a DNS label (at most 63 characters, matching regex [a-z0-9]([-a-z0-9]*[a-z0-9])?): e.g. \"my-name\""},
		"name not unique":            {[]api.Volume{{Name: "abc", VolumeSource: emptyVS}, {Name: "abc", VolumeSource: emptyVS}}, errors.ValidationErrorTypeDuplicate, "[1].name", ""},
		"empty portal":               {[]api.Volume{{Name: "badportal", VolumeSource: emptyPortal}}, errors.ValidationErrorTypeRequired, "[0].source.iscsi.targetPortal", ""},
		"empty iqn":                  {[]api.Volume{{Name: "badiqn", VolumeSource: emptyIQN}}, errors.ValidationErrorTypeRequired, "[0].source.iscsi.iqn", ""},
		"empty hosts":                {[]api.Volume{{Name: "badhost", VolumeSource: emptyHosts}}, errors.ValidationErrorTypeRequired, "[0].source.glusterfs.endpoints", ""},
		"empty path":                 {[]api.Volume{{Name: "badpath", VolumeSource: emptyPath}}, errors.ValidationErrorTypeRequired, "[0].source.glusterfs.path", ""},
		"empty datasetName":          {[]api.Volume{{Name: "badname", VolumeSource: emptyName}}, errors.ValidationErrorTypeRequired, "[0].source.flocker.datasetName", ""},
		"empty mon":                  {[]api.Volume{{Name: "badmon", VolumeSource: emptyMon}}, errors.ValidationErrorTypeRequired, "[0].source.rbd.monitors", ""},
		"empty image":                {[]api.Volume{{Name: "badimage", VolumeSource: emptyImage}}, errors.ValidationErrorTypeRequired, "[0].source.rbd.image", ""},
		"empty cephfs mon":           {[]api.Volume{{Name: "badmon", VolumeSource: emptyCephFSMon}}, errors.ValidationErrorTypeRequired, "[0].source.cephfs.monitors", ""},
		"empty metatada path":        {[]api.Volume{{Name: "emptyname", VolumeSource: emptyPathName}}, errors.ValidationErrorTypeRequired, "[0].source.downwardApi.path", ""},
		"absolute path":              {[]api.Volume{{Name: "absolutepath", VolumeSource: absolutePathName}}, errors.ValidationErrorTypeForbidden, "[0].source.downwardApi.path", ""},
		"dot dot path":               {[]api.Volume{{Name: "dotdotpath", VolumeSource: dotDotInPath}}, errors.ValidationErrorTypeInvalid, "[0].source.downwardApi.path", "must not contain \"..\"."},
		"dot dot file name":          {[]api.Volume{{Name: "dotdotfilename", VolumeSource: dotDotPathName}}, errors.ValidationErrorTypeInvalid, "[0].source.downwardApi.path", "must not start with \"..\"."},
		"dot dot first level dirent": {[]api.Volume{{Name: "dotdotdirfilename", VolumeSource: dotDotFirstLevelDirent}}, errors.ValidationErrorTypeInvalid, "[0].source.downwardApi.path", "must not start with \"..\"."},
		"empty wwn":                  {[]api.Volume{{Name: "badimage", VolumeSource: zeroWWN}}, errors.ValidationErrorTypeRequired, "[0].source.fc.targetWWNs", ""},
		"empty lun":                  {[]api.Volume{{Name: "badimage", VolumeSource: emptyLun}}, errors.ValidationErrorTypeRequired, "[0].source.fc.lun", ""},
		"slash in datasetName":       {[]api.Volume{{Name: "slashinname", VolumeSource: slashInName}}, errors.ValidationErrorTypeInvalid, "[0].source.flocker.datasetName", "must not contain '/'"},
	}
	for k, v := range errorCases {
		_, errs := validateVolumes(v.V)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.V)
			continue
		}
		for i := range errs {
			if errs[i].(*errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(*errors.ValidationError).Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
			detail := errs[i].(*errors.ValidationError).Detail
			if detail != v.D {
				t.Errorf("%s: expected error detail \"%s\", got \"%s\"", k, v.D, detail)
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
	if errs := validatePorts(successCase); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	nonCanonicalCase := []api.ContainerPort{
		{ContainerPort: 80, Protocol: "TCP"},
	}
	if errs := validatePorts(nonCanonicalCase); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		P []api.ContainerPort
		T errors.ValidationErrorType
		F string
		D string
	}{
		"name > 15 characters":                     {[]api.ContainerPort{{Name: strings.Repeat("a", 16), ContainerPort: 80, Protocol: "TCP"}}, errors.ValidationErrorTypeInvalid, "[0].name", portNameErrorMsg},
		"name not a IANA svc name ":                {[]api.ContainerPort{{Name: "a.b.c", ContainerPort: 80, Protocol: "TCP"}}, errors.ValidationErrorTypeInvalid, "[0].name", portNameErrorMsg},
		"name not a IANA svc name (i.e. a number)": {[]api.ContainerPort{{Name: "80", ContainerPort: 80, Protocol: "TCP"}}, errors.ValidationErrorTypeInvalid, "[0].name", portNameErrorMsg},
		"name not unique": {[]api.ContainerPort{
			{Name: "abc", ContainerPort: 80, Protocol: "TCP"},
			{Name: "abc", ContainerPort: 81, Protocol: "TCP"},
		}, errors.ValidationErrorTypeDuplicate, "[1].name", ""},
		"zero container port":    {[]api.ContainerPort{{ContainerPort: 0, Protocol: "TCP"}}, errors.ValidationErrorTypeInvalid, "[0].containerPort", portRangeErrorMsg},
		"invalid container port": {[]api.ContainerPort{{ContainerPort: 65536, Protocol: "TCP"}}, errors.ValidationErrorTypeInvalid, "[0].containerPort", portRangeErrorMsg},
		"invalid host port":      {[]api.ContainerPort{{ContainerPort: 80, HostPort: 65536, Protocol: "TCP"}}, errors.ValidationErrorTypeInvalid, "[0].hostPort", portRangeErrorMsg},
		"invalid protocol case":  {[]api.ContainerPort{{ContainerPort: 80, Protocol: "tcp"}}, errors.ValidationErrorTypeNotSupported, "[0].protocol", "supported values: TCP, UDP"},
		"invalid protocol":       {[]api.ContainerPort{{ContainerPort: 80, Protocol: "ICMP"}}, errors.ValidationErrorTypeNotSupported, "[0].protocol", "supported values: TCP, UDP"},
		"protocol required":      {[]api.ContainerPort{{Name: "abc", ContainerPort: 80}}, errors.ValidationErrorTypeRequired, "[0].protocol", ""},
	}
	for k, v := range errorCases {
		errs := validatePorts(v.P)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			if errs[i].(*errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(*errors.ValidationError).Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
			detail := errs[i].(*errors.ValidationError).Detail
			if detail != v.D {
				t.Errorf("%s: expected error detail either empty or %s, got %s", k, v.D, detail)
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
					APIVersion: testapi.Default.Version(),
					FieldPath:  "metadata.name",
				},
			},
		},
	}
	if errs := validateEnv(successCase); len(errs) != 0 {
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
			expectedError: "[0].name: required value",
		},
		{
			name:          "name not a C identifier",
			envs:          []api.EnvVar{{Name: "a.b.c"}},
			expectedError: `[0].name: invalid value 'a.b.c', Details: must be a C identifier (matching regex [A-Za-z_][A-Za-z0-9_]*): e.g. "my_name" or "MyName"`,
		},
		{
			name: "value and valueFrom specified",
			envs: []api.EnvVar{{
				Name:  "abc",
				Value: "foo",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: testapi.Default.Version(),
						FieldPath:  "metadata.name",
					},
				},
			}},
			expectedError: "[0].valueFrom: invalid value '', Details: sources cannot be specified when value is not empty",
		},
		{
			name: "missing FieldPath on ObjectFieldSelector",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: testapi.Default.Version(),
					},
				},
			}},
			expectedError: "[0].valueFrom.fieldRef.fieldPath: required value",
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
			expectedError: "[0].valueFrom.fieldRef.apiVersion: required value",
		},
		{
			name: "invalid fieldPath",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						FieldPath:  "metadata.whoops",
						APIVersion: testapi.Default.Version(),
					},
				},
			}},
			expectedError: "[0].valueFrom.fieldRef.fieldPath: invalid value 'metadata.whoops', Details: error converting fieldPath",
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
			expectedError: "[0].valueFrom.fieldRef.fieldPath: unsupported value 'metadata.labels', Details: supported values: metadata.name, metadata.namespace, status.podIP",
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
			expectedError: "[0].valueFrom.fieldRef.fieldPath: unsupported value 'metadata.annotations', Details: supported values: metadata.name, metadata.namespace, status.podIP",
		},
		{
			name: "unsupported fieldPath",
			envs: []api.EnvVar{{
				Name: "abc",
				ValueFrom: &api.EnvVarSource{
					FieldRef: &api.ObjectFieldSelector{
						FieldPath:  "status.phase",
						APIVersion: testapi.Default.Version(),
					},
				},
			}},
			expectedError: "[0].valueFrom.fieldRef.fieldPath: unsupported value 'status.phase', Details: supported values: metadata.name, metadata.namespace, status.podIP",
		},
	}
	for _, tc := range errorCases {
		if errs := validateEnv(tc.envs); len(errs) == 0 {
			t.Errorf("expected failure for %s", tc.name)
		} else {
			for i := range errs {
				str := errs[i].(*errors.ValidationError).Error()
				if str != "" && str != tc.expectedError {
					t.Errorf("%s: expected error detail either empty or %s, got %s", tc.name, tc.expectedError, str)
				}
			}
		}
	}
}

func TestValidateVolumeMounts(t *testing.T) {
	volumes := sets.NewString("abc", "123", "abc-123")

	successCase := []api.VolumeMount{
		{Name: "abc", MountPath: "/foo"},
		{Name: "123", MountPath: "/foo"},
		{Name: "abc-123", MountPath: "/bar"},
	}
	if errs := validateVolumeMounts(successCase, volumes); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string][]api.VolumeMount{
		"empty name":      {{Name: "", MountPath: "/foo"}},
		"name not found":  {{Name: "", MountPath: "/foo"}},
		"empty mountpath": {{Name: "abc", MountPath: ""}},
	}
	for k, v := range errorCases {
		if errs := validateVolumeMounts(v, volumes); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateProbe(t *testing.T) {
	handler := api.Handler{Exec: &api.ExecAction{Command: []string{"echo"}}}
	successCases := []*api.Probe{
		nil,
		{TimeoutSeconds: 10, InitialDelaySeconds: 0, Handler: handler},
		{TimeoutSeconds: 0, InitialDelaySeconds: 10, Handler: handler},
	}
	for _, p := range successCases {
		if errs := validateProbe(p); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []*api.Probe{
		{TimeoutSeconds: 10, InitialDelaySeconds: 10},
		{TimeoutSeconds: 10, InitialDelaySeconds: -10, Handler: handler},
		{TimeoutSeconds: -10, InitialDelaySeconds: 10, Handler: handler},
		{TimeoutSeconds: -10, InitialDelaySeconds: -10, Handler: handler},
	}
	for _, p := range errorCases {
		if errs := validateProbe(p); len(errs) == 0 {
			t.Errorf("expected failure for %v", p)
		}
	}
}

func TestValidateHandler(t *testing.T) {
	successCases := []api.Handler{
		{Exec: &api.ExecAction{Command: []string{"echo"}}},
		{HTTPGet: &api.HTTPGetAction{Path: "/", Port: util.NewIntOrStringFromInt(1), Host: "", Scheme: "HTTP"}},
		{HTTPGet: &api.HTTPGetAction{Path: "/foo", Port: util.NewIntOrStringFromInt(65535), Host: "host", Scheme: "HTTP"}},
		{HTTPGet: &api.HTTPGetAction{Path: "/", Port: util.NewIntOrStringFromString("port"), Host: "", Scheme: "HTTP"}},
	}
	for _, h := range successCases {
		if errs := validateHandler(&h); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []api.Handler{
		{},
		{Exec: &api.ExecAction{Command: []string{}}},
		{HTTPGet: &api.HTTPGetAction{Path: "", Port: util.NewIntOrStringFromInt(0), Host: ""}},
		{HTTPGet: &api.HTTPGetAction{Path: "/foo", Port: util.NewIntOrStringFromInt(65536), Host: "host"}},
		{HTTPGet: &api.HTTPGetAction{Path: "", Port: util.NewIntOrStringFromString(""), Host: ""}},
	}
	for _, h := range errorCases {
		if errs := validateHandler(&h); len(errs) == 0 {
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
		errs := validatePullPolicy(ctr)
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
	if errs := validateContainers(successCase, volumes); len(errs) != 0 {
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
							Port: util.IntOrString{IntVal: 0},
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
		if errs := validateContainers(v, volumes); len(errs) == 0 {
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
		if errs := validateRestartPolicy(&policy); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []api.RestartPolicy{"", "newpolicy"}

	for k, policy := range errorCases {
		if errs := validateRestartPolicy(&policy); len(errs) == 0 {
			t.Errorf("expected failure for %d", k)
		}
	}
}

func TestValidateDNSPolicy(t *testing.T) {
	successCases := []api.DNSPolicy{api.DNSClusterFirst, api.DNSDefault, api.DNSPolicy(api.DNSClusterFirst)}
	for _, policy := range successCases {
		if errs := validateDNSPolicy(&policy); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []api.DNSPolicy{api.DNSPolicy("invalid")}
	for _, policy := range errorCases {
		if errs := validateDNSPolicy(&policy); len(errs) == 0 {
			t.Errorf("expected failure for %v", policy)
		}
	}
}

func TestValidatePodSpec(t *testing.T) {
	activeDeadlineSeconds := int64(30)
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
	}
	for i := range successCases {
		if errs := ValidatePodSpec(&successCases[i]); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	activeDeadlineSeconds = int64(0)
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
	}
	for k, v := range failureCases {
		if errs := ValidatePodSpec(&v); len(errs) == 0 {
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
	}
	for k, v := range errorCases {
		if errs := ValidatePod(&v); len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		}
	}
}

func TestValidatePodUpdate(t *testing.T) {
	now := unversioned.Now()
	grace := int64(30)
	grace2 := int64(31)
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
				ObjectMeta: api.ObjectMeta{Name: "foo", DeletionTimestamp: &now},
				Spec:       api.PodSpec{Containers: []api.Container{{Image: "foo:V1"}}},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
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
				t.Errorf("unexpected invalid: %s %v, %v", test.test, test.a, test.b)
			}
		} else {
			if len(errs) == 0 {
				t.Errorf("unexpected valid: %s %v, %v", test.test, test.a, test.b)
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
			Ports:           []api.ServicePort{{Name: "p", Protocol: "TCP", Port: 8675, TargetPort: util.NewIntOrStringFromInt(8675)}},
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
			name: "empty port[0] name",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Name = ""
			},
			numErrs: 0,
		},
		{
			name: "empty port[1] name",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "", Protocol: "TCP", Port: 12345, TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "empty multi-port port[0] name",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].Name = ""
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "p", Protocol: "TCP", Port: 12345, TargetPort: util.NewIntOrStringFromInt(12345)})
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
				s.Spec.Ports[0].TargetPort = util.NewIntOrStringFromInt(65536)
			},
			numErrs: 1,
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
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "p", Port: 12345, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "invalid load balancer protocol 1",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports[0].Protocol = "UDP"
			},
			numErrs: 1,
		},
		{
			name: "invalid load balancer protocol 2",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "UDP", TargetPort: util.NewIntOrStringFromInt(12345)})
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
				s.Spec.Ports[0].TargetPort = util.NewIntOrStringFromInt(12345)
			},
			numErrs: 0,
		},
		{
			name: "valid 3",
			tweakSvc: func(s *api.Service) {
				s.Spec.Ports[0].TargetPort = util.NewIntOrStringFromString("http")
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
				s.Spec.Ports[0].TargetPort = util.NewIntOrStringFromString("http")
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
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid external load balancer 2 ports",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "duplicate nodeports",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: util.NewIntOrStringFromInt(1)})
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "r", Port: 2, Protocol: "TCP", NodePort: 1, TargetPort: util.NewIntOrStringFromInt(2)})
			},
			numErrs: 1,
		},
		{
			name: "duplicate nodeports (different protocols)",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: util.NewIntOrStringFromInt(1)})
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "r", Port: 2, Protocol: "UDP", NodePort: 1, TargetPort: util.NewIntOrStringFromInt(2)})
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
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid type loadbalancer with NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", NodePort: 12345, TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid type=NodePort service with NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", NodePort: 12345, TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid type=NodePort service without NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "valid cluster service without NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeClusterIP
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 0,
		},
		{
			name: "invalid cluster service with NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeClusterIP
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", NodePort: 12345, TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 1,
		},
		{
			name: "invalid public service with duplicate NodePort",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeNodePort
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "p1", Port: 1, Protocol: "TCP", NodePort: 1, TargetPort: util.NewIntOrStringFromInt(1)})
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "p2", Port: 2, Protocol: "TCP", NodePort: 1, TargetPort: util.NewIntOrStringFromInt(2)})
			},
			numErrs: 1,
		},
		{
			name: "valid type=LoadBalancer",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "q", Port: 12345, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 0,
		},
		{
			// For now we open firewalls, and its insecure if we open 10250, remove this
			// when we have better protections in place.
			name: "invalid port type=LoadBalancer",
			tweakSvc: func(s *api.Service) {
				s.Spec.Type = api.ServiceTypeLoadBalancer
				s.Spec.Ports = append(s.Spec.Ports, api.ServicePort{Name: "kubelet", Port: 10250, Protocol: "TCP", TargetPort: util.NewIntOrStringFromInt(12345)})
			},
			numErrs: 1,
		},
	}

	for _, tc := range testCases {
		svc := makeValidService()
		tc.tweakSvc(&svc)
		errs := ValidateService(&svc)
		if len(errs) != tc.numErrs {
			t.Errorf("Unexpected error list for case %q: %v", tc.name, utilerrors.NewAggregate(errs))
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
		if errs := ValidateReplicationControllerStatusUpdate(&successCase.old, &successCase.update); len(errs) != 0 {
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
		if errs := ValidateReplicationControllerStatusUpdate(&errorCase.old, &errorCase.update); len(errs) == 0 {
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
		if errs := ValidateReplicationControllerUpdate(&successCase.old, &successCase.update); len(errs) != 0 {
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
		if errs := ValidateReplicationControllerUpdate(&errorCase.old, &errorCase.update); len(errs) == 0 {
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
			field := errs[i].(*errors.ValidationError).Field
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
			field := errs[i].(*errors.ValidationError).Field
			expectedFields := map[string]bool{
				"metadata.name":        true,
				"metadata.labels":      true,
				"metadata.annotations": true,
				"metadata.namespace":   true,
				"spec.ExternalID":      true,
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
		errs := ValidateNodeUpdate(&test.oldNode, &test.node)
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
		errs := ValidateServiceUpdate(&oldSvc, &newSvc)
		if len(errs) != tc.numErrs {
			t.Errorf("Unexpected error list for case %q: %v", tc.name, utilerrors.NewAggregate(errs))
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
		err := validateResourceName(item.input, "sth")
		if len(err) != 0 && item.success {
			t.Errorf("expected no failure for input %q", item.input)
		} else if len(err) == 0 && !item.success {
			t.Errorf("expected failure for input %q", item.input)
			for i := range err {
				detail := err[i].(*errors.ValidationError).Detail
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
			"Default is not supported when limit type is Pod",
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
			"DefaultRequest is not supported when limit type is Pod",
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
			"maxLimitRequestRatio 800m is less than 1",
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
			"maxLimitRequestRatio 10 is greater than max/min = 4.000000",
		},
	}

	for k, v := range errorCases {
		errs := ValidateLimitRange(&v.R)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			detail := errs[i].(*errors.ValidationError).Detail
			if detail != v.D {
				t.Errorf("%s: expected error detail either empty or %s, got %s", k, v.D, detail)
			}
		}
	}

}

func TestValidateResourceQuota(t *testing.T) {
	spec := api.ResourceQuotaSpec{
		Hard: api.ResourceList{
			api.ResourceCPU:                    resource.MustParse("100"),
			api.ResourceMemory:                 resource.MustParse("10000"),
			api.ResourcePods:                   resource.MustParse("10"),
			api.ResourceServices:               resource.MustParse("0"),
			api.ResourceReplicationControllers: resource.MustParse("10"),
			api.ResourceQuotas:                 resource.MustParse("10"),
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
		},
	}

	successCases := []api.ResourceQuota{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: spec,
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
	}
	for k, v := range errorCases {
		errs := ValidateResourceQuota(&v.R)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].(*errors.ValidationError).Field
			detail := errs[i].(*errors.ValidationError).Detail
			if field != "metadata.name" && field != "metadata.namespace" && !api.IsStandardResourceName(field) {
				t.Errorf("%s: missing prefix for: %v", k, field)
			}
			if detail != v.D {
				t.Errorf("%s: expected error detail either empty or %s, got %s", k, v.D, detail)
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

	var (
		missingDockerConfigKey = validDockerSecret()
		emptyDockerConfigKey   = validDockerSecret()
		invalidDockerConfigKey = validDockerSecret()
	)

	delete(missingDockerConfigKey.Data, api.DockerConfigKey)
	emptyDockerConfigKey.Data[api.DockerConfigKey] = []byte("")
	invalidDockerConfigKey.Data[api.DockerConfigKey] = []byte("bad")

	tests := map[string]struct {
		secret api.Secret
		valid  bool
	}{
		"valid":             {validDockerSecret(), true},
		"missing dockercfg": {missingDockerConfigKey, false},
		"empty dockercfg":   {emptyDockerConfigKey, false},
		"invalid dockercfg": {invalidDockerConfigKey, false},
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
		errorType   fielderrors.ValidationErrorType
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
						Addresses: []api.EndpointAddress{{IP: "2001:0db8:85a3:0042:1000:8a2e:0370:7334"}},
						Ports:     []api.EndpointPort{{Name: "a", Port: 93, Protocol: "TCP"}},
					},
				},
			},
			errorType:   "FieldValueInvalid",
			errorDetail: "invalid IPv4 address",
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
			errorDetail: portRangeErrorMsg,
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
			errorDetail: "invalid IPv4 address",
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
			errorDetail: portRangeErrorMsg,
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
		if errs := ValidateEndpoints(&v.endpoints); len(errs) == 0 || errs[0].(*errors.ValidationError).Type != v.errorType || !strings.Contains(errs[0].(*errors.ValidationError).Detail, v.errorDetail) {
			t.Errorf("Expected error type %s with detail %s for %s, got %v", v.errorType, v.errorDetail, k, errs)
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
		if errs := ValidateSecurityContext(v.sc); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
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
		errorType   fielderrors.ValidationErrorType
		errorDetail string
	}{
		"request privileged when capabilities forbids": {
			sc:          privRequestWithGlobalDeny,
			errorType:   "FieldValueForbidden",
			errorDetail: "",
		},
		"negative RunAsUser": {
			sc:          negativeRunAsUser,
			errorType:   "FieldValueInvalid",
			errorDetail: "runAsUser cannot be negative",
		},
	}
	for k, v := range errorCases {
		if errs := ValidateSecurityContext(v.sc); len(errs) == 0 || errs[0].(*errors.ValidationError).Type != v.errorType || errs[0].(*errors.ValidationError).Detail != v.errorDetail {
			t.Errorf("Expected error type %s with detail %s for %s, got %v", v.errorType, v.errorDetail, k, errs)
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
