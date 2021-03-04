/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"reflect"
	"testing"

	policy "k8s.io/api/policy/v1beta1"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// TestVolumeSourceFSTypeDrift ensures that for every known type of volume source (by the fields on
// a VolumeSource object that GetVolumeFSType is returning a good value.  This ensures both that we're
// returning an FSType for the VolumeSource field (protect the GetVolumeFSType method) and that we
// haven't drifted (ensure new fields in VolumeSource are covered).
func TestVolumeSourceFSTypeDrift(t *testing.T) {
	allFSTypes := GetAllFSTypesAsSet()
	val := reflect.ValueOf(api.VolumeSource{})

	for i := 0; i < val.NumField(); i++ {
		fieldVal := val.Type().Field(i)

		volumeSource := api.VolumeSource{}
		volumeSourceVolume := reflect.New(fieldVal.Type.Elem())

		reflect.ValueOf(&volumeSource).Elem().FieldByName(fieldVal.Name).Set(volumeSourceVolume)

		fsType, err := GetVolumeFSType(api.Volume{VolumeSource: volumeSource})
		if err != nil {
			t.Errorf("error getting fstype for field %s.  This likely means that drift has occurred between FSType and VolumeSource.  Please update the api and getVolumeFSType", fieldVal.Name)
		}

		if !allFSTypes.Has(string(fsType)) {
			t.Errorf("%s was missing from GetFSTypesAsSet", fsType)
		}
	}
}

func TestPSPAllowsFSType(t *testing.T) {
	tests := map[string]struct {
		psp    *policy.PodSecurityPolicy
		fsType policy.FSType
		allows bool
	}{
		"nil psp": {
			psp:    nil,
			fsType: policy.HostPath,
			allows: false,
		},
		"empty volumes": {
			psp:    &policy.PodSecurityPolicy{},
			fsType: policy.HostPath,
			allows: false,
		},
		"non-matching": {
			psp: &policy.PodSecurityPolicy{
				Spec: policy.PodSecurityPolicySpec{
					Volumes: []policy.FSType{policy.AWSElasticBlockStore},
				},
			},
			fsType: policy.HostPath,
			allows: false,
		},
		"match on FSTypeAll": {
			psp: &policy.PodSecurityPolicy{
				Spec: policy.PodSecurityPolicySpec{
					Volumes: []policy.FSType{policy.All},
				},
			},
			fsType: policy.HostPath,
			allows: true,
		},
		"match on direct match": {
			psp: &policy.PodSecurityPolicy{
				Spec: policy.PodSecurityPolicySpec{
					Volumes: []policy.FSType{policy.HostPath},
				},
			},
			fsType: policy.HostPath,
			allows: true,
		},
	}

	for k, v := range tests {
		allows := PSPAllowsFSType(v.psp, v.fsType)
		if v.allows != allows {
			t.Errorf("%s expected PSPAllowsFSType to return %t but got %t", k, v.allows, allows)
		}
	}
}

func TestAllowsHostVolumePath(t *testing.T) {
	tests := map[string]struct {
		psp            *policy.PodSecurityPolicy
		path           string
		allows         bool
		mustBeReadOnly bool
	}{
		"nil psp": {
			psp:            nil,
			path:           "/test",
			allows:         false,
			mustBeReadOnly: false,
		},
		"empty allowed paths": {
			psp:            &policy.PodSecurityPolicy{},
			path:           "/test",
			allows:         true,
			mustBeReadOnly: false,
		},
		"non-matching": {
			psp: &policy.PodSecurityPolicy{
				Spec: policy.PodSecurityPolicySpec{
					AllowedHostPaths: []policy.AllowedHostPath{
						{
							PathPrefix: "/foo",
							ReadOnly:   true,
						},
					},
				},
			},
			path:           "/foobar",
			allows:         false,
			mustBeReadOnly: false,
		},
		"match on direct match": {
			psp: &policy.PodSecurityPolicy{
				Spec: policy.PodSecurityPolicySpec{
					AllowedHostPaths: []policy.AllowedHostPath{
						{
							PathPrefix: "/foo",
							ReadOnly:   true,
						},
					},
				},
			},
			path:           "/foo",
			allows:         true,
			mustBeReadOnly: true,
		},
		"match with trailing slash on host path": {
			psp: &policy.PodSecurityPolicy{
				Spec: policy.PodSecurityPolicySpec{
					AllowedHostPaths: []policy.AllowedHostPath{
						{PathPrefix: "/foo"},
					},
				},
			},
			path:           "/foo/",
			allows:         true,
			mustBeReadOnly: false,
		},
		"match with trailing slash on allowed path": {
			psp: &policy.PodSecurityPolicy{
				Spec: policy.PodSecurityPolicySpec{
					AllowedHostPaths: []policy.AllowedHostPath{
						{PathPrefix: "/foo/"},
					},
				},
			},
			path:           "/foo",
			allows:         true,
			mustBeReadOnly: false,
		},
		"match child directory": {
			psp: &policy.PodSecurityPolicy{
				Spec: policy.PodSecurityPolicySpec{
					AllowedHostPaths: []policy.AllowedHostPath{
						{
							PathPrefix: "/foo/",
							ReadOnly:   true,
						},
					},
				},
			},
			path:           "/foo/bar",
			allows:         true,
			mustBeReadOnly: true,
		},
		"non-matching parent directory": {
			psp: &policy.PodSecurityPolicy{
				Spec: policy.PodSecurityPolicySpec{
					AllowedHostPaths: []policy.AllowedHostPath{
						{PathPrefix: "/foo/bar"},
					},
				},
			},
			path:           "/foo",
			allows:         false,
			mustBeReadOnly: false,
		},
	}

	for k, v := range tests {
		allows, mustBeReadOnly := AllowsHostVolumePath(v.psp, v.path)
		if v.allows != allows {
			t.Errorf("allows: %s expected %t but got %t", k, v.allows, allows)
		}
		if v.mustBeReadOnly != mustBeReadOnly {
			t.Errorf("mustBeReadOnly: %s expected %t but got %t", k, v.mustBeReadOnly, mustBeReadOnly)
		}
	}
}

func TestEqualStringSlices(t *testing.T) {
	tests := map[string]struct {
		arg1           []string
		arg2           []string
		expectedResult bool
	}{
		"nil equals to nil": {
			arg1:           nil,
			arg2:           nil,
			expectedResult: true,
		},
		"equal by size": {
			arg1:           []string{"1", "1"},
			arg2:           []string{"1", "1"},
			expectedResult: true,
		},
		"not equal by size": {
			arg1:           []string{"1"},
			arg2:           []string{"1", "1"},
			expectedResult: false,
		},
		"not equal by elements": {
			arg1:           []string{"1", "1"},
			arg2:           []string{"1", "2"},
			expectedResult: false,
		},
	}

	for k, v := range tests {
		if result := EqualStringSlices(v.arg1, v.arg2); result != v.expectedResult {
			t.Errorf("%s expected to return %t but got %t", k, v.expectedResult, result)
		}
	}
}

func TestIsOnlyServiceAccountTokenSources(t *testing.T) {
	serviceAccountToken := api.VolumeProjection{
		ServiceAccountToken: &api.ServiceAccountTokenProjection{
			Path:              "token",
			ExpirationSeconds: serviceaccount.WarnOnlyBoundTokenExpirationSeconds,
		}}
	configMap := api.VolumeProjection{
		ConfigMap: &api.ConfigMapProjection{
			LocalObjectReference: api.LocalObjectReference{
				Name: "kube-root-ca.crt",
			},
			Items: []api.KeyToPath{
				{
					Key:  "ca.crt",
					Path: "ca.crt",
				},
			},
		},
	}
	downwardAPI := api.VolumeProjection{
		DownwardAPI: &api.DownwardAPIProjection{
			Items: []api.DownwardAPIVolumeFile{
				{
					Path: "namespace",
					FieldRef: &api.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.namespace",
					},
				},
			},
		},
	}

	tests := []struct {
		desc   string
		volume *api.ProjectedVolumeSource
		want   bool
	}{
		{
			desc: "deny if ServiceAccountToken has wrong path",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					{ServiceAccountToken: &api.ServiceAccountTokenProjection{
						Path:              "notatoken",
						ExpirationSeconds: serviceaccount.WarnOnlyBoundTokenExpirationSeconds,
					}},
					configMap,
					downwardAPI,
				},
			},
		},
		{
			desc: "deny if ServiceAccountToken has wrong audience",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					{ServiceAccountToken: &api.ServiceAccountTokenProjection{
						Path:              "token",
						Audience:          "not api server",
						ExpirationSeconds: serviceaccount.WarnOnlyBoundTokenExpirationSeconds,
					}},
					configMap,
					downwardAPI,
				},
			},
		},
		{
			desc: "deny if CondigMap has wrong LocalObjectReference.Name",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					serviceAccountToken,
					{
						ConfigMap: &api.ConfigMapProjection{
							LocalObjectReference: api.LocalObjectReference{
								Name: "foo-ca.crt",
							},
							Items: []api.KeyToPath{
								{
									Key:  "ca.crt",
									Path: "ca.crt",
								},
							},
						},
					},
					downwardAPI,
				},
			},
		},
		{
			desc: "deny if DownwardAPI has wrong path",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					serviceAccountToken,
					configMap,
					{
						DownwardAPI: &api.DownwardAPIProjection{
							Items: []api.DownwardAPIVolumeFile{
								{
									Path: "foo",
									FieldRef: &api.ObjectFieldSelector{
										APIVersion: "v1",
										FieldPath:  "metadata.namespace",
									},
								},
							},
						},
					},
				},
			},
		},
		{
			desc: "deny if DownwardAPI has nil field ref",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					serviceAccountToken,
					configMap,
					{
						DownwardAPI: &api.DownwardAPIProjection{
							Items: []api.DownwardAPIVolumeFile{
								{
									Path: "namespace",
								},
							},
						},
					},
				},
			},
		},
		{
			desc: "deny if DownwardAPI has wrong api version",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					serviceAccountToken,
					configMap,
					{
						DownwardAPI: &api.DownwardAPIProjection{
							Items: []api.DownwardAPIVolumeFile{
								{
									Path: "namespace",
									FieldRef: &api.ObjectFieldSelector{
										APIVersion: "v1beta1",
										FieldPath:  "metadata.namespace",
									},
								},
							},
						},
					},
				},
			},
		},
		{
			desc: "deny if DownwardAPI has wrong field path",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					serviceAccountToken,
					configMap,
					{
						DownwardAPI: &api.DownwardAPIProjection{
							Items: []api.DownwardAPIVolumeFile{
								{
									Path: "namespace",
									FieldRef: &api.ObjectFieldSelector{
										APIVersion: "v1",
										FieldPath:  "metadata.foo",
									},
								},
							},
						},
					},
				},
			},
		},
		{
			desc: "deny if Secret exists",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					{
						Secret: &api.SecretProjection{},
					},
					configMap,
					downwardAPI,
					serviceAccountToken,
				},
			},
		},
		{
			desc: "deny if none of ServiceAccountToken, ConfigMap and DownwardAPI exist",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					{},
				},
			},
		},
		{
			desc: "allow if any of ServiceAccountToken, ConfigMap and DownwardAPI matches",
			volume: &api.ProjectedVolumeSource{
				Sources: []api.VolumeProjection{
					configMap,
					downwardAPI,
					serviceAccountToken,
				},
			},
			want: true,
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.desc, func(t *testing.T) {
			t.Parallel()

			if got := IsOnlyServiceAccountTokenSources(test.volume); got != test.want {
				t.Errorf("IsOnlyServiceAccountTokenSources(%+v) = %v, want %v", test.volume, got, test.want)
			}
		})
	}
}
