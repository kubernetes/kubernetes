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

package v1beta3_test

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/resource"
	versioned "k8s.io/kubernetes/pkg/api/v1beta3"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/runtime"
	sccutil "k8s.io/kubernetes/pkg/securitycontextconstraints/util"
)

var v1beta3Codec = api.Codecs.LegacyCodec(versioned.SchemeGroupVersion)

func TestResourceQuotaStatusConversion(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	// should serialize as "0"
	expected := resource.NewQuantity(int64(0), resource.DecimalSI)
	if "0" != expected.String() {
		t.Errorf("Expected: 0, Actual: %v, do not require units", expected.String())
	}

	parsed := resource.MustParse("0")
	if "0" != parsed.String() {
		t.Errorf("Expected: 0, Actual: %v, do not require units", parsed.String())
	}

	quota := &api.ResourceQuota{}
	quota.Status = api.ResourceQuotaStatus{}
	quota.Status.Hard = api.ResourceList{}
	quota.Status.Used = api.ResourceList{}
	quota.Status.Hard[api.ResourcePods] = *expected

	// round-trip the object
	object := roundTrip(t, quota)
	after := object.(*api.ResourceQuota)

	actualQuantity := after.Status.Hard[api.ResourcePods]
	actual := &actualQuantity

	// should be "0", but was "0m"
	if expected.String() != actual.String() {
		t.Errorf("Expected %v, Actual %v", expected.String(), actual.String())
	}
}

func TestNodeConversion(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	obj, err := runtime.Decode(v1beta3Codec, []byte(`{"kind":"Minion","apiVersion":"v1beta3"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := obj.(*api.Node); !ok {
		t.Errorf("unexpected type: %#v", obj)
	}

	obj, err = runtime.Decode(v1beta3Codec, []byte(`{"kind":"MinionList","apiVersion":"v1beta3"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := obj.(*api.NodeList); !ok {
		t.Errorf("unexpected type: %#v", obj)
	}

	obj = &api.Node{}
	if err := runtime.DecodeInto(v1beta3Codec, []byte(`{"kind":"Minion","apiVersion":"v1beta3"}`), obj); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestBadSecurityContextConversion(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	priv := false
	testCases := map[string]struct {
		c   *versioned.Container
		err string
	}{
		// this use case must use true for the container and false for the sc. Otherwise the defaulter
		// will assume privileged was left undefined (since it is the default value) and copy the
		// sc setting upwards
		"mismatched privileged": {
			c: &versioned.Container{
				Privileged: true,
				SecurityContext: &versioned.SecurityContext{
					Privileged: &priv,
				},
			},
			err: "container privileged settings do not match security context settings, cannot convert",
		},
		"mismatched caps add": {
			c: &versioned.Container{
				Capabilities: versioned.Capabilities{
					Add: []versioned.Capability{"foo"},
				},
				SecurityContext: &versioned.SecurityContext{
					Capabilities: &versioned.Capabilities{
						Add: []versioned.Capability{"bar"},
					},
				},
			},
			err: "container capability settings do not match security context settings, cannot convert",
		},
		"mismatched caps drop": {
			c: &versioned.Container{
				Capabilities: versioned.Capabilities{
					Drop: []versioned.Capability{"foo"},
				},
				SecurityContext: &versioned.SecurityContext{
					Capabilities: &versioned.Capabilities{
						Drop: []versioned.Capability{"bar"},
					},
				},
			},
			err: "container capability settings do not match security context settings, cannot convert",
		},
	}

	for k, v := range testCases {
		got := api.Container{}
		err := api.Scheme.Convert(v.c, &got)
		if err == nil {
			t.Errorf("expected error for case %s but got none", k)
		} else {
			if err.Error() != v.err {
				t.Errorf("unexpected error for case %s.  Expected: %s but got: %s", k, v.err, err.Error())
			}
		}
	}

}

func TestSCCVolumeConversionsApiToV1Beta3(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	tests := map[string]struct {
		scc                      *api.SecurityContextConstraints
		allowHostDirVolumePlugin bool
	}{
		"empty volumes": {
			scc: &api.SecurityContextConstraints{},
			allowHostDirVolumePlugin: false,
		},
		"no host dir": {
			scc: &api.SecurityContextConstraints{
				Volumes: []api.FSType{api.FSTypeAWSElasticBlockStore},
			},
			allowHostDirVolumePlugin: false,
		},
		"all volumes": {
			scc: &api.SecurityContextConstraints{
				Volumes: []api.FSType{api.FSTypeAll},
			},
			allowHostDirVolumePlugin: true,
		},
		"host dir": {
			scc: &api.SecurityContextConstraints{
				Volumes: []api.FSType{api.FSTypeHostPath},
			},
			allowHostDirVolumePlugin: true,
		},
	}

	for k, v := range tests {
		got := versioned.SecurityContextConstraints{}
		err := api.Scheme.Convert(v.scc, &got)

		if err != nil {
			t.Errorf("%s received an error during conversion: %v", k, err)
			continue
		}

		if got.AllowHostDirVolumePlugin != v.allowHostDirVolumePlugin {
			t.Errorf("%s expected hostDir to be %t but received %t", k, v.allowHostDirVolumePlugin, got.AllowHostDirVolumePlugin)
		}
	}
}

func TestSCCVolumeConversionsV1Beta3ToApi(t *testing.T) {
	if !registered.IsAllowedVersion(versioned.SchemeGroupVersion) {
		return
	}

	noHostDirVolumes := []api.FSType{}
	for _, v := range sccutil.GetAllFSTypesExcept(string(api.FSTypeHostPath)).List() {
		noHostDirVolumes = append(noHostDirVolumes, api.FSType(v))
	}

	tests := map[string]struct {
		scc     *versioned.SecurityContextConstraints
		volumes []api.FSType
	}{
		"hostDir false": {
			scc: &versioned.SecurityContextConstraints{
				AllowHostDirVolumePlugin: false,
			},
			volumes: noHostDirVolumes,
		},
		"hostDir true": {
			scc: &versioned.SecurityContextConstraints{
				AllowHostDirVolumePlugin: true,
			},
			volumes: []api.FSType{api.FSTypeAll},
		},
	}

	for k, v := range tests {
		got := api.SecurityContextConstraints{}
		err := api.Scheme.Convert(v.scc, &got)

		if err != nil {
			t.Errorf("%s received an error during conversion: %v", k, err)
			continue
		}
		if !reflect.DeepEqual(got.Volumes, v.volumes) {
			t.Errorf("%s expected volumes: %v but received: %v", k, v.volumes, got.Volumes)
		}
	}
}
