/*
Copyright 2017 The Kubernetes Authors.

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

package v1alpha1

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

var scheme = runtime.NewScheme()

func init() {
	addKnownTypes(scheme)
	internalGV := schema.GroupVersion{Group: auditinternal.GroupName, Version: runtime.APIVersionInternal}
	scheme.AddKnownTypes(internalGV,
		&auditinternal.Event{},
	)
	RegisterConversions(scheme)
}

func TestConversion(t *testing.T) {
	scheme.Log(t)

	testcases := []struct {
		desc     string
		old      *ObjectReference
		expected *auditinternal.ObjectReference
	}{
		{
			"core group",
			&ObjectReference{
				APIVersion: "/v1",
			},
			&auditinternal.ObjectReference{
				APIVersion: "v1",
				APIGroup:   "",
			},
		},
		{
			"other groups",
			&ObjectReference{
				APIVersion: "rbac.authorization.k8s.io/v1beta1",
			},
			&auditinternal.ObjectReference{
				APIVersion: "v1beta1",
				APIGroup:   "rbac.authorization.k8s.io",
			},
		},
		{
			"all empty",
			&ObjectReference{},
			&auditinternal.ObjectReference{},
		},
		{
			"invalid apiversion should not cause painc",
			&ObjectReference{
				APIVersion: "invalid version without slash",
			},
			&auditinternal.ObjectReference{
				APIVersion: "invalid version without slash",
				APIGroup:   "",
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			internal := &auditinternal.ObjectReference{}
			if err := scheme.Convert(tc.old, internal, nil); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(internal, tc.expected) {
				t.Errorf("expected\n\t%#v, got \n\t%#v", tc.expected, internal)
			}
		})
	}
}
