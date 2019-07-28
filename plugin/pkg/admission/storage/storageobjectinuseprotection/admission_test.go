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

package storageobjectinuseprotection

import (
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

func TestAdmit(t *testing.T) {
	claim := &api.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			Kind: "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claim",
			Namespace: "ns",
		},
	}

	pv := &api.PersistentVolume{
		TypeMeta: metav1.TypeMeta{
			Kind: "PersistentVolume",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "pv",
		},
	}
	claimWithFinalizer := claim.DeepCopy()
	claimWithFinalizer.Finalizers = []string{volumeutil.PVCProtectionFinalizer}

	pvWithFinalizer := pv.DeepCopy()
	pvWithFinalizer.Finalizers = []string{volumeutil.PVProtectionFinalizer}

	tests := []struct {
		name           string
		resource       schema.GroupVersionResource
		object         runtime.Object
		expectedObject runtime.Object
		featureEnabled bool
		namespace      string
	}{
		{
			"create -> add finalizer",
			api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
			claim,
			claimWithFinalizer,
			true,
			claim.Namespace,
		},
		{
			"finalizer already exists -> no new finalizer",
			api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
			claimWithFinalizer,
			claimWithFinalizer,
			true,
			claimWithFinalizer.Namespace,
		},
		{
			"disabled feature -> no finalizer",
			api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
			claim,
			claim,
			false,
			claim.Namespace,
		},
		{
			"create -> add finalizer",
			api.SchemeGroupVersion.WithResource("persistentvolumes"),
			pv,
			pvWithFinalizer,
			true,
			pv.Namespace,
		},
		{
			"finalizer already exists -> no new finalizer",
			api.SchemeGroupVersion.WithResource("persistentvolumes"),
			pvWithFinalizer,
			pvWithFinalizer,
			true,
			pvWithFinalizer.Namespace,
		},
		{
			"disabled feature -> no finalizer",
			api.SchemeGroupVersion.WithResource("persistentvolumes"),
			pv,
			pv,
			false,
			pv.Namespace,
		},
	}

	ctrl := newPlugin()

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageObjectInUseProtection, test.featureEnabled)()
			obj := test.object.DeepCopyObject()
			attrs := admission.NewAttributesRecord(
				obj,                  // new object
				obj.DeepCopyObject(), // old object, copy to be sure it's not modified
				schema.GroupVersionKind{},
				test.namespace,
				"foo",
				test.resource,
				"", // subresource
				admission.Create,
				&metav1.CreateOptions{},
				false, // dryRun
				nil,   // userInfo
			)

			err := ctrl.Admit(attrs, nil)
			if err != nil {
				t.Errorf("Test %q: got unexpected error: %v", test.name, err)
			}
			if !reflect.DeepEqual(test.expectedObject, obj) {
				t.Errorf("Test %q: Expected object:\n%s\ngot:\n%s", test.name, spew.Sdump(test.expectedObject), spew.Sdump(obj))
			}
		})
	}
}
