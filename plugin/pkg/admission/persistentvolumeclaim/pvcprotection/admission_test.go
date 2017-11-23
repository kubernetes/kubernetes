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

package pvcprotection

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/controller"
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
	claimWithFinalizer := claim.DeepCopy()
	claimWithFinalizer.Finalizers = []string{volumeutil.PVCProtectionFinalizer}

	tests := []struct {
		name           string
		object         runtime.Object
		expectedObject runtime.Object
		featureEnabled bool
	}{
		{
			"create -> add finalizer",
			claim,
			claimWithFinalizer,
			true,
		},
		{
			"finalizer already exists -> no new finalizer",
			claimWithFinalizer,
			claimWithFinalizer,
			true,
		},
		{
			"disabled feature -> no finalizer",
			claim,
			claim,
			false,
		},
	}

	ctrl := newPlugin()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	ctrl.SetInternalKubeInformerFactory(informerFactory)

	for _, test := range tests {
		feature.DefaultFeatureGate.Set(fmt.Sprintf("PVCProtection=%v", test.featureEnabled))
		obj := test.object.DeepCopyObject()
		attrs := admission.NewAttributesRecord(
			obj,                  // new object
			obj.DeepCopyObject(), // old object, copy to be sure it's not modified
			api.Kind("PersistentVolumeClaim").WithVersion("version"),
			claim.Namespace,
			claim.Name,
			api.Resource("persistentvolumeclaims").WithVersion("version"),
			"", // subresource
			admission.Create,
			nil, // userInfo
		)

		err := ctrl.Admit(attrs)
		if err != nil {
			t.Errorf("Test %q: got unexpected error: %v", test.name, err)
		}
		if !reflect.DeepEqual(test.expectedObject, obj) {
			t.Errorf("Test %q: Expected object:\n%s\ngot:\n%s", test.name, spew.Sdump(test.expectedObject), spew.Sdump(obj))
		}
	}

	// Disable the feature for rest of the tests.
	// TODO: remove after alpha
	feature.DefaultFeatureGate.Set("PVCProtection=false")
}
