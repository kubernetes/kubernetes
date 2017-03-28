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

package admission

import (
	"testing"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/storage"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/util"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/controller"
)

func TestAdmission(t *testing.T) {
	empty := ""
	foo := "foo"

	defaultClass1 := &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "default1",
			Annotations: map[string]string{
				storageutil.IsDefaultStorageClassAnnotation: "true",
			},
		},
		Provisioner: "default1",
	}
	defaultClass2 := &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "default2",
			Annotations: map[string]string{
				storageutil.IsDefaultStorageClassAnnotation: "true",
			},
		},
		Provisioner: "default2",
	}
	// Class that has explicit default = false
	classWithFalseDefault := &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "nondefault1",
			Annotations: map[string]string{
				storageutil.IsDefaultStorageClassAnnotation: "false",
			},
		},
		Provisioner: "nondefault1",
	}
	// Class with missing default annotation (=non-default)
	classWithNoDefault := &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "nondefault2",
		},
		Provisioner: "nondefault1",
	}
	// Class with empty default annotation (=non-default)
	classWithEmptyDefault := &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "nondefault2",
			Annotations: map[string]string{
				storageutil.IsDefaultStorageClassAnnotation: "",
			},
		},
		Provisioner: "nondefault1",
	}

	claimWithClass := &api.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			Kind: "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claimWithClass",
			Namespace: "ns",
		},
		Spec: api.PersistentVolumeClaimSpec{
			StorageClassName: &foo,
		},
	}
	claimWithEmptyClass := &api.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			Kind: "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claimWithEmptyClass",
			Namespace: "ns",
		},
		Spec: api.PersistentVolumeClaimSpec{
			StorageClassName: &empty,
		},
	}
	claimWithNoClass := &api.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			Kind: "PersistentVolumeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "claimWithNoClass",
			Namespace: "ns",
		},
	}

	tests := []struct {
		name              string
		classes           []*storage.StorageClass
		claim             *api.PersistentVolumeClaim
		expectError       bool
		expectedClassName string
	}{
		{
			"no default, no modification of PVCs",
			[]*storage.StorageClass{classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			claimWithNoClass,
			false,
			"",
		},
		{
			"one default, modify PVC with class=nil",
			[]*storage.StorageClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			claimWithNoClass,
			false,
			"default1",
		},
		{
			"one default, no modification of PVC with class=''",
			[]*storage.StorageClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			claimWithEmptyClass,
			false,
			"",
		},
		{
			"one default, no modification of PVC with class='foo'",
			[]*storage.StorageClass{defaultClass1, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			claimWithClass,
			false,
			"foo",
		},
		{
			"two defaults, error with PVC with class=nil",
			[]*storage.StorageClass{defaultClass1, defaultClass2, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			claimWithNoClass,
			true,
			"",
		},
		{
			"two defaults, no modification of PVC with class=''",
			[]*storage.StorageClass{defaultClass1, defaultClass2, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			claimWithEmptyClass,
			false,
			"",
		},
		{
			"two defaults, no modification of PVC with class='foo'",
			[]*storage.StorageClass{defaultClass1, defaultClass2, classWithFalseDefault, classWithNoDefault, classWithEmptyDefault},
			claimWithClass,
			false,
			"foo",
		},
	}

	for _, test := range tests {
		glog.V(4).Infof("starting test %q", test.name)

		// clone the claim, it's going to be modified
		clone, err := api.Scheme.DeepCopy(test.claim)
		if err != nil {
			t.Fatalf("Cannot clone claim: %v", err)
		}
		claim := clone.(*api.PersistentVolumeClaim)

		ctrl := newPlugin()
		informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
		ctrl.SetInternalKubeInformerFactory(informerFactory)
		for _, c := range test.classes {
			informerFactory.Storage().InternalVersion().StorageClasses().Informer().GetStore().Add(c)
		}
		attrs := admission.NewAttributesRecord(
			claim, // new object
			nil,   // old object
			api.Kind("PersistentVolumeClaim").WithVersion("version"),
			claim.Namespace,
			claim.Name,
			api.Resource("persistentvolumeclaims").WithVersion("version"),
			"", // subresource
			admission.Create,
			nil, // userInfo
		)
		err = ctrl.Admit(attrs)
		glog.Infof("Got %v", err)
		if err != nil && !test.expectError {
			t.Errorf("Test %q: unexpected error received: %v", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("Test %q: expected error and no error recevied", test.name)
		}

		class := ""
		if claim.Spec.StorageClassName != nil {
			class = *claim.Spec.StorageClassName
		}
		if test.expectedClassName != "" && test.expectedClassName != class {
			t.Errorf("Test %q: expected class name %q, got %q", test.name, test.expectedClassName, class)
		}
		if test.expectedClassName == "" && class != "" {
			t.Errorf("Test %q: expected class name %q, got %q", test.name, test.expectedClassName, class)
		}
	}
}
