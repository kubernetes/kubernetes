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

package cmd

import (
	"bytes"
	"io"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest/fake"
	api "k8s.io/kubernetes/pkg/apis/core"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

type testPVCPrinter struct {
	CachedPVC *api.PersistentVolumeClaim
}

func (t *testPVCPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	t.CachedPVC = obj.(*api.PersistentVolumeClaim)
	return nil
}

func (t *testPVCPrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

func (t *testPVCPrinter) HandledResources() []string {
	return []string{}
}

func (t *testPVCPrinter) IsGeneric() bool {
	return true
}

func TestCreatePersistentVolumeClaim(t *testing.T) {
	pvcName, storageClass := "mypvc", "foo"
	capacityQuantity, _ := resource.ParseQuantity("1Gi")
	resources := api.ResourceRequirements{
		Requests: api.ResourceList{
			"storage": capacityQuantity,
		},
	}

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	printer := &testPVCPrinter{}
	tf.Printer = printer
	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{}
	tf.ClientConfig = defaultClientConfig()

	tests := map[string]struct {
		accessmode   string
		capacity     string
		storageClass string
		expectedPVC  *api.PersistentVolumeClaim
	}{
		"test-accessmode-abbreviation": {
			accessmode: "RWO",
			capacity:   "1Gi",
			expectedPVC: &api.PersistentVolumeClaim{
				ObjectMeta: v1.ObjectMeta{
					Name: pvcName,
				},
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
					Resources:   resources,
				},
			},
		},
		"test-multiple-accessmodes": {
			accessmode: "RWO,RWX",
			capacity:   "1Gi",
			expectedPVC: &api.PersistentVolumeClaim{
				ObjectMeta: v1.ObjectMeta{
					Name: pvcName,
				},
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce, api.ReadWriteMany},
					Resources:   resources,
				},
			},
		},
		"test-storageclass": {
			accessmode:   "ReadWriteOnce,ROX",
			capacity:     "1Gi",
			storageClass: storageClass,
			expectedPVC: &api.PersistentVolumeClaim{
				ObjectMeta: v1.ObjectMeta{
					Name: pvcName,
				},
				Spec: api.PersistentVolumeClaimSpec{
					AccessModes:      []api.PersistentVolumeAccessMode{api.ReadWriteOnce, api.ReadOnlyMany},
					Resources:        resources,
					StorageClassName: &storageClass,
				},
			},
		},
	}

	for name, test := range tests {
		buf := bytes.NewBuffer([]byte{})
		cmd := NewCmdCreatePersistentVolumeClaim(f, buf)
		cmd.Flags().Set("dry-run", "true")
		cmd.Flags().Set("output", "object")
		cmd.Flags().Set("access-mode", test.accessmode)
		cmd.Flags().Set("capacity", test.capacity)
		if test.storageClass != "" {
			cmd.Flags().Set("storage-class", test.storageClass)
		}
		cmd.Run(cmd, []string{pvcName})

		expectedPVC, realPVC := test.expectedPVC, printer.CachedPVC

		if expectedPVC.Name != realPVC.Name {
			t.Errorf("%s:\nexpected name:\n%#v\nsaw name:\n%#v", name, expectedPVC.Name, realPVC.Name)
		}

		if !reflect.DeepEqual(expectedPVC.Spec.AccessModes, realPVC.Spec.AccessModes) {
			t.Errorf("%s:\nexpected access modes:\n%#v\nsaw access modes:\n%#v", name, expectedPVC.Spec.AccessModes, realPVC.Spec.AccessModes)
		}

		if !reflect.DeepEqual(expectedPVC.Spec.Resources, realPVC.Spec.Resources) {
			t.Errorf("%s:\nexpected capacity:\n%#v\nsaw capacity:\n%#v", name, expectedPVC.Spec.Resources, realPVC.Spec.Resources)
		}

		if test.storageClass != "" && *expectedPVC.Spec.StorageClassName != *realPVC.Spec.StorageClassName {
			t.Errorf("%s:\nexpected storage class:\n%#v\nsaw storage class:\n%#v", name, *expectedPVC.Spec.StorageClassName, *realPVC.Spec.StorageClassName)
		}
	}
}

func TestCreatePVCOptionsValidate(t *testing.T) {
	capacityQuantity, _ := resource.ParseQuantity("1Gi")

	tests := map[string]struct {
		pvcOptions *CreatePVCOptions
		expectErr  bool
	}{
		"test-missing-name": {
			pvcOptions: &CreatePVCOptions{},
			expectErr:  true,
		},
		"test-missing-accessmode": {
			pvcOptions: &CreatePVCOptions{
				Name: "mypvc",
			},
			expectErr: true,
		},
		"test-missing-capacity": {
			pvcOptions: &CreatePVCOptions{
				Name:        "mypvc",
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany},
			},
			expectErr: true,
		},
		"test-valid-case": {
			pvcOptions: &CreatePVCOptions{
				Name:        "mypvc",
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadOnlyMany},
				Capacity:    &capacityQuantity,
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		err := test.pvcOptions.Validate()
		if test.expectErr && err == nil {
			t.Errorf("%s: expect error happens but validate passes.", name)
		}
		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
	}
}

func TestCreatePVCOptionsComplete(t *testing.T) {
	pvcName, emptyStorageClass, storageClass := "mypvc", "", "foo"
	capacityQuantity, _ := resource.ParseQuantity("1Gi")

	f, tf, _, _ := cmdtesting.NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{}
	tf.ClientConfig = defaultClientConfig()

	buf := bytes.NewBuffer([]byte{})

	tests := map[string]struct {
		params                []string
		accessMode            string
		capacity              string
		storageClass          string
		storageClassSpecified bool
		expected              *CreatePVCOptions
		expectErr             bool
	}{
		"test-missing-name": {
			params:     []string{},
			accessMode: "RWO",
			capacity:   "1Gi",
			expectErr:  true,
		},
		"test-accessmode-abbreviation": {
			params:     []string{pvcName},
			accessMode: "RWO",
			capacity:   "1Gi",
			expected: &CreatePVCOptions{
				Name: pvcName,
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
				Capacity: &capacityQuantity,
			},
			expectErr: false,
		},
		"test-multiple-accessmodes": {
			params:     []string{pvcName},
			accessMode: "RWO,RWX",
			capacity:   "1Gi",
			expected: &CreatePVCOptions{
				Name: pvcName,
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
					api.ReadWriteMany,
				},
				Capacity: &capacityQuantity,
			},
			expectErr: false,
		},
		"test-invalid-capacity": {
			params:     []string{pvcName},
			accessMode: "RWO",
			capacity:   "invalid",
			expectErr:  true,
		},
		"test-empty-storageclass": {
			params:                []string{pvcName},
			accessMode:            "RWO",
			capacity:              "1Gi",
			storageClassSpecified: true,
			expected: &CreatePVCOptions{
				Name: pvcName,
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
				Capacity:     &capacityQuantity,
				StorageClass: &emptyStorageClass,
			},
			expectErr: false,
		},
		"test-valid-complete-case": {
			params:                []string{pvcName},
			accessMode:            "ReadWriteOnce",
			capacity:              "1Gi",
			storageClass:          "foo",
			storageClassSpecified: true,
			expected: &CreatePVCOptions{
				Name: pvcName,
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
				Capacity:     &capacityQuantity,
				StorageClass: &storageClass,
			},
			expectErr: false,
		},
	}

	for name, test := range tests {
		pvcOptions := &CreatePVCOptions{}
		cmd := NewCmdCreatePersistentVolumeClaim(f, buf)

		cmd.Flags().Set("access-mode", test.accessMode)
		cmd.Flags().Set("capacity", test.capacity)

		if test.storageClassSpecified {
			cmd.Flags().Set("storage-class", test.storageClass)
		}

		err := pvcOptions.Complete(f, cmd, test.params)

		if !test.expectErr && err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}

		if test.expectErr {
			if err != nil {
				continue
			} else {
				t.Errorf("%s: expect error happens but test passes.", name)
			}
		}

		if pvcOptions.Name != test.expected.Name {
			t.Errorf("%s:\nexpected name:\n%#v\nsaw name:\n%#v", name, test.expected.Name, pvcOptions.Name)
		}

		if !reflect.DeepEqual(pvcOptions.AccessModes, test.expected.AccessModes) {
			t.Errorf("%s:\nexpected access modes:\n%#v\nsaw access modes:\n%#v", name, test.expected.AccessModes, pvcOptions.AccessModes)
		}

		if !reflect.DeepEqual(pvcOptions.Capacity, test.expected.Capacity) {
			t.Errorf("%s:\nexpected capacity:\n%#v\nsaw capacity:\n%#v", name, test.expected.Capacity, pvcOptions.Capacity)
		}

		if test.storageClassSpecified && *pvcOptions.StorageClass != *test.expected.StorageClass {
			t.Errorf("%s:\nexpected storage class:\n%#v\nsaw storage class:\n%#v", name, *test.expected.StorageClass, *pvcOptions.StorageClass)
		} else if !test.storageClassSpecified && pvcOptions.StorageClass != nil {
			t.Errorf("%s:\nexpected no storage class specified, but saw:\n%#v", name, *pvcOptions.StorageClass)
		}
	}
}
