/*
Copyright 2022 The Kubernetes Authors.

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

package csaupgrade_test

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/util/csaupgrade"
)

func TestUpgradeCSA(t *testing.T) {

	cases := []struct {
		Name           string
		CSAManager     string
		SSAManager     string
		OriginalObject []byte
		ExpectedObject []byte
	}{
		{
			// Case where there is a CSA entry with the given name, but no SSA entry
			// is found. Expect that the CSA entry is converted to an SSA entry
			// and renamed.
			Name:       "csa-basic-direct-conversion",
			CSAManager: "kubectl-client-side-apply",
			SSAManager: "kubectl",
			OriginalObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations:
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"v1","data":{"key":"value","legacy":"unused"},"kind":"ConfigMap","metadata":{"annotations":{},"name":"test","namespace":"default"}}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations:
          .: {}
          f:kubectl.kubernetes.io/last-applied-configuration: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  name: test
  namespace: default
`),
			ExpectedObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations: {}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations: {}
    manager: kubectl
    operation: Apply
    time: "2022-08-22T23:08:23Z"
  name: test
  namespace: default  
`),
		},
		{
			// This is the case when kubectl --server-side is used for the first time
			// Server creates duplicate managed fields entry - one for Update and another
			// for Apply. Expect entries to be merged into one entry, which is unchanged
			// from initial SSA.
			Name:       "csa-combine-with-ssa-duplicate-keys",
			CSAManager: "kubectl-client-side-apply",
			SSAManager: "kubectl",
			OriginalObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations:
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"v1","data":{"key":"value","legacy":"unused"},"kind":"ConfigMap","metadata":{"annotations":{},"name":"test","namespace":"default"}}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations:
          .: {}
          f:kubectl.kubernetes.io/last-applied-configuration: {}
    manager: kubectl
    operation: Apply
    time: "2022-08-23T23:08:23Z"
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations:
          .: {}
          f:kubectl.kubernetes.io/last-applied-configuration: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  name: test
  namespace: default
`),
			ExpectedObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations: {}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations: {}
    manager: kubectl
    operation: Apply
    time: "2022-08-23T23:08:23Z"
  name: test
  namespace: default  
`),
		},
		{
			// This is the case when kubectl --server-side is used for the first time,
			// but then a key is removed. A bug would take place where key is left in
			// CSA entry but no longer present in SSA entry, so it would not be pruned.
			// This shows that upgrading such an object results in correct behavior next
			// time SSA applier
			// Expect final object to have unioned keys from both entries
			Name:       "csa-combine-with-ssa-additional-keys",
			CSAManager: "kubectl-client-side-apply",
			SSAManager: "kubectl",
			OriginalObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations:
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"v1","data":{"key":"value","legacy":"unused"},"kind":"ConfigMap","metadata":{"annotations":{},"name":"test","namespace":"default"}}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
      f:metadata:
        f:annotations:
          .: {}
          f:kubectl.kubernetes.io/last-applied-configuration: {}
    manager: kubectl
    operation: Apply
    time: "2022-08-23T23:08:23Z"
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations:
          .: {}
          f:kubectl.kubernetes.io/last-applied-configuration: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  name: test
  namespace: default
`),
			ExpectedObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations: {}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations: {}
    manager: kubectl
    operation: Apply
    time: "2022-08-23T23:08:23Z"
  name: test
  namespace: default  
`),
		},
		{
			// Case when there are multiple CSA versions on the object which do not
			// match the version from the apply entry. Shows they are tossed away
			// without being merged.
			Name:       "csa-no-applicable-version",
			CSAManager: "kubectl-client-side-apply",
			SSAManager: "kubectl",
			OriginalObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations:
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"v1","data":{"key":"value","legacy":"unused"},"kind":"ConfigMap","metadata":{"annotations":{},"name":"test","namespace":"default"}}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v5
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations:
          .: {}
          f:kubectl.kubernetes.io/last-applied-configuration: {}
    manager: kubectl
    operation: Apply
    time: "2022-08-23T23:08:23Z"
  - apiVersion: v1
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        f:key2: {}
      f:metadata:
        f:annotations:
          f:hello2: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  - apiVersion: v2
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        f:key3: {}
      f:metadata:
        f:annotations:
          f:hello3: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  - apiVersion: v3
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        f:key4: {}
      f:metadata:
        f:annotations:
          f:hello3: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  - apiVersion: v4
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        f:key5: {}
      f:metadata:
        f:annotations:
          f:hello4: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  name: test
  namespace: default
`),
			ExpectedObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations: {}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v5
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations: {}
    manager: kubectl
    operation: Apply
    time: "2022-08-23T23:08:23Z"
  name: test
  namespace: default  
`),
		},
		{
			// Case when there are multiple CSA versions on the object which do not
			// match the version from the apply entry, and one which does.
			// Shows that CSA entry with matching version is unioned into the SSA entry.
			Name:       "csa-single-applicable-version",
			CSAManager: "kubectl-client-side-apply",
			SSAManager: "kubectl",
			OriginalObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations:
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"v1","data":{"key":"value","legacy":"unused"},"kind":"ConfigMap","metadata":{"annotations":{},"name":"test","namespace":"default"}}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v5
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:legacy: {}
      f:metadata:
        f:annotations:
          .: {}
          f:kubectl.kubernetes.io/last-applied-configuration: {}
    manager: kubectl
    operation: Apply
    time: "2022-08-23T23:08:23Z"
  - apiVersion: v5
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        f:key2: {}
      f:metadata:
        f:annotations:
          f:hello2: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  - apiVersion: v2
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        f:key3: {}
      f:metadata:
        f:annotations:
          f:hello3: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  - apiVersion: v3
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        f:key4: {}
      f:metadata:
        f:annotations:
          f:hello4: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  - apiVersion: v4
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        f:key5: {}
      f:metadata:
        f:annotations:
          f:hello5: {}
    manager: kubectl-client-side-apply
    operation: Update
    time: "2022-08-22T23:08:23Z"
  name: test
  namespace: default
`),
			ExpectedObject: []byte(`
apiVersion: v1
data: {}
kind: ConfigMap
metadata:
  annotations: {}
  creationTimestamp: "2022-08-22T23:08:23Z"
  managedFields:
  - apiVersion: v5
    fieldsType: FieldsV1
    fieldsV1:
      f:data:
        .: {}
        f:key: {}
        f:key2: {}
        f:legacy: {}
      f:metadata:
        f:annotations:
          .: {}
          f:hello2: {}
    manager: kubectl
    operation: Apply
    time: "2022-08-23T23:08:23Z"
  name: test
  namespace: default  
`),
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.Name, func(t *testing.T) {
			initialObject := unstructured.Unstructured{}
			err := yaml.Unmarshal(testCase.OriginalObject, &initialObject)
			if err != nil {
				t.Fatal(err)
			}

			upgraded := initialObject.DeepCopy()
			err = csaupgrade.UpgradeManagedFields(
				upgraded,
				testCase.CSAManager,
				testCase.SSAManager,
			)

			if err != nil {
				t.Fatal(err)
			}

			expectedObject := unstructured.Unstructured{}
			err = yaml.Unmarshal(testCase.ExpectedObject, &expectedObject)
			if err != nil {
				t.Fatal(err)
			}

			if !reflect.DeepEqual(&expectedObject, upgraded) {
				t.Fatal(cmp.Diff(&expectedObject, upgraded))
			}
		})
	}
}
