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
	// Initial object has managed fields from using CSA
	originalYAML := []byte(`
apiVersion: v1
data:
  key: value
  legacy: unused
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
  resourceVersion: "12502"
  uid: 1f186675-58e6-4d7b-8bc5-7ece581e3013
`)
	initialObject := unstructured.Unstructured{}
	err := yaml.Unmarshal(originalYAML, &initialObject)
	if err != nil {
		t.Fatal(err)
	}

	upgraded, err := csaupgrade.UpgradeManagedFields(&initialObject, "kubectl-client-side-apply", "kubectl", "")
	if err != nil {
		t.Fatal(err)
	}

	expectedYAML := []byte(`
apiVersion: v1
data:
  key: value
  legacy: unused
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
    time: "2022-08-22T23:08:23Z"
  name: test
  namespace: default
  resourceVersion: "12502"
  uid: 1f186675-58e6-4d7b-8bc5-7ece581e3013
`)

	expectedObject := unstructured.Unstructured{}
	err = yaml.Unmarshal(expectedYAML, &expectedObject)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(&expectedObject, upgraded) {
		t.Fatal(cmp.Diff(&expectedObject, upgraded))
	}
}
