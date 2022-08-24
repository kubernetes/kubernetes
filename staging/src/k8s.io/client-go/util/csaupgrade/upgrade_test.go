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
		CSAManager     string
		SSAManager     string
		OriginalObject []byte
		ExpectedObject []byte
	}{
		{
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
`),
		},
	}

	for _, testCase := range cases {
		initialObject := unstructured.Unstructured{}
		err := yaml.Unmarshal(testCase.OriginalObject, &initialObject)
		if err != nil {
			t.Fatal(err)
		}

		upgraded, err := csaupgrade.UpgradeManagedFields(
			&initialObject,
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
	}
}
