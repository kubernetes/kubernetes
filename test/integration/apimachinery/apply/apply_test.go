/*
Copyright 2025 The Kubernetes Authors.

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

package dra

import (
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/testapigroup/install"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/controlplane"
	controlplaneapiserver "k8s.io/kubernetes/pkg/controlplane/apiserver"
	testapigrouprest "k8s.io/kubernetes/pkg/registry/testapigroup/rest"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"sigs.k8s.io/yaml"
)

func init() {
	install.Install(legacyscheme.Scheme)
}

func TestApply(t *testing.T) {
	tCtx := ktesting.Init(t)
	etcdOptions := framework.SharedEtcd()
	apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	apiServerFlags := framework.DefaultTestServerFlags()
	runtimeConfigs := []string{"testapigroup.apimachinery.k8s.io/v1=true"}
	apiServerFlags = append(apiServerFlags, "--runtime-config="+strings.Join(runtimeConfigs, ","))

	// Sanity check. Not protected against concurrent access, but that's
	// okay: integration tests are also run with race detection, so that
	// would catch it.
	if controlplane.AdditionalStorageProvidersForTests != nil {
		t.Fatal("cannot set AdditionalStorageProvidersForTests, already set")
	}
	t.Cleanup(func() {
		controlplane.AdditionalStorageProvidersForTests = nil
	})
	controlplane.AdditionalStorageProvidersForTests = func(client *kubernetes.Clientset) []controlplaneapiserver.RESTStorageProvider {
		return []controlplaneapiserver.RESTStorageProvider{
			testapigrouprest.RESTStorageProvider{NamespaceClient: client.CoreV1().Namespaces()},
		}
	}

	server := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions, apiServerFlags, etcdOptions)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Stopping the apiserver...")
		server.TearDownFn()
	})
	tCtx = ktesting.WithRESTConfig(tCtx, server.ClientConfig)

	// More sub-tests could be added here. Currently there's only one.
	tCtx.Run("optional-list-map-key", testOptionalListMapKey)
}

func testOptionalListMapKey(tCtx ktesting.TContext) {
	requireManagedFields := func(what string, obj *unstructured.Unstructured, expectedManagedFields any) {
		tCtx.Helper()
		actualManagedFields, _, _ := unstructured.NestedFieldCopy(obj.Object, "metadata", "managedFields")
		// Strip non-deterministic time.
		if actualManagedFields != nil {
			managers := actualManagedFields.([]any)
			for i := range managers {
				unstructured.RemoveNestedField(managers[i].(map[string]any), "time")
			}
		}
		require.Equal(tCtx, dump(expectedManagedFields), dump(actualManagedFields), fmt.Sprintf("%s:\n%s", what, dump(obj)))
	}

	requireInfos := func(what string, obj *unstructured.Unstructured, expectedInfos []testapigroupv1.CarpInfo) {
		tCtx.Helper()
		actualInfos, _, _ := unstructured.NestedFieldCopy(obj.Object, "status", "infos")
		require.Equal(tCtx, dump(expectedInfos), dump(actualInfos), fmt.Sprintf("%s:\n%s", what, dump(obj)))
	}

	carp := &unstructured.Unstructured{}
	name := "test-carp"
	namespace := createTestNamespace(tCtx, nil)
	carp.SetName(name)
	carp.SetNamespace(namespace)
	client := tCtx.Dynamic().Resource(testapigroupv1.SchemeGroupVersion.WithResource("carps")).Namespace(namespace)

	// Create with no fields in spec -> managed fields still empty.
	carp, err := client.Create(tCtx, carp, metav1.CreateOptions{FieldManager: "creator"})
	tCtx.ExpectNoError(err, "create carp")
	requireManagedFields("after creation", carp, nil)

	// Set infos with "A: 1, B: x" and "A: 2, B: x".
	carp, err = client.ApplyStatus(tCtx, name, parseObj(tCtx, `
kind: Carp
apiVersion: testapigroup.apimachinery.k8s.io/v1
status:
  infos:
  - a: 1
    b: "x"
    data: status1_a1_bx
  - a: 2
    b: "x"
    data: status1_a2_bx
`),
		metav1.ApplyOptions{FieldManager: "status1"})
	tCtx.ExpectNoError(err, "add status #1")
	requireManagedFields("add status #1", carp, parseAny(tCtx, `
- apiVersion: testapigroup.apimachinery.k8s.io/v1
  fieldsType: FieldsV1
  fieldsV1:
    f:status:
      f:infos:
        k:{"a":1,"b":"x"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
        k:{"a":2,"b":"x"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
  manager: status1
  operation: Apply
  subresource: status
`))
	requireInfos("add status #1", carp, []testapigroupv1.CarpInfo{{A: 1, B: "x", Data: "status1_a1_bx"}, {A: 2, B: "x", Data: "status1_a2_bx"}})

	// Second status infos with "A: 1, B: y" and "A: 2, B: y".
	carp, err = client.ApplyStatus(tCtx, name, parseObj(tCtx, `
kind: Carp
apiVersion: testapigroup.apimachinery.k8s.io/v1
status:
  infos:
  - a: 1
    b: "y"
    data: status2_a1_by
  - a: 2
    b: "y"
    data: status2_a2_by
`),
		metav1.ApplyOptions{FieldManager: "status2"})
	tCtx.ExpectNoError(err, "add status #2")
	requireManagedFields("add status #2", carp, parseAny(tCtx, `
- apiVersion: testapigroup.apimachinery.k8s.io/v1
  fieldsType: FieldsV1
  fieldsV1:
    f:status:
      f:infos:
        k:{"a":1,"b":"x"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
        k:{"a":2,"b":"x"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
  manager: status1
  operation: Apply
  subresource: status
- apiVersion: testapigroup.apimachinery.k8s.io/v1
  fieldsType: FieldsV1
  fieldsV1:
    f:status:
      f:infos:
        k:{"a":1,"b":"y"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
        k:{"a":2,"b":"y"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
  manager: status2
  operation: Apply
  subresource: status
`))
	requireInfos("add status #2", carp, []testapigroupv1.CarpInfo{{A: 1, B: "x", Data: "status1_a1_bx"}, {A: 2, B: "x", Data: "status1_a2_bx"}, {A: 1, B: "y", Data: "status2_a1_by"}, {A: 2, B: "y", Data: "status2_a2_by"}})

	// Remove one entry of first field manager.
	carp, err = client.ApplyStatus(tCtx, name, parseObj(tCtx, `
kind: Carp
apiVersion: testapigroup.apimachinery.k8s.io/v1
status:
  infos:
  - a: 1
    b: "x"
    data: status1_a1_bx
`),
		metav1.ApplyOptions{FieldManager: "status1"})
	tCtx.ExpectNoError(err, "remove status #1")
	requireManagedFields("remove status #1", carp, parseAny(tCtx, `
- apiVersion: testapigroup.apimachinery.k8s.io/v1
  fieldsType: FieldsV1
  fieldsV1:
    f:status:
      f:infos:
        k:{"a":1,"b":"x"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
  manager: status1
  operation: Apply
  subresource: status
- apiVersion: testapigroup.apimachinery.k8s.io/v1
  fieldsType: FieldsV1
  fieldsV1:
    f:status:
      f:infos:
        k:{"a":1,"b":"y"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
        k:{"a":2,"b":"y"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
  manager: status2
  operation: Apply
  subresource: status
`))
	requireInfos("remove status #1", carp, []testapigroupv1.CarpInfo{{A: 1, B: "x", Data: "status1_a1_bx"}, {A: 1, B: "y", Data: "status2_a1_by"}, {A: 2, B: "y", Data: "status2_a2_by"}})

	// Update one entry of second field manager.
	carp, err = client.ApplyStatus(tCtx, name, parseObj(tCtx, `
kind: Carp
apiVersion: testapigroup.apimachinery.k8s.io/v1
status:
  infos:
  - a: 1
    b: "y"
    data: status2_a1_by
  - a: 2
    b: "y"
    data: status2_a2_by_updated
`),
		metav1.ApplyOptions{FieldManager: "status2"})
	tCtx.ExpectNoError(err, "update status #2")
	requireManagedFields("update status #2", carp, parseAny(tCtx, `
- apiVersion: testapigroup.apimachinery.k8s.io/v1
  fieldsType: FieldsV1
  fieldsV1:
    f:status:
      f:infos:
        k:{"a":1,"b":"x"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
  manager: status1
  operation: Apply
  subresource: status
- apiVersion: testapigroup.apimachinery.k8s.io/v1
  fieldsType: FieldsV1
  fieldsV1:
    f:status:
      f:infos:
        k:{"a":1,"b":"y"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
        k:{"a":2,"b":"y"}:
          .: {}
          f:a: {}
          f:b: {}
          f:data: {}
  manager: status2
  operation: Apply
  subresource: status
`))
	requireInfos("update status #2", carp, []testapigroupv1.CarpInfo{{A: 1, B: "x", Data: "status1_a1_bx"}, {A: 1, B: "y", Data: "status2_a1_by"}, {A: 2, B: "y", Data: "status2_a2_by_updated"}})
}

func nestedFieldNoCopy(obj *unstructured.Unstructured, fields ...string) fieldLookupResult {
	field, found, err := unstructured.NestedFieldNoCopy(obj.Object, fields...)
	return fieldLookupResult{
		field: field,
		found: found,
		err:   err,
	}
}

type fieldLookupResult struct {
	field any
	found bool
	err   error
}

// createTestNamespace creates a namespace with a name that is derived from the
// current test name:
// - Non-alpha-numeric characters replaced by hyphen.
// - Truncated in the middle to make it short enough for GenerateName.
// - Hyphen plus random suffix added by the apiserver.
func createTestNamespace(tCtx ktesting.TContext, labels map[string]string) string {
	tCtx.Helper()
	name := regexp.MustCompile(`[^[:alnum:]_-]`).ReplaceAllString(tCtx.Name(), "-")
	name = strings.ToLower(name)
	if len(name) > 63 {
		name = name[:30] + "--" + name[len(name)-30:]
	}
	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{GenerateName: name + "-"}}
	ns.Labels = labels
	ns, err := tCtx.Client().CoreV1().Namespaces().Create(tCtx, ns, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create test namespace")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(tCtx.Client().CoreV1().Namespaces().Delete(tCtx, ns.Name, metav1.DeleteOptions{}), "delete test namespace")
	})
	return ns.Name
}

func dump(in any) string {
	out, err := yaml.Marshal(in)
	if err != nil {
		return err.Error()
	}
	return string(out)
}

func parseObj(tCtx ktesting.TContext, data string) *unstructured.Unstructured {
	tCtx.Helper()

	var obj unstructured.Unstructured
	err := yaml.Unmarshal([]byte(data), &obj)
	tCtx.ExpectNoError(err, data)
	return &obj
}

func parseAny(tCtx ktesting.TContext, data string) any {
	tCtx.Helper()

	var result any
	err := yaml.Unmarshal([]byte(data), &result)
	tCtx.ExpectNoError(err, data)
	return result
}
