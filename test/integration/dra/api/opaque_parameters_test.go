/*
Copyright The Kubernetes Authors.

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

package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path"
	"strings"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestOpaqueParametersMetadataUpdates(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "opaque-parameters", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	restClient := client.ResourceV1().RESTClient()

	for _, kind := range []string{"ResourceClaim", "ResourceClaimTemplate"} {
		t.Run(kind, func(t *testing.T) {
			resourceName := strings.ToLower(kind) + "s"
			specPath := "/spec/devices"
			if kind == "ResourceClaimTemplate" {
				specPath = "/spec/spec/devices"
			}
			for jsonIndex, parameters := range []string{
				`{"z": 1, "a": "b"}`,
				`{"z":1,"a":"b"}`,
				`{"a":"b","z":1}`,
			} {
				for _, method := range []string{"merge", "strategic", "json", "apply", "replace"} {
					t.Run(fmt.Sprintf("%d/%s", jsonIndex, method), func(t *testing.T) {
						ctx := t.Context()
						name := fmt.Sprintf("claim-%d-%s", jsonIndex, method)
						spec := fmt.Sprintf(`{"devices":{"requests":[{"name":"r","exactly":{"deviceClassName":"cls"}}],"config":[{"requests":["r"],"opaque":{"driver":"driver.example.com","parameters":%s}}]}}`, parameters)
						if kind == "ResourceClaimTemplate" {
							spec = `{"spec":` + spec + `}`
						}
						body := fmt.Sprintf(`{"apiVersion":"resource.k8s.io/v1","kind":%q,"metadata":{"name":%q},"spec":%s}`, kind, name, spec)
						// Send raw JSON because marshaling a typed object would compact parameters
						// before they reach storage and conceal the regression.
						if err := restClient.Post().Namespace(ns.Name).Resource(resourceName).Param("fieldManager", "opaque-test").Body([]byte(body)).Do(ctx).Error(); err != nil {
							t.Fatalf("create: %v", err)
						}
						assertStored := func() {
							t.Helper()
							key := path.Join("/", server.EtcdStoragePrefix, resourceName, ns.Name, name)
							response, err := server.EtcdClient.Get(ctx, key)
							if err != nil {
								t.Fatalf("read storage: %v", err)
							}
							if len(response.Kvs) != 1 {
								t.Fatalf("expected one stored object at %s, got %d", key, len(response.Kvs))
							}
							stored, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), response.Kvs[0].Value)
							if err != nil {
								t.Fatalf("decode storage: %v", err)
							}
							var raw []byte
							switch stored := stored.(type) {
							case *resource.ResourceClaim:
								raw = stored.Spec.Devices.Config[0].Opaque.Parameters.Raw
							case *resource.ResourceClaimTemplate:
								raw = stored.Spec.Spec.Devices.Config[0].Opaque.Parameters.Raw
							default:
								t.Fatalf("unexpected stored type: %T", stored)
							}
							if !bytes.Equal(raw, []byte(parameters)) {
								t.Fatalf("stored parameters = %s, want original %s", raw, parameters)
							}
						}
						assertStored()

						update := func(pointer string, value any) error {
							t.Helper()
							var document map[string]any
							if method == "replace" || method == "apply" {
								raw, err := restClient.Get().Namespace(ns.Name).Resource(resourceName).Name(name).SetHeader("Accept", "application/json").Do(ctx).Raw()
								if err != nil {
									return err
								}
								if err := json.Unmarshal(raw, &document); err != nil {
									return err
								}
								if method == "apply" {
									// Preserve desired fields across successive applies; omission
									// would remove fields owned by this apply manager.
									metadata := document["metadata"].(map[string]any)
									document["metadata"] = map[string]any{
										"name":        name,
										"labels":      metadata["labels"],
										"annotations": metadata["annotations"],
										"finalizers":  metadata["finalizers"],
									}
									delete(document, "status")
								}
							} else {
								document = map[string]any{"apiVersion": "resource.k8s.io/v1", "kind": kind, "metadata": map[string]any{"name": name}}
							}
							fields := strings.Split(strings.TrimPrefix(pointer, "/"), "/")
							current := document
							for _, field := range fields[:len(fields)-1] {
								child, ok := current[field].(map[string]any)
								if !ok {
									child = map[string]any{}
									current[field] = child
								}
								current = child
							}
							current[fields[len(fields)-1]] = value
							var payload any = document
							patchType := types.MergePatchType
							switch method {
							case "strategic":
								patchType = types.StrategicMergePatchType
							case "json":
								patchType = types.JSONPatchType
								payload = []map[string]any{{"op": "add", "path": pointer, "value": value}}
							case "apply":
								patchType = types.ApplyPatchType
							}
							data, err := json.Marshal(payload)
							if err != nil {
								return err
							}
							var request *rest.Request
							if method == "replace" {
								request = restClient.Put()
							} else {
								request = restClient.Patch(patchType)
							}
							request = request.Namespace(ns.Name).Resource(resourceName).Name(name).Body(data)
							if method == "apply" {
								request = request.Param("fieldManager", "opaque-test").Param("force", "true")
							}
							return request.Do(ctx).Error()
						}
						// Let each update method own the finalizer it will later remove.
						if err := update("/metadata/finalizers", []string{"example.com/test"}); err != nil {
							t.Fatalf("add finalizer: %v", err)
						}
						assertStored()
						for _, field := range []string{"labels", "annotations"} {
							if err := update("/metadata/"+field, map[string]any{"test": "updated"}); err != nil {
								t.Fatalf("update %s: %v", field, err)
							}
							assertStored()
						}
						exactly := map[string]any{"deviceClassName": "changed"}
						opaque := map[string]any{"driver": "driver.example.com", "parameters": map[string]any{"a": "b", "z": 1}}
						changedDevices := map[string]any{
							"requests": []any{map[string]any{"name": "r", "exactly": exactly}},
							"config":   []any{map[string]any{"requests": []string{"r"}, "opaque": opaque}},
						}
						if err := update(specPath, changedDevices); !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "field is immutable") {
							t.Fatalf("expected immutable spec error for changed device class, got %v", err)
						}
						exactly["deviceClassName"] = "cls"
						opaque["parameters"] = map[string]any{"a": "changed", "z": 1}
						if err := update(specPath, changedDevices); !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "field is immutable") {
							t.Fatalf("expected immutable spec error for changed parameters, got %v", err)
						}
						assertStored()
						if err := restClient.Delete().Namespace(ns.Name).Resource(resourceName).Name(name).Body(&metav1.DeleteOptions{}).Do(ctx).Error(); err != nil {
							t.Fatalf("delete: %v", err)
						}
						assertStored()
						var finalizers any
						if method == "apply" {
							finalizers = []string{}
						}
						if err := update("/metadata/finalizers", finalizers); err != nil {
							t.Fatalf("remove finalizer from deleting object: %v", err)
						}
						if err := restClient.Get().Namespace(ns.Name).Resource(resourceName).Name(name).Do(ctx).Error(); !apierrors.IsNotFound(err) {
							raw, _ := restClient.Get().Namespace(ns.Name).Resource(resourceName).Name(name).SetHeader("Accept", "application/json").Do(ctx).Raw()
							t.Fatalf("expected object to be deleted after finalizer removal, got %v: %s", err, raw)
						}
					})
				}
			}
		})
	}
}
