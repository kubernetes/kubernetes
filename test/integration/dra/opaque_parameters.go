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

package dra

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/ktesting"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
)

func testOpaqueParametersMetadataUpdates(tCtx ktesting.TContext) {
	tCtx.Parallel()
	server := tCtx.Value(apiServerKey).(*kubeapiservertesting.TestServer)
	namespace := createTestNamespace(tCtx, nil)
	restClient := tCtx.Client().ResourceV1().RESTClient()

	tCtx.Run("Compatibility", func(tCtx ktesting.TContext) {
		testOpaqueParametersCompatibility(tCtx, server, restClient, namespace)
	})

	for _, kind := range []string{"ResourceClaim", "ResourceClaimTemplate"} {
		tCtx.Run(kind, func(tCtx ktesting.TContext) {
			resourceName := strings.ToLower(kind) + "s"
			specPath := "/spec/devices"
			if kind == "ResourceClaimTemplate" {
				specPath = "/spec/spec/devices"
			}
			for jsonIndex, tc := range []struct {
				parameters   string
				applyInvalid bool
			}{
				{parameters: `{"z": 1, "a": "b"}`},
				{parameters: `{"z":1,"a":"b"}`},
				{parameters: `{"a":"b","z":1}`},
				{parameters: `{"z": 1.0, "a": "b"}`, applyInvalid: true},
				{parameters: `{"z": 9007199254740993, "a": "b"}`},
			} {
				parameters := tc.parameters
				for _, method := range []string{"merge", "strategic", "json", "apply", "replace"} {
					tCtx.Run(fmt.Sprintf("%d/%s", jsonIndex, method), func(tCtx ktesting.TContext) {
						name := fmt.Sprintf("claim-%d-%s", jsonIndex, method)
						spec := fmt.Sprintf(`{"devices":{"requests":[{"name":"r","exactly":{"deviceClassName":"cls"}}],"config":[{"requests":["r"],"opaque":{"driver":"driver.example.com","parameters":%s}}]}}`, parameters)
						if kind == "ResourceClaimTemplate" {
							spec = `{"spec":` + spec + `}`
						}
						body := fmt.Sprintf(`{"apiVersion":"resource.k8s.io/v1","kind":%q,"metadata":{"name":%q},"spec":%s}`, kind, name, spec)
						// Send raw JSON because marshaling a typed object would compact parameters
						// before they reach storage and conceal the regression.
						if err := restClient.Post().Namespace(namespace).Resource(resourceName).Param("fieldManager", "opaque-test").Body([]byte(body)).Do(tCtx).Error(); err != nil {
							tCtx.Fatalf("create: %v", err)
						}
						assertStored := func() {
							tCtx.Helper()
							key := path.Join("/", server.EtcdStoragePrefix, resourceName, namespace, name)
							response, err := server.EtcdClient.Get(tCtx, key)
							if err != nil {
								tCtx.Fatalf("read storage: %v", err)
							}
							if len(response.Kvs) != 1 {
								tCtx.Fatalf("expected one stored object at %s, got %d", key, len(response.Kvs))
							}
							stored, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), response.Kvs[0].Value)
							if err != nil {
								tCtx.Fatalf("decode storage: %v", err)
							}
							var raw []byte
							switch stored := stored.(type) {
							case *resource.ResourceClaim:
								raw = stored.Spec.Devices.Config[0].Opaque.Parameters.Raw
							case *resource.ResourceClaimTemplate:
								raw = stored.Spec.Spec.Devices.Config[0].Opaque.Parameters.Raw
							default:
								tCtx.Fatalf("unexpected stored type: %T", stored)
							}
							if !bytes.Equal(raw, []byte(parameters)) {
								tCtx.Fatalf("stored parameters = %s, want original %s", raw, parameters)
							}
						}
						assertStored()

						update := func(pointer string, value any) error {
							tCtx.Helper()
							var document map[string]any
							if method == "replace" || method == "apply" {
								raw, err := restClient.Get().Namespace(namespace).Resource(resourceName).Name(name).SetHeader("Accept", "application/json").Do(tCtx).Raw()
								if err != nil {
									return err
								}
								// Preserve number tokens so the test client does not change
								// decimal spellings or round large integers before the update.
								decoder := json.NewDecoder(bytes.NewReader(raw))
								decoder.UseNumber()
								if err := decoder.Decode(&document); err != nil {
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
							request = request.Namespace(namespace).Resource(resourceName).Name(name).Body(data)
							if method == "apply" {
								request = request.Param("fieldManager", "opaque-test").Param("force", "true")
							}
							return request.Do(tCtx).Error()
						}
						// Let each update method own the finalizer it will later remove.
						err := update("/metadata/finalizers", []string{"example.com/test"})
						if method == "apply" && tc.applyInvalid {
							// SSA converts 1.0 to 1. The helper deliberately preserves
							// number spellings, so this update must remain rejected.
							if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "field is immutable") {
								tCtx.Fatalf("expected immutable spec error for SSA number conversion, got %v", err)
							}
							assertStored()
							if err := restClient.Delete().Namespace(namespace).Resource(resourceName).Name(name).Body(&metav1.DeleteOptions{}).Do(tCtx).Error(); err != nil {
								tCtx.Fatalf("delete after rejected apply: %v", err)
							}
							return
						}
						if err != nil {
							tCtx.Fatalf("add finalizer: %v", err)
						}
						assertStored()
						for _, field := range []string{"labels", "annotations"} {
							if err := update("/metadata/"+field, map[string]any{"test": "updated"}); err != nil {
								tCtx.Fatalf("update %s: %v", field, err)
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
							tCtx.Fatalf("expected immutable spec error for changed device class, got %v", err)
						}
						exactly["deviceClassName"] = "cls"
						opaque["parameters"] = map[string]any{"a": "changed", "z": 1}
						if err := update(specPath, changedDevices); !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "field is immutable") {
							tCtx.Fatalf("expected immutable spec error for changed parameters, got %v", err)
						}
						assertStored()
						if err := restClient.Delete().Namespace(namespace).Resource(resourceName).Name(name).Body(&metav1.DeleteOptions{}).Do(tCtx).Error(); err != nil {
							tCtx.Fatalf("delete: %v", err)
						}
						assertStored()
						var finalizers any
						if method == "apply" {
							finalizers = []string{}
						}
						if err := update("/metadata/finalizers", finalizers); err != nil {
							tCtx.Fatalf("remove finalizer from deleting object: %v", err)
						}
						if err := restClient.Get().Namespace(namespace).Resource(resourceName).Name(name).Do(tCtx).Error(); !apierrors.IsNotFound(err) {
							raw, _ := restClient.Get().Namespace(namespace).Resource(resourceName).Name(name).SetHeader("Accept", "application/json").Do(tCtx).Raw()
							tCtx.Fatalf("expected object to be deleted after finalizer removal, got %v: %s", err, raw)
						}
					})
				}
			}
		})
	}
}

func testOpaqueParametersCompatibility(tCtx ktesting.TContext, server *kubeapiservertesting.TestServer, restClient rest.Interface, namespace string) {
	for _, kind := range []string{"ResourceClaim", "ResourceClaimTemplate"} {
		for _, mode := range []string{"Ignore", "Warn", "Strict"} {
			for _, tc := range []struct {
				name, original, normalized, changed string
			}{
				{"duplicate", `{"b":3,"a":1,"a":2}`, `{"a": 1, "a": 2, "b": 3}`, `{"a":4,"a":1,"b":3}`},
				{"invalid-utf8", "{\"a\":\"\xff\"}", `{"a":"�"}`, `{"a":"changed"}`},
			} {
				tCtx.Run(kind+"/"+mode+"/"+tc.name, func(tCtx ktesting.TContext) {
					name := strings.ToLower(kind) + "-" + strings.ToLower(mode) + "-" + tc.name
					resourceName := strings.ToLower(kind) + "s"
					body := func(parameters, resourceVersion string) []byte {
						spec := fmt.Sprintf(`{"devices":{"requests":[{"name":"r","exactly":{"deviceClassName":"cls"}}],"config":[{"opaque":{"driver":"driver.example.com","parameters":%s}}]}}`, parameters)
						if kind == "ResourceClaimTemplate" {
							spec = `{"spec":` + spec + `}`
						}
						label := "original"
						if resourceVersion != "" {
							label = "updated"
						}
						return []byte(fmt.Sprintf(`{"apiVersion":"resource.k8s.io/v1","kind":%q,"metadata":{"name":%q,"resourceVersion":%q,"labels":{"test":%q}},"spec":%s}`, kind, name, resourceVersion, label, spec))
					}
					// Raw requests preserve duplicate names and invalid UTF-8 that typed
					// or map-based clients may otherwise normalize before sending.
					if err := restClient.Post().Namespace(namespace).Resource(resourceName).Param("fieldValidation", mode).Body(body(tc.original, "")).Do(tCtx).Error(); err != nil {
						tCtx.Fatalf("create: %v", err)
					}
					assertStored := func() {
						tCtx.Helper()
						key := path.Join("/", server.EtcdStoragePrefix, resourceName, namespace, name)
						response, err := server.EtcdClient.Get(tCtx, key)
						if err != nil {
							tCtx.Fatalf("read storage: %v", err)
						}
						if len(response.Kvs) != 1 {
							tCtx.Fatalf("expected one stored object at %s, got %d", key, len(response.Kvs))
						}
						stored, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), response.Kvs[0].Value)
						if err != nil {
							tCtx.Fatalf("decode storage: %v", err)
						}
						var raw []byte
						switch stored := stored.(type) {
						case *resource.ResourceClaim:
							raw = stored.Spec.Devices.Config[0].Opaque.Parameters.Raw
						case *resource.ResourceClaimTemplate:
							raw = stored.Spec.Spec.Devices.Config[0].Opaque.Parameters.Raw
						default:
							tCtx.Fatalf("unexpected stored type: %T", stored)
						}
						if !bytes.Equal(raw, []byte(tc.original)) {
							tCtx.Fatalf("stored parameters = %q, want original %q", raw, tc.original)
						}
					}
					assertStored()
					for _, parameters := range []string{tc.normalized, tc.changed} {
						data, err := restClient.Get().Namespace(namespace).Resource(resourceName).Name(name).SetHeader("Accept", "application/json").Do(tCtx).Raw()
						if err != nil {
							tCtx.Fatalf("get: %v", err)
						}
						var metadata struct {
							Metadata metav1.ObjectMeta `json:"metadata"`
						}
						if err := json.Unmarshal(data, &metadata); err != nil {
							tCtx.Fatalf("decode metadata: %v", err)
						}
						err = restClient.Put().Namespace(namespace).Resource(resourceName).Name(name).Param("fieldValidation", mode).Body(body(parameters, metadata.Metadata.ResourceVersion)).Do(tCtx).Error()
						if parameters == tc.normalized {
							if err != nil {
								tCtx.Fatalf("update normalized parameters: %v", err)
							}
						} else if !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "field is immutable") {
							tCtx.Fatalf("expected immutable spec error for changed parameters, got %v", err)
						}
						assertStored()
					}
					if err := restClient.Delete().Namespace(namespace).Resource(resourceName).Name(name).Body(&metav1.DeleteOptions{}).Do(tCtx).Error(); err != nil {
						tCtx.Fatalf("delete: %v", err)
					}
				})
			}
		}
	}
}
