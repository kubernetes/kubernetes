/*
Copyright 2019 The Kubernetes Authors.

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

package fieldmanager_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/kube-openapi/pkg/util/proto"
	prototesting "k8s.io/kube-openapi/pkg/util/proto/testing"
	"sigs.k8s.io/yaml"
)

var fakeSchema = prototesting.Fake{
	Path: filepath.Join(
		strings.Repeat(".."+string(filepath.Separator), 7),
		"api", "openapi-spec", "swagger.json"),
}

type fakeObjectConvertor struct{}

func (c *fakeObjectConvertor) Convert(in, out, context interface{}) error {
	out = in
	return nil
}

func (c *fakeObjectConvertor) ConvertToVersion(in runtime.Object, _ runtime.GroupVersioner) (runtime.Object, error) {
	return in, nil
}

func (c *fakeObjectConvertor) ConvertFieldLabel(_ schema.GroupVersionKind, _, _ string) (string, string, error) {
	return "", "", errors.New("not implemented")
}

type fakeObjectDefaulter struct{}

func (d *fakeObjectDefaulter) Default(in runtime.Object) {}

type TestFieldManager struct {
	fieldManager fieldmanager.FieldManager
	emptyObj     runtime.Object
	liveObj      runtime.Object
}

func NewTestFieldManager(gvk schema.GroupVersionKind) TestFieldManager {
	d, err := fakeSchema.OpenAPISchema()
	if err != nil {
		panic(err)
	}
	m, err := proto.NewOpenAPIData(d)
	if err != nil {
		panic(err)
	}

	f, err := fieldmanager.NewFieldManager(
		m,
		&fakeObjectConvertor{},
		&fakeObjectDefaulter{},
		gvk.GroupVersion(),
		gvk.GroupVersion(),
	)
	if err != nil {
		panic(err)
	}
	live := &unstructured.Unstructured{}
	live.SetKind(gvk.Kind)
	live.SetAPIVersion(gvk.GroupVersion().String())
	return TestFieldManager{
		fieldManager: f,
		emptyObj:     live,
		liveObj:      live.DeepCopyObject(),
	}
}

func (f *TestFieldManager) Reset() {
	f.liveObj = f.emptyObj.DeepCopyObject()
}

func (f *TestFieldManager) Apply(obj []byte, manager string, force bool) error {
	out, err := f.fieldManager.Apply(f.liveObj, obj, manager, force)
	if err == nil {
		f.liveObj = out
	}
	return err
}

func (f *TestFieldManager) Update(obj runtime.Object, manager string) error {
	out, err := f.fieldManager.Update(f.liveObj, obj, manager)
	if err == nil {
		f.liveObj = out
	}
	return err
}

func (f *TestFieldManager) ManagedFields() []metav1.ManagedFieldsEntry {
	accessor, err := meta.Accessor(f.liveObj)
	if err != nil {
		panic(fmt.Errorf("couldn't get accessor: %v", err))
	}

	return accessor.GetManagedFields()
}

// TestUpdateApplyConflict tests that applying to an object, which
// wasn't created by apply, will give conflicts
func TestUpdateApplyConflict(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	patch := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
                        "replicas": 3,
                        "selector": {
                                "matchLabels": {
                                         "app": "nginx"
                                }
                        },
                        "template": {
                                "metadata": {
                                        "labels": {
                                                "app": "nginx"
                                        }
                                },
                                "spec": {
				        "containers": [{
					        "name":  "nginx",
					        "image": "nginx:latest"
				        }]
                                }
                        }
		}
	}`)
	newObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal(patch, &newObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	if err := f.Update(newObj, "fieldmanager_test"); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	err := f.Apply([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
		},
		"spec": {
			"replicas": 101,
		}
	}`), "fieldmanager_conflict", false)
	if err == nil || !apierrors.IsConflict(err) {
		t.Fatalf("Expecting to get conflicts but got %v", err)
	}
}

func TestApplyStripsFields(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	newObj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
		},
	}

	newObj.SetName("b")
	newObj.SetNamespace("b")
	newObj.SetUID("b")
	newObj.SetClusterName("b")
	newObj.SetGeneration(0)
	newObj.SetResourceVersion("b")
	newObj.SetCreationTimestamp(metav1.NewTime(time.Now()))
	newObj.SetManagedFields([]metav1.ManagedFieldsEntry{
		{
			Manager:    "update",
			Operation:  metav1.ManagedFieldsOperationApply,
			APIVersion: "apps/v1",
		},
	})
	if err := f.Update(newObj, "fieldmanager_test"); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	if m := f.ManagedFields(); len(m) != 0 {
		t.Fatalf("fields did not get stripped: %v", m)
	}
}

func TestVersionCheck(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	// patch has 'apiVersion: apps/v1' and live version is apps/v1 -> no errors
	err := f.Apply([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
	}`), "fieldmanager_test", false)
	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	// patch has 'apiVersion: apps/v2' but live version is apps/v1 -> error
	err = f.Apply([]byte(`{
		"apiVersion": "apps/v1beta1",
		"kind": "Deployment",
	}`), "fieldmanager_test", false)
	if err == nil {
		t.Fatalf("expected an error from mismatched patch and live versions")
	}
	switch typ := err.(type) {
	default:
		t.Fatalf("expected error to be of type %T was %T", apierrors.StatusError{}, typ)
	case apierrors.APIStatus:
		if typ.Status().Code != http.StatusBadRequest {
			t.Fatalf("expected status code to be %d but was %d",
				http.StatusBadRequest, typ.Status().Code)
		}
	}
}

func TestApplyDoesNotStripLabels(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	err := f.Apply([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a": "b"
			},
		}
	}`), "fieldmanager_test", false)
	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	if m := f.ManagedFields(); len(m) != 1 {
		t.Fatalf("labels shouldn't get stripped on apply: %v", m)
	}
}

var podBytes = []byte(`
{
   "apiVersion": "v1",
   "kind": "Pod",
   "metadata": {
      "labels": {
         "app": "some-app",
         "plugin1": "some-value",
         "plugin2": "some-value",
         "plugin3": "some-value",
         "plugin4": "some-value"
      },
      "name": "some-name",
      "namespace": "default",
      "ownerReferences": [
         {
            "apiVersion": "apps/v1",
            "blockOwnerDeletion": true,
            "controller": true,
            "kind": "ReplicaSet",
            "name": "some-name",
            "uid": "0a9d2b9e-779e-11e7-b422-42010a8001be"
         }
      ]
   },
   "spec": {
      "containers": [
         {
            "args": [
               "one",
               "two",
               "three",
               "four",
               "five",
               "six",
               "seven",
               "eight",
               "nine"
            ],
            "env": [
               {
                  "name": "VAR_3",
                  "valueFrom": {
                     "secretKeyRef": {
                        "key": "some-other-key",
                        "name": "some-oher-name"
                     }
                  }
               },
               {
                  "name": "VAR_2",
                  "valueFrom": {
                     "secretKeyRef": {
                        "key": "other-key",
                        "name": "other-name"
                     }
                  }
               },
               {
                  "name": "VAR_1",
                  "valueFrom": {
                     "secretKeyRef": {
                        "key": "some-key",
                        "name": "some-name"
                     }
                  }
               }
            ],
            "image": "some-image-name",
            "imagePullPolicy": "IfNotPresent",
            "name": "some-name",
            "resources": {
               "requests": {
                  "cpu": "0"
               }
            },
            "terminationMessagePath": "/dev/termination-log",
            "terminationMessagePolicy": "File",
            "volumeMounts": [
               {
                  "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount",
                  "name": "default-token-hu5jz",
                  "readOnly": true
               }
            ]
         }
      ],
      "dnsPolicy": "ClusterFirst",
      "nodeName": "node-name",
      "priority": 0,
      "restartPolicy": "Always",
      "schedulerName": "default-scheduler",
      "securityContext": {},
      "serviceAccount": "default",
      "serviceAccountName": "default",
      "terminationGracePeriodSeconds": 30,
      "tolerations": [
         {
            "effect": "NoExecute",
            "key": "node.kubernetes.io/not-ready",
            "operator": "Exists",
            "tolerationSeconds": 300
         },
         {
            "effect": "NoExecute",
            "key": "node.kubernetes.io/unreachable",
            "operator": "Exists",
            "tolerationSeconds": 300
         }
      ],
      "volumes": [
         {
            "name": "default-token-hu5jz",
            "secret": {
               "defaultMode": 420,
               "secretName": "default-token-hu5jz"
            }
         }
      ]
   },
   "status": {
      "conditions": [
         {
            "lastProbeTime": null,
            "lastTransitionTime": "2019-07-08T09:31:18Z",
            "status": "True",
            "type": "Initialized"
         },
         {
            "lastProbeTime": null,
            "lastTransitionTime": "2019-07-08T09:41:59Z",
            "status": "True",
            "type": "Ready"
         },
         {
            "lastProbeTime": null,
            "lastTransitionTime": null,
            "status": "True",
            "type": "ContainersReady"
         },
         {
            "lastProbeTime": null,
            "lastTransitionTime": "2019-07-08T09:31:18Z",
            "status": "True",
            "type": "PodScheduled"
         }
      ],
      "containerStatuses": [
         {
            "containerID": "docker://885e82a1ed0b7356541bb410a0126921ac42439607c09875cd8097dd5d7b5376",
            "image": "some-image-name",
            "imageID": "docker-pullable://some-image-id",
            "lastState": {
               "terminated": {
                  "containerID": "docker://d57290f9e00fad626b20d2dd87a3cf69bbc22edae07985374f86a8b2b4e39565",
                  "exitCode": 255,
                  "finishedAt": "2019-07-08T09:39:09Z",
                  "reason": "Error",
                  "startedAt": "2019-07-08T09:38:54Z"
               }
            },
            "name": "name",
            "ready": true,
            "restartCount": 6,
            "state": {
               "running": {
                  "startedAt": "2019-07-08T09:41:59Z"
               }
            }
         }
      ],
      "hostIP": "10.0.0.1",
      "phase": "Running",
      "podIP": "10.0.0.1",
      "qosClass": "BestEffort",
      "startTime": "2019-07-08T09:31:18Z"
   }
}`)

func TestApplyNewObject(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	if err := f.Apply(podBytes, "fieldmanager_test", false); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkApplyNewObject(b *testing.B) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		err := f.Apply(podBytes, "fieldmanager_test", false)
		if err != nil {
			b.Fatal(err)
		}
		f.Reset()
	}
}

func BenchmarkUpdateNewObject(b *testing.B) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	newObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := json.Unmarshal(podBytes, &newObj.Object); err != nil {
		b.Fatalf("Failed to parse yaml object: %v", err)
	}
	newObj.SetManagedFields([]metav1.ManagedFieldsEntry{
		{
			Manager:    "default",
			Operation:  "Update",
			APIVersion: "v1",
		},
	})

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		err := f.Update(newObj, "fieldmanager_test")
		if err != nil {
			b.Fatal(err)
		}
		f.Reset()
	}
}

func BenchmarkRepeatedUpdate(b *testing.B) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	var obj *corev1.Pod
	if err := json.Unmarshal(podBytes, &obj); err != nil {
		b.Fatalf("Failed to parse yaml object: %v", err)
	}
	obj.Spec.Containers[0].Image = "nginx:latest"
	objs := []*corev1.Pod{obj}
	obj = obj.DeepCopy()
	obj.Spec.Containers[0].Image = "nginx:4.3"
	objs = append(objs, obj)

	err := f.Apply(podBytes, "fieldmanager_apply", false)
	if err != nil {
		b.Fatal(err)
	}

	if err := f.Update(objs[1], "fieldmanager_1"); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		err := f.Update(objs[n%len(objs)], fmt.Sprintf("fieldmanager_%d", n%len(objs)))
		if err != nil {
			b.Fatal(err)
		}
		f.Reset()
	}
}

func TestApplyFailsWithManagedFields(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	err := f.Apply([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"managedFields": [
				{
				  "manager": "test",
				}
			]
		}
	}`), "fieldmanager_test", false)

	if err == nil {
		t.Fatalf("successfully applied with set managed fields")
	}
}

func TestApplySuccessWithNoManagedFields(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	err := f.Apply([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a": "b"
			},
		}
	}`), "fieldmanager_test", false)

	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}
}
