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

package featuregates

import (
	"context"
	"fmt"
	"path"
	"strings"
	"testing"

	"github.com/google/uuid"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestFeatureRollbackDoesNotChoke validates that an object created with a new
// API field (associated with an experimental feature gate) does not cause the
// APIServer to crash or return decode errors when that feature gate is subsequently disabled.
func TestFeatureRollbackDoesNotChoke(t *testing.T) {
	testCases := []struct {
		name        string
		featureGate string
		gvr         schema.GroupVersionResource
		object      runtime.Object
		assertFn    func(t *testing.T, obj runtime.Object, featureEnabled bool)
	}{
		{
			name:        "EnvFiles field drop",
			featureGate: string(features.EnvFiles),
			gvr:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
			object: &corev1.Pod{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-envfiles",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{Name: "my-volume", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
					},
					Containers: []corev1.Container{
						{
							Name:  "nginx",
							Image: "nginx",
							Env: []corev1.EnvVar{
								{
									Name: "TEST_ENV",
									ValueFrom: &corev1.EnvVarSource{
										FileKeyRef: &corev1.FileKeySelector{
											VolumeName: "my-volume",
											Path:       "my-path",
											Key:        "my-key",
										},
									},
								},
							},
						},
					},
				},
			},
			assertFn: func(t *testing.T, obj runtime.Object, featureEnabled bool) {
				pod := obj.(*corev1.Pod)
				hasField := false
				if len(pod.Spec.Containers) > 0 && len(pod.Spec.Containers[0].Env) > 0 && pod.Spec.Containers[0].Env[0].ValueFrom != nil {
					hasField = pod.Spec.Containers[0].Env[0].ValueFrom.FileKeyRef != nil
				}
				if featureEnabled && !hasField {
					t.Errorf("Expected FileKeyRef to be present when EnvFiles is enabled, but it was missing")
				}
				if !featureEnabled && !hasField {
					t.Errorf("Expected FileKeyRef to be PRESERVED when EnvFiles is disabled, but it was dropped")
				}
			},
		},
	}

	// start etcd instance
	etcdOptions := framework.SharedEtcd()

	// Collect all feature gates
	enabledArgs := []string{"--disable-admission-plugins=ServiceAccount", "--service-cluster-ip-range=10.0.0.0/24"}
	disabledArgs := []string{"--disable-admission-plugins=ServiceAccount", "--service-cluster-ip-range=10.0.0.0/24"}

	for _, tc := range testCases {
		enabledArgs = append(enabledArgs, fmt.Sprintf("--feature-gates=%s=true", tc.featureGate))
		disabledArgs = append(disabledArgs, fmt.Sprintf("--feature-gates=%s=false", tc.featureGate))
	}

	etcdPrefix := path.Join("/", uuid.New().String(), "registry")
	enabledArgs = append(enabledArgs, fmt.Sprintf("--etcd-servers=%s", strings.Join(etcdOptions.Transport.ServerList, ",")), fmt.Sprintf("--etcd-prefix=%s", etcdPrefix))
	disabledArgs = append(disabledArgs, fmt.Sprintf("--etcd-servers=%s", strings.Join(etcdOptions.Transport.ServerList, ",")), fmt.Sprintf("--etcd-prefix=%s", etcdPrefix))

	// 1. Start APIServer with all the features ENABLED
	client1, config1, tearDown1 := framework.StartTestServerProcess(context.TODO(), t, framework.TestServerSetup{
		Flags: enabledArgs,
	})

	// Verify the features are enabled via metrics
	for _, tc := range testCases {
		if err := verifyFeatureEnabledMetric(client1, tc.featureGate, true); err != nil {
			t.Errorf("Failed to verify feature %s enabled on Server 1: %v", tc.featureGate, err)
		}
	}

	dynClient1, err := dynamic.NewForConfig(config1)
	if err != nil {
		t.Fatalf("Unexpected error creating dynamic client: %v", err)
	}

	createdObjs := make(map[string]*unstructured.Unstructured)

	for _, tc := range testCases {
		objMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(tc.object)
		if err != nil {
			t.Fatalf("[%s] Failed to convert object to unstructured: %v", tc.name, err)
		}
		obj := &unstructured.Unstructured{Object: objMap}
		// ensure apiVersion and kind are set for unstructured creation
		obj.SetGroupVersionKind(tc.gvr.GroupVersion().WithKind(tc.object.GetObjectKind().GroupVersionKind().Kind))
		if obj.GetKind() == "" {
			// fallback if kind isn't populated
			// Note: This relies on the GVR resource being somewhat mappable, but better to set it properly in the test case if needed
			// Let's rely on standard mapping or let dynClient handle it.
		}

		_ = dynClient1.Resource(tc.gvr).Namespace(obj.GetNamespace()).Delete(context.TODO(), obj.GetName(), metav1.DeleteOptions{})

		createdObj, err := dynClient1.Resource(tc.gvr).Namespace(obj.GetNamespace()).Create(context.TODO(), obj, metav1.CreateOptions{})
		if err != nil {
			tearDown1()
			t.Fatalf("[%s] Failed to create object with feature enabled: %v", tc.name, err)
		}
		t.Logf("[%s] Successfully created %s '%s' with feature gate %s=true", tc.name, tc.gvr.Resource, createdObj.GetName(), tc.featureGate)

		if tc.assertFn != nil {
			typedObj := tc.object.DeepCopyObject()
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(createdObj.Object, typedObj); err != nil {
				t.Fatalf("[%s] Failed to convert created object to typed: %v", tc.name, err)
			}
			tc.assertFn(t, typedObj, true)
		}

		createdObjs[tc.name] = createdObj
	}

	// Shutdown Server 1
	tearDown1()

	// 2. Start APIServer with all the features DISABLED
	client2, config2, tearDown2 := framework.StartTestServerProcess(context.TODO(), t, framework.TestServerSetup{
		Flags: disabledArgs,
	})
	defer tearDown2()

	// Verify the features are disabled via metrics
	for _, tc := range testCases {
		if err := verifyFeatureEnabledMetric(client2, tc.featureGate, false); err != nil {
			t.Errorf("Failed to verify feature %s disabled on Server 2: %v", tc.featureGate, err)
		}
	}

	dynClient2, err := dynamic.NewForConfig(config2)
	if err != nil {
		t.Fatalf("Unexpected error creating dynamic client 2: %v", err)
	}

	for _, tc := range testCases {
		createdObj := createdObjs[tc.name]

		readObj, err := dynClient2.Resource(tc.gvr).Namespace(createdObj.GetNamespace()).Get(context.TODO(), createdObj.GetName(), metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get object with feature disabled: %v", err)
		}
		t.Logf("Successfully read %s '%s' with feature gate %s=false", tc.gvr.Resource, readObj.GetName(), tc.featureGate)

		if tc.assertFn != nil {
			typedObj := tc.object.DeepCopyObject()
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(readObj.Object, typedObj); err != nil {
				t.Fatalf("[%s] Failed to convert read object to typed: %v", tc.name, err)
			}
			tc.assertFn(t, typedObj, false)
		}

		annotations := readObj.GetAnnotations()
		if annotations == nil {
			annotations = make(map[string]string)
		}
		annotations["test-update"] = "true"
		readObj.SetAnnotations(annotations)

		_, err = dynClient2.Resource(tc.gvr).Namespace(createdObj.GetNamespace()).Update(context.TODO(), readObj, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Failed to update object with feature disabled: %v", err)
		}
		t.Logf("Successfully updated %s '%s' with feature gate %s=false", tc.gvr.Resource, readObj.GetName(), tc.featureGate)

		err = dynClient2.Resource(tc.gvr).Namespace(createdObj.GetNamespace()).Delete(context.TODO(), readObj.GetName(), metav1.DeleteOptions{})
		if err != nil {
			t.Fatalf("Failed to delete object with feature disabled: %v", err)
		}
		t.Logf("Successfully deleted %s '%s' with feature gate %s=false", tc.gvr.Resource, readObj.GetName(), tc.featureGate)
	}
}

func verifyFeatureEnabledMetric(client kubernetes.Interface, featureGate string, expectedEnabled bool) error {
	res := client.Discovery().RESTClient().Get().AbsPath("/metrics").Do(context.TODO())
	raw, err := res.Raw()
	if err != nil {
		return fmt.Errorf("failed to fetch metrics: %w", err)
	}
	
	expectedVal := "0"
	if expectedEnabled {
		expectedVal = "1"
	}

	prefix := fmt.Sprintf("kubernetes_feature_enabled{name=\"%s\"", featureGate)
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.HasPrefix(line, prefix) {
			if strings.HasSuffix(strings.TrimSpace(line), "} "+expectedVal) {
				return nil
			}
			return fmt.Errorf("found metric line %q, but expected value %s", line, expectedVal)
		}
	}

	return fmt.Errorf("expected metric prefix %s not found in /metrics endpoint", prefix)
}

