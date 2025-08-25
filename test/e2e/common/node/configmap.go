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

package node

import (
	"context"
	"encoding/json"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("ConfigMap", func() {
	f := framework.NewDefaultFramework("configmap")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	/*
		Release: v1.9
		Testname: ConfigMap, from environment field
		Description: Create a Pod with an environment variable value set using a value from ConfigMap. A ConfigMap value MUST be accessible in the container environment.
	*/
	framework.ConformanceIt("should be consumable via environment variable", f.WithNodeConformance(), func(ctx context.Context) {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newConfigMap(f, name)
		ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, configMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "env-test",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env"},
						Env: []v1.EnvVar{
							{
								Name: "CONFIG_DATA_1",
								ValueFrom: &v1.EnvVarSource{
									ConfigMapKeyRef: &v1.ConfigMapKeySelector{
										LocalObjectReference: v1.LocalObjectReference{
											Name: name,
										},
										Key: "data-1",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		e2epodoutput.TestContainerOutput(ctx, f, "consume configMaps", pod, 0, []string{
			"CONFIG_DATA_1=value-1",
		})
	})

	/*
		Release: v1.9
		Testname: ConfigMap, from environment variables
		Description: Create a Pod with a environment source from ConfigMap. All ConfigMap values MUST be available as environment variables in the container.
	*/
	framework.ConformanceIt("should be consumable via the environment", f.WithNodeConformance(), func(ctx context.Context) {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newConfigMap(f, name)
		ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, configMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "env-test",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env"},
						EnvFrom: []v1.EnvFromSource{
							{
								ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
							{
								Prefix:       "p-",
								ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		e2epodoutput.TestContainerOutput(ctx, f, "consume configMaps", pod, 0, []string{
			"data-1=value-1", "data-2=value-2", "data-3=value-3",
			"p-data-1=value-1", "p-data-2=value-2", "p-data-3=value-3",
		})
	})

	/*
	   Release: v1.14
	   Testname: ConfigMap, with empty-key
	   Description: Attempt to create a ConfigMap with an empty key. The creation MUST fail.
	*/
	framework.ConformanceIt("should fail to create ConfigMap with empty key", func(ctx context.Context) {
		configMap, err := newConfigMapWithEmptyKey(ctx, f)
		gomega.Expect(err).To(gomega.HaveOccurred(), "created configMap %q with empty key in namespace %q", configMap.Name, f.Namespace.Name)
	})

	ginkgo.It("should update ConfigMap successfully", func(ctx context.Context) {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newConfigMap(f, name)
		ginkgo.By(fmt.Sprintf("Creating ConfigMap %v/%v", f.Namespace.Name, configMap.Name))
		_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, configMap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create ConfigMap")

		configMap.Data = map[string]string{
			"data": "value",
		}
		ginkgo.By(fmt.Sprintf("Updating configMap %v/%v", f.Namespace.Name, configMap.Name))
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(ctx, configMap, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "failed to update ConfigMap")

		configMapFromUpdate, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Get(ctx, name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get ConfigMap")
		ginkgo.By(fmt.Sprintf("Verifying update of ConfigMap %v/%v", f.Namespace.Name, configMap.Name))
		gomega.Expect(configMapFromUpdate.Data).To(gomega.Equal(configMap.Data))
	})

	/*
	   Release: v1.19
	   Testname: ConfigMap lifecycle
	   Description: Attempt to create a ConfigMap. Patch the created ConfigMap. Fetching the ConfigMap MUST reflect changes.
	   By fetching all the ConfigMaps via a Label selector it MUST find the ConfigMap by it's static label and updated value. The ConfigMap must be deleted by Collection.
	*/
	framework.ConformanceIt("should run through a ConfigMap lifecycle", func(ctx context.Context) {
		testNamespaceName := f.Namespace.Name
		testConfigMapName := "test-configmap" + string(uuid.NewUUID())

		testConfigMap := v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: testConfigMapName,
				Labels: map[string]string{
					"test-configmap-static": "true",
				},
			},
			Data: map[string]string{
				"valueName": "value",
			},
		}

		ginkgo.By("creating a ConfigMap")
		_, err := f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).Create(ctx, &testConfigMap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create ConfigMap")

		ginkgo.By("fetching the ConfigMap")
		configMap, err := f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).Get(ctx, testConfigMapName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get ConfigMap")
		gomega.Expect(configMap.Data["valueName"]).To(gomega.Equal(testConfigMap.Data["valueName"]))
		gomega.Expect(configMap.Labels["test-configmap-static"]).To(gomega.Equal(testConfigMap.Labels["test-configmap-static"]))

		configMapPatchPayload, err := json.Marshal(v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"test-configmap": "patched",
				},
			},
			Data: map[string]string{
				"valueName": "value1",
			},
		})
		framework.ExpectNoError(err, "failed to marshal patch data")

		ginkgo.By("patching the ConfigMap")
		_, err = f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).Patch(ctx, testConfigMapName, types.StrategicMergePatchType, []byte(configMapPatchPayload), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch ConfigMap")

		ginkgo.By("listing all ConfigMaps in all namespaces with a label selector")
		configMapList, err := f.ClientSet.CoreV1().ConfigMaps("").List(ctx, metav1.ListOptions{
			LabelSelector: "test-configmap=patched",
		})
		framework.ExpectNoError(err, "failed to list ConfigMaps with LabelSelector")
		testConfigMapFound := false
		for _, cm := range configMapList.Items {
			if cm.ObjectMeta.Name == testConfigMap.ObjectMeta.Name &&
				cm.ObjectMeta.Namespace == testNamespaceName &&
				cm.ObjectMeta.Labels["test-configmap-static"] == testConfigMap.ObjectMeta.Labels["test-configmap-static"] &&
				cm.ObjectMeta.Labels["test-configmap"] == "patched" &&
				cm.Data["valueName"] == "value1" {
				testConfigMapFound = true
				break
			}
		}
		if !testConfigMapFound {
			framework.Failf("failed to find ConfigMap %s/%s by label selector", testNamespaceName, testConfigMap.ObjectMeta.Name)
		}

		ginkgo.By("deleting the ConfigMap by collection with a label selector")
		err = f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{
			LabelSelector: "test-configmap-static=true",
		})
		framework.ExpectNoError(err, "failed to delete ConfigMap collection with LabelSelector")

		ginkgo.By("listing all ConfigMaps in test namespace")
		configMapList, err = f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).List(ctx, metav1.ListOptions{
			LabelSelector: "test-configmap-static=true",
		})
		framework.ExpectNoError(err, "failed to list ConfigMap by LabelSelector")
		gomega.Expect(configMapList.Items).To(gomega.BeEmpty(), "ConfigMap is still present after being deleted by collection")
	})

	/*
		Release: v1.34
		Testname: ConfigMap, from environment field with various prefixes
		Description: Create a Pod with environment variable values set using values from ConfigMap.
		Allows users to use envFrom to set prefixes with various printable ASCII characters excluding '=' as environment variable names.
		This test verifies that different prefixes including digits, special characters, and letters can be correctly used.
	*/
	framework.ConformanceIt("should be consumable as environment variable names with various prefixes", func(ctx context.Context) {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newConfigMap(f, name)
		ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, configMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "env-test",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env"},
						EnvFrom: []v1.EnvFromSource{
							{
								// No prefix
								ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
							{
								// Prefix starting with a digit
								Prefix:       "1-",
								ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
							{
								// Prefix with special characters
								Prefix:       "$_-",
								ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
							{
								// Prefix with uppercase letters
								Prefix:       "ABC_",
								ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
							{
								// Prefix with symbols
								Prefix:       "#@!",
								ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		e2epodoutput.TestContainerOutput(ctx, f, "consume configMaps", pod, 0, []string{
			// Original values without prefix
			"data-1=value-1", "data-2=value-2", "data-3=value-3",
			// Values with digit prefix
			"1-data-1=value-1", "1-data-2=value-2", "1-data-3=value-3",
			// Values with special character prefix
			"$_-data-1=value-1", "$_-data-2=value-2", "$_-data-3=value-3",
			// Values with uppercase letter prefix
			"ABC_data-1=value-1", "ABC_data-2=value-2", "ABC_data-3=value-3",
			// Values with symbol prefix
			"#@!data-1=value-1", "#@!data-2=value-2", "#@!data-3=value-3",
		})
	})
})

func newConfigMap(f *framework.Framework, name string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: f.Namespace.Name,
			Name:      name,
		},
		Data: map[string]string{
			"data-1": "value-1",
			"data-2": "value-2",
			"data-3": "value-3",
		},
	}
}

func newConfigMapWithEmptyKey(ctx context.Context, f *framework.Framework) (*v1.ConfigMap, error) {
	name := "configmap-test-emptyKey-" + string(uuid.NewUUID())
	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: f.Namespace.Name,
			Name:      name,
		},
		Data: map[string]string{
			"": "value-1",
		},
	}

	ginkgo.By(fmt.Sprintf("Creating configMap that has name %s", configMap.Name))
	return f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, configMap, metav1.CreateOptions{})
}
