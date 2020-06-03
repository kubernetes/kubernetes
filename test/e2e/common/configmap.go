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

package common

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	watch "k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

var _ = ginkgo.Describe("[sig-node] ConfigMap", func() {
	f := framework.NewDefaultFramework("configmap")

	var dc dynamic.Interface

	ginkgo.BeforeEach(func() {
		dc = f.DynamicClient
	})

	/*
		Release : v1.9
		Testname: ConfigMap, from environment field
		Description: Create a Pod with an environment variable value set using a value from ConfigMap. A ConfigMap value MUST be accessible in the container environment.
	*/
	framework.ConformanceIt("should be consumable via environment variable [NodeConformance]", func() {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newConfigMap(f, name)
		ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
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

		f.TestContainerOutput("consume configMaps", pod, 0, []string{
			"CONFIG_DATA_1=value-1",
		})
	})

	/*
		Release: v1.9
		Testname: ConfigMap, from environment variables
		Description: Create a Pod with a environment source from ConfigMap. All ConfigMap values MUST be available as environment variables in the container.
	*/
	framework.ConformanceIt("should be consumable via the environment [NodeConformance]", func() {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newEnvFromConfigMap(f, name)
		ginkgo.By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
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
								Prefix:       "p_",
								ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: name}},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("consume configMaps", pod, 0, []string{
			"data_1=value-1", "data_2=value-2", "data_3=value-3",
			"p_data_1=value-1", "p_data_2=value-2", "p_data_3=value-3",
		})
	})

	/*
	   Release : v1.14
	   Testname: ConfigMap, with empty-key
	   Description: Attempt to create a ConfigMap with an empty key. The creation MUST fail.
	*/
	framework.ConformanceIt("should fail to create ConfigMap with empty key", func() {
		configMap, err := newConfigMapWithEmptyKey(f)
		framework.ExpectError(err, "created configMap %q with empty key in namespace %q", configMap.Name, f.Namespace.Name)
	})

	ginkgo.It("should update ConfigMap successfully", func() {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newConfigMap(f, name)
		ginkgo.By(fmt.Sprintf("Creating ConfigMap %v/%v", f.Namespace.Name, configMap.Name))
		_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create ConfigMap")

		configMap.Data = map[string]string{
			"data": "value",
		}
		ginkgo.By(fmt.Sprintf("Updating configMap %v/%v", f.Namespace.Name, configMap.Name))
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(context.TODO(), configMap, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "failed to update ConfigMap")

		configMapFromUpdate, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Get(context.TODO(), name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get ConfigMap")
		ginkgo.By(fmt.Sprintf("Verifying update of ConfigMap %v/%v", f.Namespace.Name, configMap.Name))
		framework.ExpectEqual(configMapFromUpdate.Data, configMap.Data)
	})

	ginkgo.It("should run through a ConfigMap lifecycle", func() {
		testNamespaceName := f.Namespace.Name
		testConfigMapName := "test-configmap" + string(uuid.NewUUID())

		configMapResource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"}
		expectedWatchEvents := []watch.Event{
			{Type: watch.Added},
			{Type: watch.Modified},
			{Type: watch.Deleted},
		}
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

		framework.WatchEventSequenceVerifier(context.TODO(), dc, configMapResource, testNamespaceName, testConfigMapName, metav1.ListOptions{LabelSelector: "test-configmap-static=true"}, expectedWatchEvents, func(retryWatcher *watchtools.RetryWatcher) (actualWatchEvents []watch.Event) {
			ginkgo.By("creating a ConfigMap")
			_, err := f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).Create(context.TODO(), &testConfigMap, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create ConfigMap")
			eventFound := false
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			_, err = framework.WatchUntilWithoutRetry(ctx, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				if watchEvent.Type != watch.Added {
					return false, nil
				}
				actualWatchEvents = append(actualWatchEvents, watchEvent)
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")
			framework.ExpectEqual(eventFound, true, "failed to find ConfigMap %v event", watch.Added)

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
			_, err = f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).Patch(context.TODO(), testConfigMapName, types.StrategicMergePatchType, []byte(configMapPatchPayload), metav1.PatchOptions{})
			framework.ExpectNoError(err, "failed to patch ConfigMap")
			ginkgo.By("waiting for the ConfigMap to be modified")
			eventFound = false
			ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			_, err = framework.WatchUntilWithoutRetry(ctx, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				if watchEvent.Type != watch.Modified {
					return false, nil
				}
				actualWatchEvents = append(actualWatchEvents, watchEvent)
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")
			framework.ExpectEqual(eventFound, true, "failed to find ConfigMap %v event", watch.Modified)

			ginkgo.By("fetching the ConfigMap")
			configMap, err := f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).Get(context.TODO(), testConfigMapName, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to get ConfigMap")
			framework.ExpectEqual(configMap.Data["valueName"], "value1", "failed to patch ConfigMap")
			framework.ExpectEqual(configMap.Labels["test-configmap"], "patched", "failed to patch ConfigMap")

			ginkgo.By("listing all ConfigMaps in all namespaces")
			configMapList, err := f.ClientSet.CoreV1().ConfigMaps("").List(context.TODO(), metav1.ListOptions{
				LabelSelector: "test-configmap-static=true",
			})
			framework.ExpectNoError(err, "failed to list ConfigMaps with LabelSelector")
			framework.ExpectNotEqual(len(configMapList.Items), 0, "no ConfigMaps found in ConfigMap list")
			testConfigMapFound := false
			for _, cm := range configMapList.Items {
				if cm.ObjectMeta.Name == testConfigMapName &&
					cm.ObjectMeta.Namespace == testNamespaceName &&
					cm.ObjectMeta.Labels["test-configmap-static"] == "true" &&
					cm.Data["valueName"] == "value1" {
					testConfigMapFound = true
					break
				}
			}
			framework.ExpectEqual(testConfigMapFound, true, "failed to find ConfigMap in list")

			ginkgo.By("deleting the ConfigMap by a collection")
			err = f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{
				LabelSelector: "test-configmap-static=true",
			})
			framework.ExpectNoError(err, "failed to delete ConfigMap collection with LabelSelector")
			ginkgo.By("waiting for the ConfigMap to be deleted")
			eventFound = false
			ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			_, err = framework.WatchUntilWithoutRetry(ctx, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				if watchEvent.Type != watch.Deleted {
					return false, nil
				}
				actualWatchEvents = append(actualWatchEvents, watchEvent)
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")
			framework.ExpectEqual(eventFound, true, "failed to find ConfigMap %v event", watch.Deleted)

			return actualWatchEvents
		}, func() (err error) {
			_ = f.ClientSet.CoreV1().ConfigMaps(testNamespaceName).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "test-configmap-static=true"})
			return err
		})
	})
})

func newEnvFromConfigMap(f *framework.Framework, name string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: f.Namespace.Name,
			Name:      name,
		},
		Data: map[string]string{
			"data_1": "value-1",
			"data_2": "value-2",
			"data_3": "value-3",
		},
	}
}

func newConfigMapWithEmptyKey(f *framework.Framework) (*v1.ConfigMap, error) {
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
	return f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{})
}
