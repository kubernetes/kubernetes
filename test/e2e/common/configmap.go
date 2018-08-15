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
	"fmt"

	. "github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = Describe("[sig-api-machinery] ConfigMap", func() {
	f := framework.NewDefaultFramework("configmap")

	/*
		Release : v1.9
		Testname: ConfigMap, from environment field
		Description: Create a Pod with an environment variable value set using a value from ConfigMap. A ConfigMap value MUST be accessible in the container environment.
	*/
	framework.ConformanceIt("should be consumable via environment variable [NodeConformance]", func() {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newConfigMap(f, name)
		By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
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
		By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
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
