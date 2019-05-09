/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("[sig-storage] Projected combined", func() {
	f := framework.NewDefaultFramework("projected")

	// Test multiple projections
	/*
	   Release : v1.9
	   Testname: Projected Volume, multiple projections
	   Description: A Pod is created with a projected volume source for secrets, configMap and downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the secrets, configMap values and the cpu and memory limits as well as cpu and memory requests from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should project all components that make up the projection API [Projection][NodeConformance]", func() {
		var err error
		podName := "projected-volume-" + string(uuid.NewUUID())
		secretName := "secret-projected-all-test-volume-" + string(uuid.NewUUID())
		configMapName := "configmap-projected-all-test-volume-" + string(uuid.NewUUID())
		configMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      configMapName,
			},
			Data: map[string]string{
				"configmap-data": "configmap-value-1",
			},
		}
		secret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      secretName,
			},
			Data: map[string][]byte{
				"secret-data": []byte("secret-value-1"),
			},
		}

		By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}
		By(fmt.Sprintf("Creating secret with name %s", secret.Name))
		if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(secret); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		pod := projectedAllVolumeBasePod(podName, secretName, configMapName, nil, nil)
		pod.Spec.Containers = []v1.Container{
			{
				Name:    "projected-all-volume-test",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"sh", "-c", "cat /all/podname && cat /all/secret-data && cat /all/configmap-data"},
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      "podinfo",
						MountPath: "/all",
						ReadOnly:  false,
					},
				},
			},
		}
		f.TestContainerOutput("Check all projections for projected volume plugin", pod, 0, []string{
			fmt.Sprintf("%s", podName),
			"secret-value-1",
			"configmap-value-1",
		})
	})
})

func projectedAllVolumeBasePod(podName string, secretName string, configMapName string, labels, annotations map[string]string) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "podinfo",
					VolumeSource: v1.VolumeSource{
						Projected: &v1.ProjectedVolumeSource{
							Sources: []v1.VolumeProjection{
								{
									DownwardAPI: &v1.DownwardAPIProjection{
										Items: []v1.DownwardAPIVolumeFile{
											{
												Path: "podname",
												FieldRef: &v1.ObjectFieldSelector{
													APIVersion: "v1",
													FieldPath:  "metadata.name",
												},
											},
										},
									},
								},
								{
									Secret: &v1.SecretProjection{
										LocalObjectReference: v1.LocalObjectReference{
											Name: secretName,
										},
									},
								},
								{
									ConfigMap: &v1.ConfigMapProjection{
										LocalObjectReference: v1.LocalObjectReference{
											Name: configMapName,
										},
									},
								},
							},
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	return pod
}
