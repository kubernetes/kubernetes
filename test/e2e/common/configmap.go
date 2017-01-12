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
	"os"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("ConfigMap", func() {
	f := framework.NewDefaultFramework("configmap")

	It("should be consumable from pods in volume [Conformance] [Volume]", func() {
		doConfigMapE2EWithoutMappings(f, 0, 0, nil)
	})

	It("should be consumable from pods in volume with defaultMode set [Conformance] [Volume]", func() {
		defaultMode := int32(0400)
		doConfigMapE2EWithoutMappings(f, 0, 0, &defaultMode)
	})

	It("should be consumable from pods in volume as non-root with defaultMode and fsGroup set [Feature:FSGroup] [Volume]", func() {
		defaultMode := int32(0440) /* setting fsGroup sets mode to at least 440 */
		doConfigMapE2EWithoutMappings(f, 1000, 1001, &defaultMode)
	})

	It("should be consumable from pods in volume as non-root [Conformance] [Volume]", func() {
		doConfigMapE2EWithoutMappings(f, 1000, 0, nil)
	})

	It("should be consumable from pods in volume as non-root with FSGroup [Feature:FSGroup] [Volume]", func() {
		doConfigMapE2EWithoutMappings(f, 1000, 1001, nil)
	})

	It("should be consumable from pods in volume with mappings [Conformance] [Volume]", func() {
		doConfigMapE2EWithMappings(f, 0, 0, nil)
	})

	It("should be consumable from pods in volume with mappings and Item mode set[Conformance] [Volume]", func() {
		mode := int32(0400)
		doConfigMapE2EWithMappings(f, 0, 0, &mode)
	})

	It("should be consumable from pods in volume with mappings as non-root [Conformance] [Volume]", func() {
		doConfigMapE2EWithMappings(f, 1000, 0, nil)
	})

	It("should be consumable from pods in volume with mappings as non-root with FSGroup [Feature:FSGroup] [Volume]", func() {
		doConfigMapE2EWithMappings(f, 1000, 1001, nil)
	})

	It("updates should be reflected in volume [Conformance] [Volume]", func() {

		// We may have to wait or a full sync period to elapse before the
		// Kubelet projects the update into the volume and the container picks
		// it up. This timeout is based on the default Kubelet sync period (1
		// minute) plus additional time for fudge factor.
		const podLogTimeout = 300 * time.Second

		name := "configmap-test-upd-" + string(uuid.NewUUID())
		volumeName := "configmap-volume"
		volumeMountPath := "/etc/configmap-volume"
		containerName := "configmap-volume-test"

		configMap := &v1.ConfigMap{
			ObjectMeta: v1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string]string{
				"data-1": "value-1",
			},
		}

		By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		var err error
		if configMap, err = f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: "pod-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: volumeName,
						VolumeSource: v1.VolumeSource{
							ConfigMap: &v1.ConfigMapVolumeSource{
								LocalObjectReference: v1.LocalObjectReference{
									Name: name,
								},
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    containerName,
						Image:   "gcr.io/google_containers/mounttest:0.7",
						Command: []string{"/mt", "--break_on_expected_content=false", "--retry_time=120", "--file_content_in_loop=/etc/configmap-volume/data-1"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}
		By("Creating the pod")
		f.PodClient().CreateSync(pod)

		pollLogs := func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName)
		}

		Eventually(pollLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("value-1"))

		By(fmt.Sprintf("Updating configmap %v", configMap.Name))
		configMap.ResourceVersion = "" // to force update
		configMap.Data["data-1"] = "value-2"
		_, err = f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Update(configMap)
		Expect(err).NotTo(HaveOccurred(), "Failed to update configmap %q in namespace %q", configMap.Name, f.Namespace.Name)

		By("waiting to observe update in volume")
		Eventually(pollLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("value-2"))
	})

	It("should be consumable via environment variable [Conformance]", func() {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newConfigMap(f, name)
		By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		var err error
		if configMap, err = f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: "pod-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "env-test",
						Image:   "gcr.io/google_containers/busybox:1.24",
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

	It("should be consumable via the environment [Conformance]", func() {
		name := "configmap-test-" + string(uuid.NewUUID())
		configMap := newEnvFromConfigMap(f, name)
		By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		var err error
		if configMap, err = f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: "pod-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "env-test",
						Image:   "gcr.io/google_containers/busybox:1.24",
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

	It("should be consumable in multiple volumes in the same pod [Conformance] [Volume]", func() {
		var (
			name             = "configmap-test-volume-" + string(uuid.NewUUID())
			volumeName       = "configmap-volume"
			volumeMountPath  = "/etc/configmap-volume"
			volumeName2      = "configmap-volume-2"
			volumeMountPath2 = "/etc/configmap-volume-2"
			configMap        = newConfigMap(f, name)
		)

		By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		var err error
		if configMap, err = f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: v1.ObjectMeta{
				Name: "pod-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: volumeName,
						VolumeSource: v1.VolumeSource{
							ConfigMap: &v1.ConfigMapVolumeSource{
								LocalObjectReference: v1.LocalObjectReference{
									Name: name,
								},
							},
						},
					},
					{
						Name: volumeName2,
						VolumeSource: v1.VolumeSource{
							ConfigMap: &v1.ConfigMapVolumeSource{
								LocalObjectReference: v1.LocalObjectReference{
									Name: name,
								},
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:  "configmap-volume-test",
						Image: "gcr.io/google_containers/mounttest:0.7",
						Args:  []string{"--file_content=/etc/configmap-volume/data-1"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
							{
								Name:      volumeName2,
								MountPath: volumeMountPath2,
								ReadOnly:  true,
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("consume configMaps", pod, 0, []string{
			"content of file \"/etc/configmap-volume/data-1\": value-1",
		})

	})
})

func newConfigMap(f *framework.Framework, name string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: v1.ObjectMeta{
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

func newEnvFromConfigMap(f *framework.Framework, name string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: v1.ObjectMeta{
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

func doConfigMapE2EWithoutMappings(f *framework.Framework, uid, fsGroup int64, defaultMode *int32) {
	var (
		name            = "configmap-test-volume-" + string(uuid.NewUUID())
		volumeName      = "configmap-volume"
		volumeMountPath = "/etc/configmap-volume"
		configMap       = newConfigMap(f, name)
	)

	By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
	var err error
	if configMap, err = f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}

	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "pod-configmaps-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{},
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: name,
							},
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "configmap-volume-test",
					Image: "gcr.io/google_containers/mounttest:0.7",
					Args: []string{
						"--file_content=/etc/configmap-volume/data-1",
						"--file_mode=/etc/configmap-volume/data-1"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumeMountPath,
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	if uid != 0 {
		pod.Spec.SecurityContext.RunAsUser = &uid
	}

	if fsGroup != 0 {
		pod.Spec.SecurityContext.FSGroup = &fsGroup
	}
	if defaultMode != nil {
		pod.Spec.Volumes[0].VolumeSource.ConfigMap.DefaultMode = defaultMode
	} else {
		mode := int32(0644)
		defaultMode = &mode
	}

	modeString := fmt.Sprintf("%v", os.FileMode(*defaultMode))
	output := []string{
		"content of file \"/etc/configmap-volume/data-1\": value-1",
		"mode of file \"/etc/configmap-volume/data-1\": " + modeString,
	}
	f.TestContainerOutput("consume configMaps", pod, 0, output)
}

func doConfigMapE2EWithMappings(f *framework.Framework, uid, fsGroup int64, itemMode *int32) {
	var (
		name            = "configmap-test-volume-map-" + string(uuid.NewUUID())
		volumeName      = "configmap-volume"
		volumeMountPath = "/etc/configmap-volume"
		configMap       = newConfigMap(f, name)
	)

	By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))

	var err error
	if configMap, err = f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}

	pod := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "pod-configmaps-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{},
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: name,
							},
							Items: []v1.KeyToPath{
								{
									Key:  "data-2",
									Path: "path/to/data-2",
								},
							},
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "configmap-volume-test",
					Image: "gcr.io/google_containers/mounttest:0.7",
					Args: []string{"--file_content=/etc/configmap-volume/path/to/data-2",
						"--file_mode=/etc/configmap-volume/path/to/data-2"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumeMountPath,
							ReadOnly:  true,
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	if uid != 0 {
		pod.Spec.SecurityContext.RunAsUser = &uid
	}

	if fsGroup != 0 {
		pod.Spec.SecurityContext.FSGroup = &fsGroup
	}
	if itemMode != nil {
		pod.Spec.Volumes[0].VolumeSource.ConfigMap.Items[0].Mode = itemMode
	} else {
		mode := int32(0644)
		itemMode = &mode
	}

	// Just check file mode if fsGroup is not set. If fsGroup is set, the
	// final mode is adjusted and we are not testing that case.
	output := []string{
		"content of file \"/etc/configmap-volume/path/to/data-2\": value-2",
	}
	if fsGroup == 0 {
		modeString := fmt.Sprintf("%v", os.FileMode(*itemMode))
		output = append(output, "mode of file \"/etc/configmap-volume/path/to/data-2\": "+modeString)
	}
	f.TestContainerOutput("consume configMaps", pod, 0, output)
}
