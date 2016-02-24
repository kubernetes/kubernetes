/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("ConfigMap", func() {

	f := NewDefaultFramework("configmap")

	It("should be consumable from pods in volume [Conformance]", func() {
		name := "configmap-test-volume-" + string(util.NewUUID())
		volumeName := "configmap-volume"
		volumeMountPath := "/etc/configmap-volume"

		configMap := &api.ConfigMap{
			ObjectMeta: api.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string]string{
				"data-1": "value-1",
				"data-2": "value-2",
				"data-3": "value-3",
			},
		}

		By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		defer func() {
			By("Cleaning up the configMap")
			if err := f.Client.ConfigMaps(f.Namespace.Name).Delete(configMap.Name); err != nil {
				Failf("unable to delete configMap %v: %v", configMap.Name, err)
			}
		}()
		var err error
		if configMap, err = f.Client.ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-configmaps-" + string(util.NewUUID()),
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: volumeName,
						VolumeSource: api.VolumeSource{
							ConfigMap: &api.ConfigMapVolumeSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: name,
								},
							},
						},
					},
				},
				Containers: []api.Container{
					{
						Name:  "configmap-volume-test",
						Image: "gcr.io/google_containers/mounttest:0.6",
						Args:  []string{"--file_content=/etc/configmap-volume/data-1"},
						VolumeMounts: []api.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		testContainerOutput("consume configMaps", f.Client, pod, 0, []string{
			"content of file \"/etc/configmap-volume/data-1\": value-1",
		}, f.Namespace.Name)
	})

	It("should be consumable from pods in volume with mappings [Conformance]", func() {
		name := "configmap-test-volume-map-" + string(util.NewUUID())
		volumeName := "configmap-volume"
		volumeMountPath := "/etc/configmap-volume"

		configMap := &api.ConfigMap{
			ObjectMeta: api.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string]string{
				"data-1": "value-1",
				"data-2": "value-2",
				"data-3": "value-3",
			},
		}

		By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		defer func() {
			By("Cleaning up the configMap")
			if err := f.Client.ConfigMaps(f.Namespace.Name).Delete(configMap.Name); err != nil {
				Failf("unable to delete configMap %v: %v", configMap.Name, err)
			}
		}()
		var err error
		if configMap, err = f.Client.ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-configmaps-" + string(util.NewUUID()),
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: volumeName,
						VolumeSource: api.VolumeSource{
							ConfigMap: &api.ConfigMapVolumeSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: name,
								},
								Items: []api.KeyToPath{
									{
										Key:  "data-2",
										Path: "path/to/data-2",
									},
								},
							},
						},
					},
				},
				Containers: []api.Container{
					{
						Name:  "configmap-volume-test",
						Image: "gcr.io/google_containers/mounttest:0.6",
						Args:  []string{"--file_content=/etc/configmap-volume/path/to/data-2"},
						VolumeMounts: []api.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		testContainerOutput("consume configMaps", f.Client, pod, 0, []string{
			"content of file \"/etc/configmap-volume/path/to/data-2\": value-2",
		}, f.Namespace.Name)
	})

	It("updates should be reflected in volume [Conformance]", func() {

		// We may have to wait or a full sync period to elapse before the
		// Kubelet projects the update into the volume and the container picks
		// it up. This timeout is based on the default Kubelet sync period (1
		// minute) plus additional time for fudge factor.
		const podLogTimeout = 300 * time.Second

		name := "configmap-test-upd-" + string(util.NewUUID())
		volumeName := "configmap-volume"
		volumeMountPath := "/etc/configmap-volume"
		containerName := "configmap-volume-test"

		configMap := &api.ConfigMap{
			ObjectMeta: api.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string]string{
				"data-1": "value-1",
			},
		}

		By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		defer func() {
			By("Cleaning up the configMap")
			if err := f.Client.ConfigMaps(f.Namespace.Name).Delete(configMap.Name); err != nil {
				Failf("unable to delete configMap %v: %v", configMap.Name, err)
			}
		}()
		var err error
		if configMap, err = f.Client.ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-configmaps-" + string(util.NewUUID()),
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: volumeName,
						VolumeSource: api.VolumeSource{
							ConfigMap: &api.ConfigMapVolumeSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: name,
								},
							},
						},
					},
				},
				Containers: []api.Container{
					{
						Name:    containerName,
						Image:   "gcr.io/google_containers/mounttest:0.6",
						Command: []string{"/mt", "--break_on_expected_content=false", "--retry_time=120", "--file_content_in_loop=/etc/configmap-volume/data-1"},
						VolumeMounts: []api.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		defer func() {
			By("Deleting the pod")
			f.Client.Pods(f.Namespace.Name).Delete(pod.Name, api.NewDeleteOptions(0))
		}()
		By("Creating the pod")
		_, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		expectNoError(waitForPodRunningInNamespace(f.Client, pod.Name, f.Namespace.Name))

		pollLogs := func() (string, error) {
			return getPodLogs(f.Client, f.Namespace.Name, pod.Name, containerName)
		}

		Eventually(pollLogs, podLogTimeout, poll).Should(ContainSubstring("value-1"))

		By(fmt.Sprintf("Updating configmap %v", configMap.Name))
		configMap.ResourceVersion = "" // to force update
		configMap.Data["data-1"] = "value-2"
		_, err = f.Client.ConfigMaps(f.Namespace.Name).Update(configMap)
		Expect(err).NotTo(HaveOccurred())

		By("waiting to observe update in volume")
		Eventually(pollLogs, podLogTimeout, poll).Should(ContainSubstring("value-2"))
	})

	It("should be consumable via environment variable [Conformance]", func() {
		name := "configmap-test-" + string(util.NewUUID())
		configMap := &api.ConfigMap{
			ObjectMeta: api.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string]string{
				"data-1": "value-1",
				"data-2": "value-2",
				"data-3": "value-3",
			},
		}

		By(fmt.Sprintf("Creating configMap %v/%v", f.Namespace.Name, configMap.Name))
		defer func() {
			By("Cleaning up the configMap")
			if err := f.Client.ConfigMaps(f.Namespace.Name).Delete(configMap.Name); err != nil {
				Failf("unable to delete configMap %v: %v", configMap.Name, err)
			}
		}()
		var err error
		if configMap, err = f.Client.ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-configmaps-" + string(util.NewUUID()),
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "env-test",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"sh", "-c", "env"},
						Env: []api.EnvVar{
							{
								Name: "CONFIG_DATA_1",
								ValueFrom: &api.EnvVarSource{
									ConfigMapKeyRef: &api.ConfigMapKeySelector{
										LocalObjectReference: api.LocalObjectReference{
											Name: name,
										},
										Key: "data-1",
									},
								},
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		testContainerOutput("consume configMaps", f.Client, pod, 0, []string{
			"CONFIG_DATA_1=value-1",
		}, f.Namespace.Name)
	})
})
