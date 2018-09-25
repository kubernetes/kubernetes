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
	"os"
	"path"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("[sig-storage] Projected", func() {
	// Part 1/3 - Secrets
	f := framework.NewDefaultFramework("projected")

	/*
	   Release : v1.9
	   Testname: Projected Volume, Secrets, volume mode default
	   Description: A Pod is created with a projected volume source ‘secret’ to store a secret with a specified key with default permission mode. Pod MUST be able to read the content of the key successfully and the mode MUST be -rw-r--r-- by default.
	*/
	framework.ConformanceIt("should be consumable from pods in volume [NodeConformance]", func() {
		doProjectedSecretE2EWithoutMapping(f, nil /* default mode */, "projected-secret-test-"+string(uuid.NewUUID()), nil, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, Secrets, volume mode 0400
	   Description: A Pod is created with a projected volume source ‘secret’ to store a secret with a specified key with permission mode set to 0x400 on the Pod. Pod MUST be able to read the content of the key successfully and the mode MUST be -r—-—————.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with defaultMode set [NodeConformance]", func() {
		defaultMode := int32(0400)
		doProjectedSecretE2EWithoutMapping(f, &defaultMode, "projected-secret-test-"+string(uuid.NewUUID()), nil, nil)
	})

	/*
	   Release : v1.9
	   Testname: Project Volume, Secrets, non-root, custom fsGroup
	   Description: A Pod is created with a projected volume source ‘secret’ to store a secret with a specified key. The volume has permission mode set to 0440, fsgroup set to 1001 and user set to non-root uid of 1000. Pod MUST be able to read the content of the key successfully and the mode MUST be -r—-r————-.
	*/
	framework.ConformanceIt("should be consumable from pods in volume as non-root with defaultMode and fsGroup set [NodeConformance]", func() {
		defaultMode := int32(0440) /* setting fsGroup sets mode to at least 440 */
		fsGroup := int64(1001)
		uid := int64(1000)
		doProjectedSecretE2EWithoutMapping(f, &defaultMode, "projected-secret-test-"+string(uuid.NewUUID()), &fsGroup, &uid)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, Secrets, mapped
	   Description: A Pod is created with a projected volume source ‘secret’ to store a secret with a specified key with default permission mode. The secret is also mapped to a custom path. Pod MUST be able to read the content of the key successfully and the mode MUST be -r—-—————— on the mapped volume.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings [NodeConformance]", func() {
		doProjectedSecretE2EWithMapping(f, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, Secrets, mapped, volume mode 0400
	   Description: A Pod is created with a projected volume source ‘secret’ to store a secret with a specified key with permission mode set to 0400. The secret is also mapped to a specific name. Pod MUST be able to read the content of the key successfully and the mode MUST be -r—-—————— on the mapped volume.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings and Item Mode set [NodeConformance]", func() {
		mode := int32(0400)
		doProjectedSecretE2EWithMapping(f, &mode)
	})

	It("should be able to mount in a volume regardless of a different secret existing with same name in different namespace [NodeConformance]", func() {
		var (
			namespace2  *v1.Namespace
			err         error
			secret2Name = "projected-secret-test-" + string(uuid.NewUUID())
		)

		if namespace2, err = f.CreateNamespace("secret-namespace", nil); err != nil {
			framework.Failf("unable to create new namespace %s: %v", namespace2.Name, err)
		}

		secret2 := secretForTest(namespace2.Name, secret2Name)
		secret2.Data = map[string][]byte{
			"this_should_not_match_content_of_other_secret": []byte("similarly_this_should_not_match_content_of_other_secret\n"),
		}
		if secret2, err = f.ClientSet.CoreV1().Secrets(namespace2.Name).Create(secret2); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret2.Name, err)
		}
		doProjectedSecretE2EWithoutMapping(f, nil /* default mode */, secret2.Name, nil, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, Secrets, mapped, multiple paths
	   Description: A Pod is created with a projected volume source ‘secret’ to store a secret with a specified key. The secret is mapped to two different volume mounts. Pod MUST be able to read the content of the key successfully from the two volume mounts and the mode MUST be -r—-—————— on the mapped volumes.
	*/
	framework.ConformanceIt("should be consumable in multiple volumes in a pod [NodeConformance]", func() {
		// This test ensures that the same secret can be mounted in multiple
		// volumes in the same pod.  This test case exists to prevent
		// regressions that break this use-case.
		var (
			name             = "projected-secret-test-" + string(uuid.NewUUID())
			volumeName       = "projected-secret-volume"
			volumeMountPath  = "/etc/projected-secret-volume"
			volumeName2      = "projected-secret-volume-2"
			volumeMountPath2 = "/etc/projected-secret-volume-2"
			secret           = secretForTest(f.Namespace.Name, name)
		)

		By(fmt.Sprintf("Creating secret with name %s", secret.Name))
		var err error
		if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(secret); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-projected-secrets-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: volumeName,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										Secret: &v1.SecretProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: name,
											},
										},
									},
								},
							},
						},
					},
					{
						Name: volumeName2,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										Secret: &v1.SecretProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: name,
											},
										},
									},
								},
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:  "secret-volume-test",
						Image: imageutils.GetE2EImage(imageutils.Mounttest),
						Args: []string{
							"--file_content=/etc/projected-secret-volume/data-1",
							"--file_mode=/etc/projected-secret-volume/data-1"},
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

		f.TestContainerOutput("consume secrets", pod, 0, []string{
			"content of file \"/etc/projected-secret-volume/data-1\": value-1",
			"mode of file \"/etc/projected-secret-volume/data-1\": -rw-r--r--",
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, Secrets, create, update delete
	   Description: Create a Pod with three containers with secrets namely a create, update and delete container. Create Container when started MUST no have a secret, update and delete containers MUST be created with a secret value. Create a secret in the create container, the Pod MUST be able to read the secret from the create container. Update the secret in the update container, Pod MUST be able to read the updated secret value. Delete the secret in the delete container. Pod MUST fail to read the secret from the delete container.
	*/
	framework.ConformanceIt("optional updates should be reflected in volume [NodeConformance]", func() {
		podLogTimeout := framework.GetPodSecretUpdateTimeout(f.ClientSet)
		containerTimeoutArg := fmt.Sprintf("--retry_time=%v", int(podLogTimeout.Seconds()))
		trueVal := true
		volumeMountPath := "/etc/projected-secret-volumes"

		deleteName := "s-test-opt-del-" + string(uuid.NewUUID())
		deleteContainerName := "dels-volume-test"
		deleteVolumeName := "deletes-volume"
		deleteSecret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      deleteName,
			},
			Data: map[string][]byte{
				"data-1": []byte("value-1"),
			},
		}

		updateName := "s-test-opt-upd-" + string(uuid.NewUUID())
		updateContainerName := "upds-volume-test"
		updateVolumeName := "updates-volume"
		updateSecret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      updateName,
			},
			Data: map[string][]byte{
				"data-1": []byte("value-1"),
			},
		}

		createName := "s-test-opt-create-" + string(uuid.NewUUID())
		createContainerName := "creates-volume-test"
		createVolumeName := "creates-volume"
		createSecret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      createName,
			},
			Data: map[string][]byte{
				"data-1": []byte("value-1"),
			},
		}

		By(fmt.Sprintf("Creating secret with name %s", deleteSecret.Name))
		var err error
		if deleteSecret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(deleteSecret); err != nil {
			framework.Failf("unable to create test secret %s: %v", deleteSecret.Name, err)
		}

		By(fmt.Sprintf("Creating secret with name %s", updateSecret.Name))
		if updateSecret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(updateSecret); err != nil {
			framework.Failf("unable to create test secret %s: %v", updateSecret.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-projected-secrets-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: deleteVolumeName,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										Secret: &v1.SecretProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: deleteName,
											},
											Optional: &trueVal,
										},
									},
								},
							},
						},
					},
					{
						Name: updateVolumeName,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										Secret: &v1.SecretProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: updateName,
											},
											Optional: &trueVal,
										},
									},
								},
							},
						},
					},
					{
						Name: createVolumeName,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										Secret: &v1.SecretProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: createName,
											},
											Optional: &trueVal,
										},
									},
								},
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    deleteContainerName,
						Image:   imageutils.GetE2EImage(imageutils.Mounttest),
						Command: []string{"/mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-secret-volumes/delete/data-1"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      deleteVolumeName,
								MountPath: path.Join(volumeMountPath, "delete"),
								ReadOnly:  true,
							},
						},
					},
					{
						Name:    updateContainerName,
						Image:   imageutils.GetE2EImage(imageutils.Mounttest),
						Command: []string{"/mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-secret-volumes/update/data-3"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      updateVolumeName,
								MountPath: path.Join(volumeMountPath, "update"),
								ReadOnly:  true,
							},
						},
					},
					{
						Name:    createContainerName,
						Image:   imageutils.GetE2EImage(imageutils.Mounttest),
						Command: []string{"/mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-secret-volumes/create/data-1"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      createVolumeName,
								MountPath: path.Join(volumeMountPath, "create"),
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

		pollCreateLogs := func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, createContainerName)
		}
		Eventually(pollCreateLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("Error reading file /etc/projected-secret-volumes/create/data-1"))

		pollUpdateLogs := func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, updateContainerName)
		}
		Eventually(pollUpdateLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("Error reading file /etc/projected-secret-volumes/update/data-3"))

		pollDeleteLogs := func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, deleteContainerName)
		}
		Eventually(pollDeleteLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("value-1"))

		By(fmt.Sprintf("Deleting secret %v", deleteSecret.Name))
		err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Delete(deleteSecret.Name, &metav1.DeleteOptions{})
		Expect(err).NotTo(HaveOccurred(), "Failed to delete secret %q in namespace %q", deleteSecret.Name, f.Namespace.Name)

		By(fmt.Sprintf("Updating secret %v", updateSecret.Name))
		updateSecret.ResourceVersion = "" // to force update
		delete(updateSecret.Data, "data-1")
		updateSecret.Data["data-3"] = []byte("value-3")
		_, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Update(updateSecret)
		Expect(err).NotTo(HaveOccurred(), "Failed to update secret %q in namespace %q", updateSecret.Name, f.Namespace.Name)

		By(fmt.Sprintf("Creating secret with name %s", createSecret.Name))
		if createSecret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(createSecret); err != nil {
			framework.Failf("unable to create test secret %s: %v", createSecret.Name, err)
		}

		By("waiting to observe update in volume")

		Eventually(pollCreateLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("value-1"))
		Eventually(pollUpdateLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("value-3"))
		Eventually(pollDeleteLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("Error reading file /etc/projected-secret-volumes/delete/data-1"))
	})

	// Part 2/3 - ConfigMaps
	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, volume mode default
	   Description: A Pod is created with projected volume source ‘ConfigMap’ to store a configMap with default permission mode. Pod MUST be able to read the content of the ConfigMap successfully and the mode on the volume MUST be -rw-r—-r—-.
	*/
	framework.ConformanceIt("should be consumable from pods in volume [NodeConformance]", func() {
		doProjectedConfigMapE2EWithoutMappings(f, 0, 0, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, volume mode 0400
	   Description: A Pod is created with projected volume source ‘ConfigMap’ to store a configMap with permission mode set to 0400. Pod MUST be able to read the content of the ConfigMap successfully and the mode on the volume MUST be -r——-——-—-.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with defaultMode set [NodeConformance]", func() {
		defaultMode := int32(0400)
		doProjectedConfigMapE2EWithoutMappings(f, 0, 0, &defaultMode)
	})

	It("should be consumable from pods in volume as non-root with defaultMode and fsGroup set [NodeFeature:FSGroup]", func() {
		defaultMode := int32(0440) /* setting fsGroup sets mode to at least 440 */
		doProjectedConfigMapE2EWithoutMappings(f, 1000, 1001, &defaultMode)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, non-root user
	   Description: A Pod is created with projected volume source ‘ConfigMap’ to store a configMap as non-root user with uid 1000. Pod MUST be able to read the content of the ConfigMap successfully and the mode on the volume MUST be -rw—r——r—-.
	*/
	framework.ConformanceIt("should be consumable from pods in volume as non-root [NodeConformance]", func() {
		doProjectedConfigMapE2EWithoutMappings(f, 1000, 0, nil)
	})

	It("should be consumable from pods in volume as non-root with FSGroup [NodeFeature:FSGroup]", func() {
		doProjectedConfigMapE2EWithoutMappings(f, 1000, 1001, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, mapped
	   Description: A Pod is created with projected volume source ‘ConfigMap’ to store a configMap with default permission mode. The ConfigMap is also mapped to a custom path. Pod MUST be able to read the content of the ConfigMap from the custom location successfully and the mode on the volume MUST be -rw—r——r—-.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings [NodeConformance]", func() {
		doProjectedConfigMapE2EWithMappings(f, 0, 0, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, mapped, volume mode 0400
	   Description: A Pod is created with projected volume source ‘ConfigMap’ to store a configMap with permission mode set to 0400. The ConfigMap is also mapped to a custom path. Pod MUST be able to read the content of the ConfigMap from the custom location successfully and the mode on the volume MUST be -r-—r——r—-.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings and Item mode set [NodeConformance]", func() {
		mode := int32(0400)
		doProjectedConfigMapE2EWithMappings(f, 0, 0, &mode)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, mapped, non-root user
	   Description: A Pod is created with projected volume source ‘ConfigMap’ to store a configMap as non-root user with uid 1000. The ConfigMap is also mapped to a custom path. Pod MUST be able to read the content of the ConfigMap from the custom location successfully and the mode on the volume MUST be -r-—r——r—-.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings as non-root [NodeConformance]", func() {
		doProjectedConfigMapE2EWithMappings(f, 1000, 0, nil)
	})

	It("should be consumable from pods in volume with mappings as non-root with FSGroup [NodeFeature:FSGroup]", func() {
		doProjectedConfigMapE2EWithMappings(f, 1000, 1001, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, update
	   Description: A Pod is created with projected volume source ‘ConfigMap’ to store a configMap and performs a create and update to new value. Pod MUST be able to create the configMap with value-1. Pod MUST be able to update the value in the confgiMap to value-2.
	*/
	framework.ConformanceIt("updates should be reflected in volume [NodeConformance]", func() {
		podLogTimeout := framework.GetPodSecretUpdateTimeout(f.ClientSet)
		containerTimeoutArg := fmt.Sprintf("--retry_time=%v", int(podLogTimeout.Seconds()))

		name := "projected-configmap-test-upd-" + string(uuid.NewUUID())
		volumeName := "projected-configmap-volume"
		volumeMountPath := "/etc/projected-configmap-volume"
		containerName := "projected-configmap-volume-test"
		configMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string]string{
				"data-1": "value-1",
			},
		}

		By(fmt.Sprintf("Creating projection with configMap that has name %s", configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-projected-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: volumeName,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										ConfigMap: &v1.ConfigMapProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: name,
											},
										},
									},
								},
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    containerName,
						Image:   imageutils.GetE2EImage(imageutils.Mounttest),
						Command: []string{"/mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-configmap-volume/data-1"},
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
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(configMap)
		Expect(err).NotTo(HaveOccurred(), "Failed to update configmap %q in namespace %q", configMap.Name, f.Namespace.Name)

		By("waiting to observe update in volume")
		Eventually(pollLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("value-2"))
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, create, update and delete
	   Description: Create a Pod with three containers with ConfigMaps namely a create, update and delete container. Create Container when started MUST not have configMap, update and delete containers MUST be created with a ConfigMap value as ‘value-1’. Create a configMap in the create container, the Pod MUST be able to read the configMap from the create container. Update the configMap in the update container, Pod MUST be able to read the updated configMap value. Delete the configMap in the delete container. Pod MUST fail to read the configMap from the delete container.
	*/
	framework.ConformanceIt("optional updates should be reflected in volume [NodeConformance]", func() {
		podLogTimeout := framework.GetPodSecretUpdateTimeout(f.ClientSet)
		containerTimeoutArg := fmt.Sprintf("--retry_time=%v", int(podLogTimeout.Seconds()))
		trueVal := true
		volumeMountPath := "/etc/projected-configmap-volumes"

		deleteName := "cm-test-opt-del-" + string(uuid.NewUUID())
		deleteContainerName := "delcm-volume-test"
		deleteVolumeName := "deletecm-volume"
		deleteConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      deleteName,
			},
			Data: map[string]string{
				"data-1": "value-1",
			},
		}

		updateName := "cm-test-opt-upd-" + string(uuid.NewUUID())
		updateContainerName := "updcm-volume-test"
		updateVolumeName := "updatecm-volume"
		updateConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      updateName,
			},
			Data: map[string]string{
				"data-1": "value-1",
			},
		}

		createName := "cm-test-opt-create-" + string(uuid.NewUUID())
		createContainerName := "createcm-volume-test"
		createVolumeName := "createcm-volume"
		createConfigMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      createName,
			},
			Data: map[string]string{
				"data-1": "value-1",
			},
		}

		By(fmt.Sprintf("Creating configMap with name %s", deleteConfigMap.Name))
		var err error
		if deleteConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(deleteConfigMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", deleteConfigMap.Name, err)
		}

		By(fmt.Sprintf("Creating configMap with name %s", updateConfigMap.Name))
		if updateConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(updateConfigMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", updateConfigMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-projected-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: deleteVolumeName,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										ConfigMap: &v1.ConfigMapProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: deleteName,
											},
											Optional: &trueVal,
										},
									},
								},
							},
						},
					},
					{
						Name: updateVolumeName,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										ConfigMap: &v1.ConfigMapProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: updateName,
											},
											Optional: &trueVal,
										},
									},
								},
							},
						},
					},
					{
						Name: createVolumeName,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										ConfigMap: &v1.ConfigMapProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: createName,
											},
											Optional: &trueVal,
										},
									},
								},
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    deleteContainerName,
						Image:   imageutils.GetE2EImage(imageutils.Mounttest),
						Command: []string{"/mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-configmap-volumes/delete/data-1"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      deleteVolumeName,
								MountPath: path.Join(volumeMountPath, "delete"),
								ReadOnly:  true,
							},
						},
					},
					{
						Name:    updateContainerName,
						Image:   imageutils.GetE2EImage(imageutils.Mounttest),
						Command: []string{"/mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-configmap-volumes/update/data-3"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      updateVolumeName,
								MountPath: path.Join(volumeMountPath, "update"),
								ReadOnly:  true,
							},
						},
					},
					{
						Name:    createContainerName,
						Image:   imageutils.GetE2EImage(imageutils.Mounttest),
						Command: []string{"/mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-configmap-volumes/create/data-1"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      createVolumeName,
								MountPath: path.Join(volumeMountPath, "create"),
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

		pollCreateLogs := func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, createContainerName)
		}
		Eventually(pollCreateLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("Error reading file /etc/projected-configmap-volumes/create/data-1"))

		pollUpdateLogs := func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, updateContainerName)
		}
		Eventually(pollUpdateLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("Error reading file /etc/projected-configmap-volumes/update/data-3"))

		pollDeleteLogs := func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, deleteContainerName)
		}
		Eventually(pollDeleteLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("value-1"))

		By(fmt.Sprintf("Deleting configmap %v", deleteConfigMap.Name))
		err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Delete(deleteConfigMap.Name, &metav1.DeleteOptions{})
		Expect(err).NotTo(HaveOccurred(), "Failed to delete configmap %q in namespace %q", deleteConfigMap.Name, f.Namespace.Name)

		By(fmt.Sprintf("Updating configmap %v", updateConfigMap.Name))
		updateConfigMap.ResourceVersion = "" // to force update
		delete(updateConfigMap.Data, "data-1")
		updateConfigMap.Data["data-3"] = "value-3"
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(updateConfigMap)
		Expect(err).NotTo(HaveOccurred(), "Failed to update configmap %q in namespace %q", updateConfigMap.Name, f.Namespace.Name)

		By(fmt.Sprintf("Creating configMap with name %s", createConfigMap.Name))
		if createConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(createConfigMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", createConfigMap.Name, err)
		}

		By("waiting to observe update in volume")

		Eventually(pollCreateLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("value-1"))
		Eventually(pollUpdateLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("value-3"))
		Eventually(pollDeleteLogs, podLogTimeout, framework.Poll).Should(ContainSubstring("Error reading file /etc/projected-configmap-volumes/delete/data-1"))
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, multiple volume paths
	   Description: A Pod is created with a projected volume source ‘ConfigMap’ to store a configMap. The configMap is mapped to two different volume mounts. Pod MUST be able to read the content of the configMap successfully from the two volume mounts.
	*/
	framework.ConformanceIt("should be consumable in multiple volumes in the same pod [NodeConformance]", func() {
		var (
			name             = "projected-configmap-test-volume-" + string(uuid.NewUUID())
			volumeName       = "projected-configmap-volume"
			volumeMountPath  = "/etc/projected-configmap-volume"
			volumeName2      = "projected-configmap-volume-2"
			volumeMountPath2 = "/etc/projected-configmap-volume-2"
			configMap        = newConfigMap(f, name)
		)

		By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-projected-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: volumeName,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										ConfigMap: &v1.ConfigMapProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: name,
											},
										},
									},
								},
							},
						},
					},
					{
						Name: volumeName2,
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{

										ConfigMap: &v1.ConfigMapProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: name,
											},
										},
									},
								},
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:  "projected-configmap-volume-test",
						Image: imageutils.GetE2EImage(imageutils.Mounttest),
						Args:  []string{"--file_content=/etc/projected-configmap-volume/data-1"},
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
			"content of file \"/etc/projected-configmap-volume/data-1\": value-1",
		})

	})

	// Part 3/3 - DownwardAPI
	// How long to wait for a log pod to be displayed
	const podLogTimeout = 2 * time.Minute
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, pod name
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the pod name from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide podname only [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumePodForSimpleTest(podName, "/etc/podinfo/podname")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("%s\n", podName),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, volume mode 0400
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. The default mode for the volume mount is set to 0400. Pod MUST be able to read the pod name from the mounted DownwardAPIVolumeFiles and the volume mode must be -r—-—————.
	*/
	framework.ConformanceIt("should set DefaultMode on files [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		defaultMode := int32(0400)
		pod := projectedDownwardAPIVolumePodForModeTest(podName, "/etc/podinfo/podname", nil, &defaultMode)

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			"mode of file \"/etc/podinfo/podname\": -r--------",
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, volume mode 0400
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. The default mode for the volume mount is set to 0400. Pod MUST be able to read the pod name from the mounted DownwardAPIVolumeFiles and the volume mode must be -r—-—————.
	*/
	framework.ConformanceIt("should set mode on item file [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		mode := int32(0400)
		pod := projectedDownwardAPIVolumePodForModeTest(podName, "/etc/podinfo/podname", &mode, nil)

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			"mode of file \"/etc/podinfo/podname\": -r--------",
		})
	})

	It("should provide podname as non-root with fsgroup [NodeFeature:FSGroup]", func() {
		podName := "metadata-volume-" + string(uuid.NewUUID())
		uid := int64(1001)
		gid := int64(1234)
		pod := downwardAPIVolumePodForSimpleTest(podName, "/etc/podinfo/podname")
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			RunAsUser: &uid,
			FSGroup:   &gid,
		}
		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("%s\n", podName),
		})
	})

	It("should provide podname as non-root with fsgroup and defaultMode [NodeFeature:FSGroup]", func() {
		podName := "metadata-volume-" + string(uuid.NewUUID())
		uid := int64(1001)
		gid := int64(1234)
		mode := int32(0440) /* setting fsGroup sets mode to at least 440 */
		pod := projectedDownwardAPIVolumePodForModeTest(podName, "/etc/podinfo/podname", &mode, nil)
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			RunAsUser: &uid,
			FSGroup:   &gid,
		}
		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			"mode of file \"/etc/podinfo/podname\": -r--r-----",
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, update labels
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests and label items. Pod MUST be able to read the labels from the mounted DownwardAPIVolumeFiles. Labels are then updated. Pod MUST be able to read the updated values for the Labels.
	*/
	framework.ConformanceIt("should update labels on modification [NodeConformance]", func() {
		labels := map[string]string{}
		labels["key1"] = "value1"
		labels["key2"] = "value2"

		podName := "labelsupdate" + string(uuid.NewUUID())
		pod := projectedDownwardAPIVolumePodForUpdateTest(podName, labels, map[string]string{}, "/etc/podinfo/labels")
		containerName := "client-container"
		By("Creating the pod")
		podClient.CreateSync(pod)

		Eventually(func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, podName, containerName)
		},
			podLogTimeout, framework.Poll).Should(ContainSubstring("key1=\"value1\"\n"))

		//modify labels
		podClient.Update(podName, func(pod *v1.Pod) {
			pod.Labels["key3"] = "value3"
		})

		Eventually(func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName)
		},
			podLogTimeout, framework.Poll).Should(ContainSubstring("key3=\"value3\"\n"))
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, update annotation
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests and annotation items. Pod MUST be able to read the annotations from the mounted DownwardAPIVolumeFiles. Annotations are then updated. Pod MUST be able to read the updated values for the Annotations.
	*/
	framework.ConformanceIt("should update annotations on modification [NodeConformance]", func() {
		annotations := map[string]string{}
		annotations["builder"] = "bar"
		podName := "annotationupdate" + string(uuid.NewUUID())
		pod := projectedDownwardAPIVolumePodForUpdateTest(podName, map[string]string{}, annotations, "/etc/podinfo/annotations")

		containerName := "client-container"
		By("Creating the pod")
		podClient.CreateSync(pod)

		pod, err := podClient.Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "Failed to get pod %q", pod.Name)

		Eventually(func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName)
		},
			podLogTimeout, framework.Poll).Should(ContainSubstring("builder=\"bar\"\n"))

		//modify annotations
		podClient.Update(podName, func(pod *v1.Pod) {
			pod.Annotations["builder"] = "foo"
		})

		Eventually(func() (string, error) {
			return framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName)
		},
			podLogTimeout, framework.Poll).Should(ContainSubstring("builder=\"foo\"\n"))
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, CPU limits
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the cpu limits from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide container's cpu limit [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/podinfo/cpu_limit")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("2\n"),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, memory limits
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the memory limits from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide container's memory limit [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/podinfo/memory_limit")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("67108864\n"),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, CPU request
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the cpu request from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide container's cpu request [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/podinfo/cpu_request")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("1\n"),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, memory request
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests. Pod MUST be able to read the memory request from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide container's memory request [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForContainerResources(podName, "/etc/podinfo/memory_request")

		f.TestContainerOutput("downward API volume plugin", pod, 0, []string{
			fmt.Sprintf("33554432\n"),
		})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, CPU limit, node allocatable
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests.  The CPU and memory resources for requests and limits are NOT specified for the container. Pod MUST be able to read the default cpu limits from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide node allocatable (cpu) as default cpu limit if the limit is not set [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForDefaultContainerResources(podName, "/etc/podinfo/cpu_limit")

		f.TestContainerOutputRegexp("downward API volume plugin", pod, 0, []string{"[1-9]"})
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, DownwardAPI, memory limit, node allocatable
	   Description: A Pod is created with a projected volume source for downwardAPI with pod name, cpu and memory limits and cpu and memory requests.  The CPU and memory resources for requests and limits are NOT specified for the container. Pod MUST be able to read the default memory limits from the mounted DownwardAPIVolumeFiles.
	*/
	framework.ConformanceIt("should provide node allocatable (memory) as default memory limit if the limit is not set [NodeConformance]", func() {
		podName := "downwardapi-volume-" + string(uuid.NewUUID())
		pod := downwardAPIVolumeForDefaultContainerResources(podName, "/etc/podinfo/memory_limit")

		f.TestContainerOutputRegexp("downward API volume plugin", pod, 0, []string{"[1-9]"})
	})

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

func doProjectedSecretE2EWithoutMapping(f *framework.Framework, defaultMode *int32,
	secretName string, fsGroup *int64, uid *int64) {
	var (
		volumeName      = "projected-secret-volume"
		volumeMountPath = "/etc/projected-secret-volume"
		secret          = secretForTest(f.Namespace.Name, secretName)
	)

	By(fmt.Sprintf("Creating projection with secret that has name %s", secret.Name))
	var err error
	if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(secret); err != nil {
		framework.Failf("unable to create test secret %s: %v", secret.Name, err)
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-projected-secrets-" + string(uuid.NewUUID()),
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						Projected: &v1.ProjectedVolumeSource{
							Sources: []v1.VolumeProjection{
								{
									Secret: &v1.SecretProjection{
										LocalObjectReference: v1.LocalObjectReference{
											Name: secretName,
										},
									},
								},
							},
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "projected-secret-volume-test",
					Image: imageutils.GetE2EImage(imageutils.Mounttest),
					Args: []string{
						"--file_content=/etc/projected-secret-volume/data-1",
						"--file_mode=/etc/projected-secret-volume/data-1"},
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

	if defaultMode != nil {
		//pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].Secret.DefaultMode = defaultMode
		pod.Spec.Volumes[0].VolumeSource.Projected.DefaultMode = defaultMode
	} else {
		mode := int32(0644)
		defaultMode = &mode
	}

	if fsGroup != nil || uid != nil {
		pod.Spec.SecurityContext = &v1.PodSecurityContext{
			FSGroup:   fsGroup,
			RunAsUser: uid,
		}
	}

	modeString := fmt.Sprintf("%v", os.FileMode(*defaultMode))
	expectedOutput := []string{
		"content of file \"/etc/projected-secret-volume/data-1\": value-1",
		"mode of file \"/etc/projected-secret-volume/data-1\": " + modeString,
	}

	f.TestContainerOutput("consume secrets", pod, 0, expectedOutput)
}

func doProjectedSecretE2EWithMapping(f *framework.Framework, mode *int32) {
	var (
		name            = "projected-secret-test-map-" + string(uuid.NewUUID())
		volumeName      = "projected-secret-volume"
		volumeMountPath = "/etc/projected-secret-volume"
		secret          = secretForTest(f.Namespace.Name, name)
	)

	By(fmt.Sprintf("Creating projection with secret that has name %s", secret.Name))
	var err error
	if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(secret); err != nil {
		framework.Failf("unable to create test secret %s: %v", secret.Name, err)
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod-projected-secrets-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						Projected: &v1.ProjectedVolumeSource{
							Sources: []v1.VolumeProjection{
								{
									Secret: &v1.SecretProjection{
										LocalObjectReference: v1.LocalObjectReference{
											Name: name,
										},
										Items: []v1.KeyToPath{
											{
												Key:  "data-1",
												Path: "new-path-data-1",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "projected-secret-volume-test",
					Image: imageutils.GetE2EImage(imageutils.Mounttest),
					Args: []string{
						"--file_content=/etc/projected-secret-volume/new-path-data-1",
						"--file_mode=/etc/projected-secret-volume/new-path-data-1"},
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

	if mode != nil {
		//pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].Secret.Items[0].Mode = mode
		pod.Spec.Volumes[0].VolumeSource.Projected.DefaultMode = mode
	} else {
		defaultItemMode := int32(0644)
		mode = &defaultItemMode
	}

	modeString := fmt.Sprintf("%v", os.FileMode(*mode))
	expectedOutput := []string{
		"content of file \"/etc/projected-secret-volume/new-path-data-1\": value-1",
		"mode of file \"/etc/projected-secret-volume/new-path-data-1\": " + modeString,
	}

	f.TestContainerOutput("consume secrets", pod, 0, expectedOutput)
}

func doProjectedConfigMapE2EWithoutMappings(f *framework.Framework, uid, fsGroup int64, defaultMode *int32) {
	userID := int64(uid)
	groupID := int64(fsGroup)

	var (
		name            = "projected-configmap-test-volume-" + string(uuid.NewUUID())
		volumeName      = "projected-configmap-volume"
		volumeMountPath = "/etc/projected-configmap-volume"
		configMap       = newConfigMap(f, name)
	)

	By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
	var err error
	if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod-projected-configmaps-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{},
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						Projected: &v1.ProjectedVolumeSource{
							Sources: []v1.VolumeProjection{
								{
									ConfigMap: &v1.ConfigMapProjection{
										LocalObjectReference: v1.LocalObjectReference{
											Name: name,
										},
									},
								},
							},
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "projected-configmap-volume-test",
					Image: imageutils.GetE2EImage(imageutils.Mounttest),
					Args: []string{
						"--file_content=/etc/projected-configmap-volume/data-1",
						"--file_mode=/etc/projected-configmap-volume/data-1"},
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

	if userID != 0 {
		pod.Spec.SecurityContext.RunAsUser = &userID
	}

	if groupID != 0 {
		pod.Spec.SecurityContext.FSGroup = &groupID
	}

	if defaultMode != nil {
		//pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].ConfigMap.DefaultMode = defaultMode
		pod.Spec.Volumes[0].VolumeSource.Projected.DefaultMode = defaultMode
	} else {
		mode := int32(0644)
		defaultMode = &mode
	}

	modeString := fmt.Sprintf("%v", os.FileMode(*defaultMode))
	output := []string{
		"content of file \"/etc/projected-configmap-volume/data-1\": value-1",
		"mode of file \"/etc/projected-configmap-volume/data-1\": " + modeString,
	}
	f.TestContainerOutput("consume configMaps", pod, 0, output)
}

func doProjectedConfigMapE2EWithMappings(f *framework.Framework, uid, fsGroup int64, itemMode *int32) {
	userID := int64(uid)
	groupID := int64(fsGroup)

	var (
		name            = "projected-configmap-test-volume-map-" + string(uuid.NewUUID())
		volumeName      = "projected-configmap-volume"
		volumeMountPath = "/etc/projected-configmap-volume"
		configMap       = newConfigMap(f, name)
	)

	By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))

	var err error
	if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(configMap); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod-projected-configmaps-" + string(uuid.NewUUID()),
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{},
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						Projected: &v1.ProjectedVolumeSource{
							Sources: []v1.VolumeProjection{
								{
									ConfigMap: &v1.ConfigMapProjection{
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
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "projected-configmap-volume-test",
					Image: imageutils.GetE2EImage(imageutils.Mounttest),
					Args: []string{"--file_content=/etc/projected-configmap-volume/path/to/data-2",
						"--file_mode=/etc/projected-configmap-volume/path/to/data-2"},
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

	if userID != 0 {
		pod.Spec.SecurityContext.RunAsUser = &userID
	}

	if groupID != 0 {
		pod.Spec.SecurityContext.FSGroup = &groupID
	}

	if itemMode != nil {
		//pod.Spec.Volumes[0].VolumeSource.ConfigMap.Items[0].Mode = itemMode
		pod.Spec.Volumes[0].VolumeSource.Projected.DefaultMode = itemMode
	} else {
		mode := int32(0644)
		itemMode = &mode
	}

	// Just check file mode if fsGroup is not set. If fsGroup is set, the
	// final mode is adjusted and we are not testing that case.
	output := []string{
		"content of file \"/etc/projected-configmap-volume/path/to/data-2\": value-2",
	}
	if fsGroup == 0 {
		modeString := fmt.Sprintf("%v", os.FileMode(*itemMode))
		output = append(output, "mode of file \"/etc/projected-configmap-volume/path/to/data-2\": "+modeString)
	}
	f.TestContainerOutput("consume configMaps", pod, 0, output)
}

func projectedDownwardAPIVolumePodForModeTest(name, filePath string, itemMode, defaultMode *int32) *v1.Pod {
	pod := projectedDownwardAPIVolumeBasePod(name, nil, nil)

	pod.Spec.Containers = []v1.Container{
		{
			Name:    "client-container",
			Image:   imageutils.GetE2EImage(imageutils.Mounttest),
			Command: []string{"/mounttest", "--file_mode=" + filePath},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "podinfo",
					MountPath: "/etc/podinfo",
				},
			},
		},
	}
	if itemMode != nil {
		pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items[0].Mode = itemMode
	}
	if defaultMode != nil {
		pod.Spec.Volumes[0].VolumeSource.Projected.DefaultMode = defaultMode
	}

	return pod
}

func projectedDownwardAPIVolumePodForUpdateTest(name string, labels, annotations map[string]string, filePath string) *v1.Pod {
	pod := projectedDownwardAPIVolumeBasePod(name, labels, annotations)

	pod.Spec.Containers = []v1.Container{
		{
			Name:    "client-container",
			Image:   imageutils.GetE2EImage(imageutils.Mounttest),
			Command: []string{"/mounttest", "--break_on_expected_content=false", "--retry_time=1200", "--file_content_in_loop=" + filePath},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "podinfo",
					MountPath: "/etc/podinfo",
					ReadOnly:  false,
				},
			},
		},
	}

	applyLabelsAndAnnotationsToProjectedDownwardAPIPod(labels, annotations, pod)
	return pod
}

func projectedDownwardAPIVolumeBasePod(name string, labels, annotations map[string]string) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
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
											{
												Path: "cpu_limit",
												ResourceFieldRef: &v1.ResourceFieldSelector{
													ContainerName: "client-container",
													Resource:      "limits.cpu",
												},
											},
											{
												Path: "cpu_request",
												ResourceFieldRef: &v1.ResourceFieldSelector{
													ContainerName: "client-container",
													Resource:      "requests.cpu",
												},
											},
											{
												Path: "memory_limit",
												ResourceFieldRef: &v1.ResourceFieldSelector{
													ContainerName: "client-container",
													Resource:      "limits.memory",
												},
											},
											{
												Path: "memory_request",
												ResourceFieldRef: &v1.ResourceFieldSelector{
													ContainerName: "client-container",
													Resource:      "requests.memory",
												},
											},
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

func applyLabelsAndAnnotationsToProjectedDownwardAPIPod(labels, annotations map[string]string, pod *v1.Pod) {
	if len(labels) > 0 {
		pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items = append(pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items, v1.DownwardAPIVolumeFile{
			Path: "labels",
			FieldRef: &v1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.labels",
			},
		})
	}

	if len(annotations) > 0 {
		pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items = append(pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items, v1.DownwardAPIVolumeFile{
			Path: "annotations",
			FieldRef: &v1.ObjectFieldSelector{
				APIVersion: "v1",
				FieldPath:  "metadata.annotations",
			},
		})
	}
}

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
