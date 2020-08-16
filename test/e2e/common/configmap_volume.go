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
	"fmt"
	"path"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = ginkgo.Describe("[sig-storage] ConfigMap", func() {
	f := framework.NewDefaultFramework("configmap")

	/*
		Release: v1.9
		Testname: ConfigMap Volume, without mapping
		Description: Create a ConfigMap, create a Pod that mounts a volume and populates the volume with data stored in the ConfigMap. The ConfigMap that is created MUST be accessible to read from the newly created Pod using the volume mount. The data content of the file MUST be readable and verified and file modes MUST default to 0x644.
	*/
	framework.ConformanceIt("should be consumable from pods in volume [NodeConformance]", func() {
		doConfigMapE2EWithoutMappings(f, false, 0, nil)
	})

	/*
		Release: v1.9
		Testname: ConfigMap Volume, without mapping, volume mode set
		Description: Create a ConfigMap, create a Pod that mounts a volume and populates the volume with data stored in the ConfigMap. File mode is changed to a custom value of '0x400'. The ConfigMap that is created MUST be accessible to read from the newly created Pod using the volume mount. The data content of the file MUST be readable and verified and file modes MUST be set to the custom value of '0x400'
		This test is marked LinuxOnly since Windows does not support setting specific file permissions.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with defaultMode set [LinuxOnly] [NodeConformance]", func() {
		defaultMode := int32(0400)
		doConfigMapE2EWithoutMappings(f, false, 0, &defaultMode)
	})

	ginkgo.It("should be consumable from pods in volume as non-root with defaultMode and fsGroup set [LinuxOnly] [NodeFeature:FSGroup]", func() {
		// Windows does not support RunAsUser / FSGroup SecurityContext options, and it does not support setting file permissions.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
		defaultMode := int32(0440) /* setting fsGroup sets mode to at least 440 */
		doConfigMapE2EWithoutMappings(f, true, 1001, &defaultMode)
	})

	/*
		Release: v1.9
		Testname: ConfigMap Volume, without mapping, non-root user
		Description: Create a ConfigMap, create a Pod that mounts a volume and populates the volume with data stored in the ConfigMap. Pod is run as a non-root user with uid=1000. The ConfigMap that is created MUST be accessible to read from the newly created Pod using the volume mount. The file on the volume MUST have file mode set to default value of 0x644.
	*/
	framework.ConformanceIt("should be consumable from pods in volume as non-root [NodeConformance]", func() {
		doConfigMapE2EWithoutMappings(f, true, 0, nil)
	})

	ginkgo.It("should be consumable from pods in volume as non-root with FSGroup [LinuxOnly] [NodeFeature:FSGroup]", func() {
		// Windows does not support RunAsUser / FSGroup SecurityContext options.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
		doConfigMapE2EWithoutMappings(f, true, 1001, nil)
	})

	/*
		Release: v1.9
		Testname: ConfigMap Volume, with mapping
		Description: Create a ConfigMap, create a Pod that mounts a volume and populates the volume with data stored in the ConfigMap. Files are mapped to a path in the volume. The ConfigMap that is created MUST be accessible to read from the newly created Pod using the volume mount. The data content of the file MUST be readable and verified and file modes MUST default to 0x644.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings [NodeConformance]", func() {
		doConfigMapE2EWithMappings(f, false, 0, nil)
	})

	/*
		Release: v1.9
		Testname: ConfigMap Volume, with mapping, volume mode set
		Description: Create a ConfigMap, create a Pod that mounts a volume and populates the volume with data stored in the ConfigMap. Files are mapped to a path in the volume. File mode is changed to a custom value of '0x400'. The ConfigMap that is created MUST be accessible to read from the newly created Pod using the volume mount. The data content of the file MUST be readable and verified and file modes MUST be set to the custom value of '0x400'
		This test is marked LinuxOnly since Windows does not support setting specific file permissions.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings and Item mode set [LinuxOnly] [NodeConformance]", func() {
		mode := int32(0400)
		doConfigMapE2EWithMappings(f, false, 0, &mode)
	})

	/*
		Release: v1.9
		Testname: ConfigMap Volume, with mapping, non-root user
		Description: Create a ConfigMap, create a Pod that mounts a volume and populates the volume with data stored in the ConfigMap. Files are mapped to a path in the volume. Pod is run as a non-root user with uid=1000. The ConfigMap that is created MUST be accessible to read from the newly created Pod using the volume mount. The file on the volume MUST have file mode set to default value of 0x644.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings as non-root [NodeConformance]", func() {
		doConfigMapE2EWithMappings(f, true, 0, nil)
	})

	ginkgo.It("should be consumable from pods in volume with mappings as non-root with FSGroup [LinuxOnly] [NodeFeature:FSGroup]", func() {
		// Windows does not support RunAsUser / FSGroup SecurityContext options.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
		doConfigMapE2EWithMappings(f, true, 1001, nil)
	})

	/*
		Release: v1.9
		Testname: ConfigMap Volume, update
		Description: The ConfigMap that is created MUST be accessible to read from the newly created Pod using the volume mount that is mapped to custom path in the Pod. When the ConfigMap is updated the change to the config map MUST be verified by reading the content from the mounted file in the Pod.
	*/
	framework.ConformanceIt("updates should be reflected in volume [NodeConformance]", func() {
		podLogTimeout := e2epod.GetPodSecretUpdateTimeout(f.ClientSet)
		containerTimeoutArg := fmt.Sprintf("--retry_time=%v", int(podLogTimeout.Seconds()))

		name := "configmap-test-upd-" + string(uuid.NewUUID())
		volumeName := "configmap-volume"
		volumeMountPath := "/etc/configmap-volume"
		containerName := "configmap-volume-test"

		configMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string]string{
				"data-1": "value-1",
			},
		}

		ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
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
						Name:  containerName,
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/configmap-volume/data-1"},
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
		ginkgo.By("Creating the pod")
		f.PodClient().CreateSync(pod)

		pollLogs := func() (string, error) {
			return e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName)
		}

		gomega.Eventually(pollLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("value-1"))

		ginkgo.By(fmt.Sprintf("Updating configmap %v", configMap.Name))
		configMap.ResourceVersion = "" // to force update
		configMap.Data["data-1"] = "value-2"
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(context.TODO(), configMap, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "Failed to update configmap %q in namespace %q", configMap.Name, f.Namespace.Name)

		ginkgo.By("waiting to observe update in volume")
		gomega.Eventually(pollLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("value-2"))
	})

	/*
		Release: v1.12
		Testname: ConfigMap Volume, text data, binary data
		Description: The ConfigMap that is created with text data and binary data MUST be accessible to read from the newly created Pod using the volume mount that is mapped to custom path in the Pod. ConfigMap's text data and binary data MUST be verified by reading the content from the mounted files in the Pod.
	*/
	framework.ConformanceIt("binary data should be reflected in volume [NodeConformance]", func() {
		podLogTimeout := e2epod.GetPodSecretUpdateTimeout(f.ClientSet)
		containerTimeoutArg := fmt.Sprintf("--retry_time=%v", int(podLogTimeout.Seconds()))

		name := "configmap-test-upd-" + string(uuid.NewUUID())
		volumeName := "configmap-volume"
		volumeMountPath := "/etc/configmap-volume"
		containerName1 := "configmap-volume-data-test"
		containerName2 := "configmap-volume-binary-test"

		configMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string]string{
				"data-1": "value-1",
			},
			BinaryData: map[string][]byte{
				"dump.bin": {0xde, 0xca, 0xfe, 0xba, 0xd0, 0xfe, 0xff},
			},
		}

		ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
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
						Name:  containerName1,
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/configmap-volume/data-1"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
						},
					},
					{
						Name:    containerName2,
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"hexdump", "-C", "/etc/configmap-volume/dump.bin"},
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
		ginkgo.By("Creating the pod")
		f.PodClient().CreateSync(pod)

		pollLogs1 := func() (string, error) {
			return e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName1)
		}
		pollLogs2 := func() (string, error) {
			return e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, containerName2)
		}

		ginkgo.By("Waiting for pod with text data")
		gomega.Eventually(pollLogs1, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("value-1"))
		ginkgo.By("Waiting for pod with binary data")
		gomega.Eventually(pollLogs2, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("de ca fe ba d0 fe ff"))
	})

	/*
		Release: v1.9
		Testname: ConfigMap Volume, create, update and delete
		Description: The ConfigMap that is created MUST be accessible to read from the newly created Pod using the volume mount that is mapped to custom path in the Pod. When the config map is updated the change to the config map MUST be verified by reading the content from the mounted file in the Pod. Also when the item(file) is deleted from the map that MUST result in a error reading that item(file).
	*/
	framework.ConformanceIt("optional updates should be reflected in volume [NodeConformance]", func() {
		podLogTimeout := e2epod.GetPodSecretUpdateTimeout(f.ClientSet)
		containerTimeoutArg := fmt.Sprintf("--retry_time=%v", int(podLogTimeout.Seconds()))
		trueVal := true
		volumeMountPath := "/etc/configmap-volumes"

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

		ginkgo.By(fmt.Sprintf("Creating configMap with name %s", deleteConfigMap.Name))
		var err error
		if deleteConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), deleteConfigMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", deleteConfigMap.Name, err)
		}

		ginkgo.By(fmt.Sprintf("Creating configMap with name %s", updateConfigMap.Name))
		if updateConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), updateConfigMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", updateConfigMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-configmaps-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: deleteVolumeName,
						VolumeSource: v1.VolumeSource{
							ConfigMap: &v1.ConfigMapVolumeSource{
								LocalObjectReference: v1.LocalObjectReference{
									Name: deleteName,
								},
								Optional: &trueVal,
							},
						},
					},
					{
						Name: updateVolumeName,
						VolumeSource: v1.VolumeSource{
							ConfigMap: &v1.ConfigMapVolumeSource{
								LocalObjectReference: v1.LocalObjectReference{
									Name: updateName,
								},
								Optional: &trueVal,
							},
						},
					},
					{
						Name: createVolumeName,
						VolumeSource: v1.VolumeSource{
							ConfigMap: &v1.ConfigMapVolumeSource{
								LocalObjectReference: v1.LocalObjectReference{
									Name: createName,
								},
								Optional: &trueVal,
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:  deleteContainerName,
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/configmap-volumes/delete/data-1"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      deleteVolumeName,
								MountPath: path.Join(volumeMountPath, "delete"),
								ReadOnly:  true,
							},
						},
					},
					{
						Name:  updateContainerName,
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/configmap-volumes/update/data-3"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      updateVolumeName,
								MountPath: path.Join(volumeMountPath, "update"),
								ReadOnly:  true,
							},
						},
					},
					{
						Name:  createContainerName,
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/configmap-volumes/create/data-1"},
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
		ginkgo.By("Creating the pod")
		f.PodClient().CreateSync(pod)

		pollCreateLogs := func() (string, error) {
			return e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, createContainerName)
		}
		gomega.Eventually(pollCreateLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("Error reading file /etc/configmap-volumes/create/data-1"))

		pollUpdateLogs := func() (string, error) {
			return e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, updateContainerName)
		}
		gomega.Eventually(pollUpdateLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("Error reading file /etc/configmap-volumes/update/data-3"))

		pollDeleteLogs := func() (string, error) {
			return e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, deleteContainerName)
		}
		gomega.Eventually(pollDeleteLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("value-1"))

		ginkgo.By(fmt.Sprintf("Deleting configmap %v", deleteConfigMap.Name))
		err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), deleteConfigMap.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Failed to delete configmap %q in namespace %q", deleteConfigMap.Name, f.Namespace.Name)

		ginkgo.By(fmt.Sprintf("Updating configmap %v", updateConfigMap.Name))
		updateConfigMap.ResourceVersion = "" // to force update
		delete(updateConfigMap.Data, "data-1")
		updateConfigMap.Data["data-3"] = "value-3"
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(context.TODO(), updateConfigMap, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "Failed to update configmap %q in namespace %q", updateConfigMap.Name, f.Namespace.Name)

		ginkgo.By(fmt.Sprintf("Creating configMap with name %s", createConfigMap.Name))
		if createConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), createConfigMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", createConfigMap.Name, err)
		}

		ginkgo.By("waiting to observe update in volume")

		gomega.Eventually(pollCreateLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("value-1"))
		gomega.Eventually(pollUpdateLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("value-3"))
		gomega.Eventually(pollDeleteLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("Error reading file /etc/configmap-volumes/delete/data-1"))
	})

	/*
		Release: v1.9
		Testname: ConfigMap Volume, multiple volume maps
		Description: The ConfigMap that is created MUST be accessible to read from the newly created Pod using the volume mount that is mapped to multiple paths in the Pod. The content MUST be accessible from all the mapped volume mounts.
	*/
	framework.ConformanceIt("should be consumable in multiple volumes in the same pod [NodeConformance]", func() {
		var (
			name             = "configmap-test-volume-" + string(uuid.NewUUID())
			volumeName       = "configmap-volume"
			volumeMountPath  = "/etc/configmap-volume"
			volumeName2      = "configmap-volume-2"
			volumeMountPath2 = "/etc/configmap-volume-2"
			configMap        = newConfigMap(f, name)
		)

		ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
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
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"mounttest", "--file_content=/etc/configmap-volume/data-1"},
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

	// It should be forbidden to change data for configmaps marked as immutable, but
	// allowed to modify its metadata independently of its state.
	ginkgo.It("should be immutable if `immutable` field is set", func() {
		name := "immutable"
		configMap := newConfigMap(f, name)

		currentConfigMap, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create config map %q in namespace %q", configMap.Name, configMap.Namespace)

		currentConfigMap.Data["data-4"] = "value-4"
		currentConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(context.TODO(), currentConfigMap, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "Failed to update config map %q in namespace %q", configMap.Name, configMap.Namespace)

		// Mark config map as immutable.
		trueVal := true
		currentConfigMap.Immutable = &trueVal
		currentConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(context.TODO(), currentConfigMap, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "Failed to mark config map %q in namespace %q as immutable", configMap.Name, configMap.Namespace)

		// Ensure data can't be changed now.
		currentConfigMap.Data["data-5"] = "value-5"
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(context.TODO(), currentConfigMap, metav1.UpdateOptions{})
		framework.ExpectEqual(apierrors.IsInvalid(err), true)

		// Ensure config map can't be switched from immutable to mutable.
		currentConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Get(context.TODO(), name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to get config map %q in namespace %q", configMap.Name, configMap.Namespace)
		framework.ExpectEqual(*currentConfigMap.Immutable, true)

		falseVal := false
		currentConfigMap.Immutable = &falseVal
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(context.TODO(), currentConfigMap, metav1.UpdateOptions{})
		framework.ExpectEqual(apierrors.IsInvalid(err), true)

		// Ensure that metadata can be changed.
		currentConfigMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Get(context.TODO(), name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to get config map %q in namespace %q", configMap.Name, configMap.Namespace)
		currentConfigMap.Labels = map[string]string{"label1": "value1"}
		_, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Update(context.TODO(), currentConfigMap, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "Failed to update config map %q in namespace %q", configMap.Name, configMap.Namespace)

		// Ensure that immutable config map can be deleted.
		err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Failed to delete config map %q in namespace %q", configMap.Name, configMap.Namespace)
	})

	// The pod is in pending during volume creation until the configMap objects are available
	// or until mount the configMap volume times out. There is no configMap object defined for the pod, so it should return timout exception unless it is marked optional.
	// Slow (~5 mins)
	ginkgo.It("Should fail non-optional pod creation due to configMap object does not exist [Slow]", func() {
		volumeMountPath := "/etc/configmap-volumes"
		podName := "pod-configmaps-" + string(uuid.NewUUID())
		err := createNonOptionalConfigMapPod(f, volumeMountPath, podName)
		framework.ExpectError(err, "created pod %q with non-optional configMap in namespace %q", podName, f.Namespace.Name)
	})

	// ConfigMap object defined for the pod, If a key is specified which is not present in the ConfigMap,
	// the volume setup will error unless it is marked optional, during the pod creation.
	// Slow (~5 mins)
	ginkgo.It("Should fail non-optional pod creation due to the key in the configMap object does not exist [Slow]", func() {
		volumeMountPath := "/etc/configmap-volumes"
		podName := "pod-configmaps-" + string(uuid.NewUUID())
		err := createNonOptionalConfigMapPodWithConfig(f, volumeMountPath, podName)
		framework.ExpectError(err, "created pod %q with non-optional configMap in namespace %q", podName, f.Namespace.Name)
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

func doConfigMapE2EWithoutMappings(f *framework.Framework, asUser bool, fsGroup int64, defaultMode *int32) {
	groupID := int64(fsGroup)

	var (
		name            = "configmap-test-volume-" + string(uuid.NewUUID())
		volumeName      = "configmap-volume"
		volumeMountPath = "/etc/configmap-volume"
		configMap       = newConfigMap(f, name)
	)

	ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
	var err error
	if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}

	one := int64(1)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
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
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args: []string{
						"mounttest",
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
			RestartPolicy:                 v1.RestartPolicyNever,
			TerminationGracePeriodSeconds: &one,
		},
	}

	if asUser {
		setPodNonRootUser(pod)
	}

	if groupID != 0 {
		pod.Spec.SecurityContext.FSGroup = &groupID
	}

	if defaultMode != nil {
		pod.Spec.Volumes[0].VolumeSource.ConfigMap.DefaultMode = defaultMode
	}

	fileModeRegexp := getFileModeRegex("/etc/configmap-volume/data-1", defaultMode)
	output := []string{
		"content of file \"/etc/configmap-volume/data-1\": value-1",
		fileModeRegexp,
	}
	f.TestContainerOutputRegexp("consume configMaps", pod, 0, output)
}

func doConfigMapE2EWithMappings(f *framework.Framework, asUser bool, fsGroup int64, itemMode *int32) {
	groupID := int64(fsGroup)

	var (
		name            = "configmap-test-volume-map-" + string(uuid.NewUUID())
		volumeName      = "configmap-volume"
		volumeMountPath = "/etc/configmap-volume"
		configMap       = newConfigMap(f, name)
	)

	ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))

	var err error
	if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}

	one := int64(1)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
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
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args: []string{
						"mounttest",
						"--file_content=/etc/configmap-volume/path/to/data-2",
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
			RestartPolicy:                 v1.RestartPolicyNever,
			TerminationGracePeriodSeconds: &one,
		},
	}

	if asUser {
		setPodNonRootUser(pod)
	}

	if groupID != 0 {
		pod.Spec.SecurityContext.FSGroup = &groupID
	}

	if itemMode != nil {
		pod.Spec.Volumes[0].VolumeSource.ConfigMap.Items[0].Mode = itemMode
	}

	// Just check file mode if fsGroup is not set. If fsGroup is set, the
	// final mode is adjusted and we are not testing that case.
	output := []string{
		"content of file \"/etc/configmap-volume/path/to/data-2\": value-2",
	}
	if fsGroup == 0 {
		fileModeRegexp := getFileModeRegex("/etc/configmap-volume/path/to/data-2", itemMode)
		output = append(output, fileModeRegexp)
	}
	f.TestContainerOutputRegexp("consume configMaps", pod, 0, output)
}

func createNonOptionalConfigMapPod(f *framework.Framework, volumeMountPath, podName string) error {
	podLogTimeout := e2epod.GetPodSecretUpdateTimeout(f.ClientSet)
	containerTimeoutArg := fmt.Sprintf("--retry_time=%v", int(podLogTimeout.Seconds()))
	falseValue := false

	createName := "cm-test-opt-create-" + string(uuid.NewUUID())
	createContainerName := "createcm-volume-test"
	createVolumeName := "createcm-volume"

	// creating a pod without configMap object created, by mentioning the configMap volume source's local reference name
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: createVolumeName,
					VolumeSource: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: createName,
							},
							Optional: &falseValue,
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  createContainerName,
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/configmap-volumes/create/data-1"},
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
	ginkgo.By("Creating the pod")
	pod = f.PodClient().Create(pod)
	return e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
}

func createNonOptionalConfigMapPodWithConfig(f *framework.Framework, volumeMountPath, podName string) error {
	podLogTimeout := e2epod.GetPodSecretUpdateTimeout(f.ClientSet)
	containerTimeoutArg := fmt.Sprintf("--retry_time=%v", int(podLogTimeout.Seconds()))
	falseValue := false

	createName := "cm-test-opt-create-" + string(uuid.NewUUID())
	createContainerName := "createcm-volume-test"
	createVolumeName := "createcm-volume"
	configMap := newConfigMap(f, createName)

	ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
	var err error
	if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
	}
	// creating a pod with configMap object, but with different key which is not present in configMap object.
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: createVolumeName,
					VolumeSource: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: createName,
							},
							Items: []v1.KeyToPath{
								{
									Key:  "data-4",
									Path: "path/to/data-4",
								},
							},
							Optional: &falseValue,
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  createContainerName,
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/configmap-volumes/create/data-1"},
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
	ginkgo.By("Creating the pod")
	pod = f.PodClient().Create(pod)
	return e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
}
