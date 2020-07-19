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
	"context"
	"fmt"
	"path"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = ginkgo.Describe("[sig-storage] Projected configMap", func() {
	f := framework.NewDefaultFramework("projected")

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, volume mode default
	   Description: A Pod is created with projected volume source 'ConfigMap' to store a configMap with default permission mode. Pod MUST be able to read the content of the ConfigMap successfully and the mode on the volume MUST be -rw-r--r--.
	*/
	framework.ConformanceIt("should be consumable from pods in volume [NodeConformance]", func() {
		doProjectedConfigMapE2EWithoutMappings(f, false, 0, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, volume mode 0400
	   Description: A Pod is created with projected volume source 'ConfigMap' to store a configMap with permission mode set to 0400. Pod MUST be able to read the content of the ConfigMap successfully and the mode on the volume MUST be -r--------.
	   This test is marked LinuxOnly since Windows does not support setting specific file permissions.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with defaultMode set [LinuxOnly] [NodeConformance]", func() {
		defaultMode := int32(0400)
		doProjectedConfigMapE2EWithoutMappings(f, false, 0, &defaultMode)
	})

	ginkgo.It("should be consumable from pods in volume as non-root with defaultMode and fsGroup set [LinuxOnly] [NodeFeature:FSGroup]", func() {
		// Windows does not support RunAsUser / FSGroup SecurityContext options, and it does not support setting file permissions.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
		defaultMode := int32(0440) /* setting fsGroup sets mode to at least 440 */
		doProjectedConfigMapE2EWithoutMappings(f, true, 1001, &defaultMode)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, non-root user
	   Description: A Pod is created with projected volume source 'ConfigMap' to store a configMap as non-root user with uid 1000. Pod MUST be able to read the content of the ConfigMap successfully and the mode on the volume MUST be -rw-r--r--.
	*/
	framework.ConformanceIt("should be consumable from pods in volume as non-root [NodeConformance]", func() {
		doProjectedConfigMapE2EWithoutMappings(f, true, 0, nil)
	})

	ginkgo.It("should be consumable from pods in volume as non-root with FSGroup [LinuxOnly] [NodeFeature:FSGroup]", func() {
		// Windows does not support RunAsUser / FSGroup SecurityContext options.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
		doProjectedConfigMapE2EWithoutMappings(f, true, 1001, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, mapped
	   Description: A Pod is created with projected volume source 'ConfigMap' to store a configMap with default permission mode. The ConfigMap is also mapped to a custom path. Pod MUST be able to read the content of the ConfigMap from the custom location successfully and the mode on the volume MUST be -rw-r--r--.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings [NodeConformance]", func() {
		doProjectedConfigMapE2EWithMappings(f, false, 0, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, mapped, volume mode 0400
	   Description: A Pod is created with projected volume source 'ConfigMap' to store a configMap with permission mode set to 0400. The ConfigMap is also mapped to a custom path. Pod MUST be able to read the content of the ConfigMap from the custom location successfully and the mode on the volume MUST be -r--r--r--.
	   This test is marked LinuxOnly since Windows does not support setting specific file permissions.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings and Item mode set [LinuxOnly] [NodeConformance]", func() {
		mode := int32(0400)
		doProjectedConfigMapE2EWithMappings(f, false, 0, &mode)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, mapped, non-root user
	   Description: A Pod is created with projected volume source 'ConfigMap' to store a configMap as non-root user with uid 1000. The ConfigMap is also mapped to a custom path. Pod MUST be able to read the content of the ConfigMap from the custom location successfully and the mode on the volume MUST be -r--r--r--.
	*/
	framework.ConformanceIt("should be consumable from pods in volume with mappings as non-root [NodeConformance]", func() {
		doProjectedConfigMapE2EWithMappings(f, true, 0, nil)
	})

	ginkgo.It("should be consumable from pods in volume with mappings as non-root with FSGroup [LinuxOnly] [NodeFeature:FSGroup]", func() {
		// Windows does not support RunAsUser / FSGroup SecurityContext options.
		e2eskipper.SkipIfNodeOSDistroIs("windows")
		doProjectedConfigMapE2EWithMappings(f, true, 1001, nil)
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, update
	   Description: A Pod is created with projected volume source 'ConfigMap' to store a configMap and performs a create and update to new value. Pod MUST be able to create the configMap with value-1. Pod MUST be able to update the value in the confgiMap to value-2.
	*/
	framework.ConformanceIt("updates should be reflected in volume [NodeConformance]", func() {
		podLogTimeout := e2epod.GetPodSecretUpdateTimeout(f.ClientSet)
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

		ginkgo.By(fmt.Sprintf("Creating projection with configMap that has name %s", configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
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
						Name:  containerName,
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-configmap-volume/data-1"},
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
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, create, update and delete
	   Description: Create a Pod with three containers with ConfigMaps namely a create, update and delete container. Create Container when started MUST not have configMap, update and delete containers MUST be created with a ConfigMap value as 'value-1'. Create a configMap in the create container, the Pod MUST be able to read the configMap from the create container. Update the configMap in the update container, Pod MUST be able to read the updated configMap value. Delete the configMap in the delete container. Pod MUST fail to read the configMap from the delete container.
	*/
	framework.ConformanceIt("optional updates should be reflected in volume [NodeConformance]", func() {
		podLogTimeout := e2epod.GetPodSecretUpdateTimeout(f.ClientSet)
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
						Name:  deleteContainerName,
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-configmap-volumes/delete/data-1"},
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
						Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-configmap-volumes/update/data-3"},
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
						Args:  []string{"mounttest", "--break_on_expected_content=false", containerTimeoutArg, "--file_content_in_loop=/etc/projected-configmap-volumes/create/data-1"},
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
		gomega.Eventually(pollCreateLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("Error reading file /etc/projected-configmap-volumes/create/data-1"))

		pollUpdateLogs := func() (string, error) {
			return e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, updateContainerName)
		}
		gomega.Eventually(pollUpdateLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("Error reading file /etc/projected-configmap-volumes/update/data-3"))

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
		gomega.Eventually(pollDeleteLogs, podLogTimeout, framework.Poll).Should(gomega.ContainSubstring("Error reading file /etc/projected-configmap-volumes/delete/data-1"))
	})

	/*
	   Release : v1.9
	   Testname: Projected Volume, ConfigMap, multiple volume paths
	   Description: A Pod is created with a projected volume source 'ConfigMap' to store a configMap. The configMap is mapped to two different volume mounts. Pod MUST be able to read the content of the configMap successfully from the two volume mounts.
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

		ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
		var err error
		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
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
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"mounttest", "--file_content=/etc/projected-configmap-volume/data-1"},
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

	//The pod is in pending during volume creation until the configMap objects are available
	//or until mount the configMap volume times out. There is no configMap object defined for the pod, so it should return timout exception unless it is marked optional.
	//Slow (~5 mins)
	ginkgo.It("Should fail non-optional pod creation due to configMap object does not exist [Slow]", func() {
		volumeMountPath := "/etc/projected-configmap-volumes"
		podName := "pod-projected-configmaps-" + string(uuid.NewUUID())
		err := createNonOptionalConfigMapPod(f, volumeMountPath, podName)
		framework.ExpectError(err, "created pod %q with non-optional configMap in namespace %q", podName, f.Namespace.Name)
	})

	//ConfigMap object defined for the pod, If a key is specified which is not present in the ConfigMap,
	// the volume setup will error unless it is marked optional, during the pod creation.
	//Slow (~5 mins)
	ginkgo.It("Should fail non-optional pod creation due to the key in the configMap object does not exist [Slow]", func() {
		volumeMountPath := "/etc/configmap-volumes"
		podName := "pod-configmaps-" + string(uuid.NewUUID())
		err := createNonOptionalConfigMapPodWithConfig(f, volumeMountPath, podName)
		framework.ExpectError(err, "created pod %q with non-optional configMap in namespace %q", podName, f.Namespace.Name)
	})
})

func doProjectedConfigMapE2EWithoutMappings(f *framework.Framework, asUser bool, fsGroup int64, defaultMode *int32) {
	groupID := int64(fsGroup)

	var (
		name            = "projected-configmap-test-volume-" + string(uuid.NewUUID())
		volumeName      = "projected-configmap-volume"
		volumeMountPath = "/etc/projected-configmap-volume"
		configMap       = newConfigMap(f, name)
	)

	ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))
	var err error
	if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
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
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args: []string{
						"mounttest",
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

	if asUser {
		setPodNonRootUser(pod)
	}

	if groupID != 0 {
		pod.Spec.SecurityContext.FSGroup = &groupID
	}

	if defaultMode != nil {
		//pod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].ConfigMap.DefaultMode = defaultMode
		pod.Spec.Volumes[0].VolumeSource.Projected.DefaultMode = defaultMode
	}

	fileModeRegexp := getFileModeRegex("/etc/projected-configmap-volume/data-1", defaultMode)
	output := []string{
		"content of file \"/etc/projected-configmap-volume/data-1\": value-1",
		fileModeRegexp,
	}
	f.TestContainerOutputRegexp("consume configMaps", pod, 0, output)
}

func doProjectedConfigMapE2EWithMappings(f *framework.Framework, asUser bool, fsGroup int64, itemMode *int32) {
	groupID := int64(fsGroup)

	var (
		name            = "projected-configmap-test-volume-map-" + string(uuid.NewUUID())
		volumeName      = "projected-configmap-volume"
		volumeMountPath = "/etc/projected-configmap-volume"
		configMap       = newConfigMap(f, name)
	)

	ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))

	var err error
	if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
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
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args: []string{
						"mounttest",
						"--file_content=/etc/projected-configmap-volume/path/to/data-2",
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

	if asUser {
		setPodNonRootUser(pod)
	}

	if groupID != 0 {
		pod.Spec.SecurityContext.FSGroup = &groupID
	}

	if itemMode != nil {
		//pod.Spec.Volumes[0].VolumeSource.ConfigMap.Items[0].Mode = itemMode
		pod.Spec.Volumes[0].VolumeSource.Projected.DefaultMode = itemMode
	}

	// Just check file mode if fsGroup is not set. If fsGroup is set, the
	// final mode is adjusted and we are not testing that case.
	output := []string{
		"content of file \"/etc/projected-configmap-volume/path/to/data-2\": value-2",
	}
	if fsGroup == 0 {
		fileModeRegexp := getFileModeRegex("/etc/projected-configmap-volume/path/to/data-2", itemMode)
		output = append(output, fileModeRegexp)
	}
	f.TestContainerOutputRegexp("consume configMaps", pod, 0, output)
}
