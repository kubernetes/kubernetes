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

package windows

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

// This test verifies that pods with multiple subPath volume mounts on Windows
// nodes terminate cleanly. It validates the fix for
// https://github.com/kubernetes/kubernetes/issues/112630 where leaked file
// handles from subPath preparation caused emptyDir volumes to fail cleanup,
// leaving pods stuck in Terminating state.
var _ = sigDescribe(feature.Windows, "Windows subpath cleanup", skipUnlessWindows(func() {
	f := framework.NewDefaultFramework("windows-subpath-cleanup")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("pod with multiple subPath mounts should terminate cleanly", func(ctx context.Context) {
		podName := "subpath-cleanup-" + string(uuid.NewUUID())
		configMapName := "subpath-cm-" + string(uuid.NewUUID())

		ginkgo.By("creating a ConfigMap with multiple keys")
		configMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      configMapName,
				Namespace: f.Namespace.Name,
			},
			Data: map[string]string{
				"key1": "value1",
				"key2": "value2",
				"key3": "value3",
			},
		}
		_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, configMap, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating ConfigMap")

		ginkgo.By("creating a Secret with multiple keys")
		secretName := "subpath-secret-" + string(uuid.NewUUID())
		secret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      secretName,
				Namespace: f.Namespace.Name,
			},
			StringData: map[string]string{
				"secret1": "secretvalue1",
				"secret2": "secretvalue2",
			},
		}
		_, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(ctx, secret, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating Secret")

		ginkgo.By("creating a pod with multiple volume types and subPath mounts")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				NodeSelector: map[string]string{
					"kubernetes.io/os": "windows",
				},
				Containers: []v1.Container{
					{
						Name:  "test-container",
						Image: imageutils.GetE2EImage(imageutils.Pause),
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config-volume",
								MountPath: `C:\etc\config\key1`,
								SubPath:   "key1",
							},
							{
								Name:      "config-volume",
								MountPath: `C:\etc\config\key2`,
								SubPath:   "key2",
							},
							{
								Name:      "config-volume",
								MountPath: `C:\etc\config\key3`,
								SubPath:   "key3",
							},
							// Nested HostPath mounts: each is inside the previous,
							// testing cleanup with overlapping mount points. Note that these are intentionally not
							// in the right order to test a mount that is containing an existing mount.
							{
								Name:      "hostpath-outer",
								MountPath: `C:\data`,
							},
							{
								Name:      "hostpath-inner",
								MountPath: `C:\data\nested\deep`,
							},
							{
								Name:      "hostpath-middle",
								MountPath: `C:\data\nested`,
							},
							// EmptyDir mounts
							{
								Name:      "emptydir-1",
								MountPath: `C:\scratch\dir1`,
							},
							{
								Name:      "emptydir-2",
								MountPath: `C:\scratch\dir2`,
							},
							// EmptyDir memory-backed mounts
							{
								Name:      "emptydir-memory-1",
								MountPath: `C:\scratch\mem1`,
							},
							{
								Name:      "emptydir-memory-2",
								MountPath: `C:\scratch\mem2`,
							},
							// Secret mounts with subPath
							{
								Name:      "secret-volume",
								MountPath: `C:\etc\secret\secret1`,
								SubPath:   "secret1",
							},
							{
								Name:      "secret-volume",
								MountPath: `C:\etc\secret\secret2`,
								SubPath:   "secret2",
							},
							// Projected volume mounts with subPath
							{
								Name:      "projected-volume",
								MountPath: `C:\etc\projected\key1`,
								SubPath:   "key1",
							},
							{
								Name:      "projected-volume",
								MountPath: `C:\etc\projected\secret1`,
								SubPath:   "secret1",
							},
							// DownwardAPI mounts with subPath
							{
								Name:      "downwardapi-volume",
								MountPath: `C:\etc\podinfo\labels`,
								SubPath:   "labels",
							},
							{
								Name:      "downwardapi-volume",
								MountPath: `C:\etc\podinfo\name`,
								SubPath:   "name",
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config-volume",
						VolumeSource: v1.VolumeSource{
							ConfigMap: &v1.ConfigMapVolumeSource{
								LocalObjectReference: v1.LocalObjectReference{
									Name: configMapName,
								},
							},
						},
					},
					{
						Name: "hostpath-outer",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: `C:\var\hostpath\outer`,
							},
						},
					},
					{
						Name: "hostpath-middle",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: `C:\var\hostpath\middle`,
							},
						},
					},
					{
						Name: "hostpath-inner",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: `C:\var\hostpath\inner`,
							},
						},
					},
					{
						Name: "emptydir-1",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
					{
						Name: "emptydir-2",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
					{
						Name: "emptydir-memory-1",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{
								Medium: v1.StorageMediumMemory,
							},
						},
					},
					{
						Name: "emptydir-memory-2",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{
								Medium: v1.StorageMediumMemory,
							},
						},
					},
					{
						Name: "secret-volume",
						VolumeSource: v1.VolumeSource{
							Secret: &v1.SecretVolumeSource{
								SecretName: secretName,
							},
						},
					},
					{
						Name: "projected-volume",
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										ConfigMap: &v1.ConfigMapProjection{
											LocalObjectReference: v1.LocalObjectReference{
												Name: configMapName,
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
								},
							},
						},
					},
					{
						Name: "downwardapi-volume",
						VolumeSource: v1.VolumeSource{
							DownwardAPI: &v1.DownwardAPIVolumeSource{
								Items: []v1.DownwardAPIVolumeFile{
									{
										Path: "labels",
										FieldRef: &v1.ObjectFieldSelector{
											FieldPath: "metadata.labels",
										},
									},
									{
										Path: "name",
										FieldRef: &v1.ObjectFieldSelector{
											FieldPath: "metadata.name",
										},
									},
								},
							},
						},
					},
				},
			},
		}

		pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

		ginkgo.By("deleting the pod and verifying it terminates within a reasonable time")
		err = e2epod.DeletePodWithGracePeriod(ctx, f.ClientSet, pod, 30)
		framework.ExpectNoError(err, "deleting pod")

		// Prior to addressing https://github.com/kubernetes/kubernetes/issues/112630, this would time out
		err = e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, podName, f.Namespace.Name, 5*time.Minute)
		framework.ExpectNoError(err, "waiting for pod to be fully deleted")
	})
}))
