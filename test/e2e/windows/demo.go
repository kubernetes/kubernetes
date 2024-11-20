/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"os"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
)

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

func createConfigMapVolumeMounttestPod(namespace, volumeName, referenceName, mountPath string, mounttestArgs ...string) *v1.Pod {
	volumes := []v1.Volume{
		{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{
						Name: referenceName,
					},
				},
			},
		},
	}
	podName := "pod-configmaps-" + string(uuid.NewUUID())
	mounttestArgs = append([]string{"mounttest"}, mounttestArgs...)
	pod := e2epod.NewAgnhostPod(namespace, podName, volumes, createMounts(volumeName, mountPath, true), nil, mounttestArgs...)
	pod.Spec.RestartPolicy = v1.RestartPolicyNever
	return pod
}

// getFileModeRegex returns a file mode related regex which should be matched by the mounttest pods' output.
// If the given mask is nil, then the regex will contain the default OS file modes, which are 0644 for Linux and 0775 for Windows.
func getFileModeRegex(filePath string, mask *int32) string {
	var (
		linuxMask   int32
		windowsMask int32
	)
	if mask == nil {
		linuxMask = int32(0644)
		windowsMask = int32(0775)
	} else {
		linuxMask = *mask
		windowsMask = *mask
	}

	linuxOutput := fmt.Sprintf("mode of file \"%s\": %v", filePath, os.FileMode(linuxMask))
	windowsOutput := fmt.Sprintf("mode of Windows file \"%v\": %s", filePath, os.FileMode(windowsMask))

	return fmt.Sprintf("(%s|%s)", linuxOutput, windowsOutput)
}

// createMounts creates a v1.VolumeMount list with a single element.
func createMounts(volumeName, volumeMountPath string, readOnly bool) []v1.VolumeMount {
	return []v1.VolumeMount{
		{
			Name:      volumeName,
			MountPath: volumeMountPath,
			ReadOnly:  readOnly,
		},
	}
}

var _ = sigDescribe(feature.Windows, "Windows tests", skipUnlessWindows(func() {
	f := framework.NewDefaultFramework("windows-tests")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	})

	ginkgo.Context("check file permissions", func() {

		ginkgo.It("container should have right permissions", func(ctx context.Context) {

			var (
				name            = "configmap-test-volume-map-" + string(uuid.NewUUID())
				volumeName      = "configmap-volume"
				volumeMountPath = "/etc/configmap-volume"
				configMap       = newConfigMap(f, name)
			)

			ginkgo.By(fmt.Sprintf("Creating configMap with name %s", configMap.Name))

			var err error
			if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, configMap, metav1.CreateOptions{}); err != nil {
				framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
			}

			pod1 := createConfigMapVolumeMounttestPod(f.Namespace.Name, volumeName, "name", volumeMountPath,
				"--file_content=/etc/configmap-volume/path/to/data-2", "--file_mode=/etc/configmap-volume/path/to/data-2")
			one1 := int64(1)
			pod1.Spec.TerminationGracePeriodSeconds = &one1
			pod1.Spec.Volumes[0].VolumeSource.ConfigMap.Items = []v1.KeyToPath{
				{
					Key:  "data-2",
					Path: "path/to/data-2",
				},
			}
			pod1.Spec.Containers[0].Command = []string{
				"powershell.exe",
				"Get-Acl",
				"-Path",
				"/etc/configmap-volume/path/to/data-2",
				"|",
				"Format-List",
			}

			_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod1, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, "name", f.Namespace.Name)
			framework.ExpectNoError(err)
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod1.Name, pod1.Spec.Containers[0].Name)
			framework.ExpectNoError(err)
			framework.Logf("Pod logs: \n%s", logs)

			pod2 := createConfigMapVolumeMounttestPod(f.Namespace.Name, volumeName, name, volumeMountPath,
				"--file_content=/etc/configmap-volume/path/to/data-2", "--file_mode=/etc/configmap-volume/path/to/data-2")
			one2 := int64(1)
			pod2.Spec.TerminationGracePeriodSeconds = &one2
			pod2.Spec.Volumes[0].VolumeSource.ConfigMap.Items = []v1.KeyToPath{
				{
					Key:  "data-2",
					Path: "path/to/data-2",
				},
			}

			output := []string{
				"content of file \"/etc/projected-configmap-volume/path/to/data-2\": value-2",
			}
			fileModeRegexp := getFileModeRegex("/etc/projected-configmap-volume/path/to/data-2", nil)
			output = append(output, fileModeRegexp)
			e2epodoutput.TestContainerOutputRegexp(ctx, f, "consume configMaps", pod2, 0, output)
		})
	})
}))
