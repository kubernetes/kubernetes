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

package node

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

// These tests exercise the Kubernetes expansion syntax $(VAR).
// For more information, see:
// https://github.com/kubernetes/community/blob/master/contributors/design-proposals/node/expansion.md
var _ = SIGDescribe("Variable Expansion", func() {
	f := framework.NewDefaultFramework("var-expansion")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	/*
		Release: v1.9
		Testname: Environment variables, expansion
		Description: Create a Pod with environment variables. Environment variables defined using previously defined environment variables MUST expand to proper values.
	*/
	framework.ConformanceIt("should allow composing env vars into new env vars [NodeConformance]", func() {
		envVars := []v1.EnvVar{
			{
				Name:  "FOO",
				Value: "foo-value",
			},
			{
				Name:  "BAR",
				Value: "bar-value",
			},
			{
				Name:  "FOOBAR",
				Value: "$(FOO);;$(BAR)",
			},
		}
		pod := newPod([]string{"sh", "-c", "env"}, envVars, nil, nil)

		f.TestContainerOutput("env composition", pod, 0, []string{
			"FOO=foo-value",
			"BAR=bar-value",
			"FOOBAR=foo-value;;bar-value",
		})
	})

	/*
		Release: v1.9
		Testname: Environment variables, command expansion
		Description: Create a Pod with environment variables and container command using them. Container command using the  defined environment variables MUST expand to proper values.
	*/
	framework.ConformanceIt("should allow substituting values in a container's command [NodeConformance]", func() {
		envVars := []v1.EnvVar{
			{
				Name:  "TEST_VAR",
				Value: "test-value",
			},
		}
		pod := newPod([]string{"sh", "-c", "TEST_VAR=wrong echo \"$(TEST_VAR)\""}, envVars, nil, nil)

		f.TestContainerOutput("substitution in container's command", pod, 0, []string{
			"test-value",
		})
	})

	/*
		Release: v1.9
		Testname: Environment variables, command argument expansion
		Description: Create a Pod with environment variables and container command arguments using them. Container command arguments using the  defined environment variables MUST expand to proper values.
	*/
	framework.ConformanceIt("should allow substituting values in a container's args [NodeConformance]", func() {
		envVars := []v1.EnvVar{
			{
				Name:  "TEST_VAR",
				Value: "test-value",
			},
		}
		pod := newPod([]string{"sh", "-c"}, envVars, nil, nil)
		pod.Spec.Containers[0].Args = []string{"TEST_VAR=wrong echo \"$(TEST_VAR)\""}

		f.TestContainerOutput("substitution in container's args", pod, 0, []string{
			"test-value",
		})
	})

	/*
		Release: v1.19
		Testname: VolumeSubpathEnvExpansion, subpath expansion
		Description: Make sure a container's subpath can be set using an expansion of environment variables.
	*/
	framework.ConformanceIt("should allow substituting values in a volume subpath", func() {
		envVars := []v1.EnvVar{
			{
				Name:  "POD_NAME",
				Value: "foo",
			},
		}
		mounts := []v1.VolumeMount{
			{
				Name:        "workdir1",
				MountPath:   "/logscontainer",
				SubPathExpr: "$(POD_NAME)",
			},
			{
				Name:      "workdir1",
				MountPath: "/testcontainer",
			},
		}
		volumes := []v1.Volume{
			{
				Name: "workdir1",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			},
		}
		pod := newPod([]string{}, envVars, mounts, volumes)
		envVars[0].Value = pod.ObjectMeta.Name
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "test -d /testcontainer/" + pod.ObjectMeta.Name + ";echo $?"}

		f.TestContainerOutput("substitution in volume subpath", pod, 0, []string{
			"0",
		})
	})

	/*
		Release: v1.19
		Testname: VolumeSubpathEnvExpansion, subpath with backticks
		Description: Make sure a container's subpath can not be set using an expansion of environment variables when backticks are supplied.
	*/
	framework.ConformanceIt("should fail substituting values in a volume subpath with backticks [Slow]", func() {

		envVars := []v1.EnvVar{
			{
				Name:  "POD_NAME",
				Value: "..",
			},
		}
		mounts := []v1.VolumeMount{
			{
				Name:        "workdir1",
				MountPath:   "/logscontainer",
				SubPathExpr: "$(POD_NAME)",
			},
		}
		volumes := []v1.Volume{
			{
				Name: "workdir1",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			},
		}
		pod := newPod(nil, envVars, mounts, volumes)

		// Pod should fail
		testPodFailSubpath(f, pod)
	})

	/*
		Release: v1.19
		Testname: VolumeSubpathEnvExpansion, subpath with absolute path
		Description: Make sure a container's subpath can not be set using an expansion of environment variables when absolute path is supplied.
	*/
	framework.ConformanceIt("should fail substituting values in a volume subpath with absolute path [Slow]", func() {
		absolutePath := "/tmp"
		if framework.NodeOSDistroIs("windows") {
			// Windows does not typically have a C:\tmp folder.
			absolutePath = "C:\\Users"
		}

		envVars := []v1.EnvVar{
			{
				Name:  "POD_NAME",
				Value: absolutePath,
			},
		}
		mounts := []v1.VolumeMount{
			{
				Name:        "workdir1",
				MountPath:   "/logscontainer",
				SubPathExpr: "$(POD_NAME)",
			},
		}
		volumes := []v1.Volume{
			{
				Name: "workdir1",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			},
		}
		pod := newPod(nil, envVars, mounts, volumes)

		// Pod should fail
		testPodFailSubpath(f, pod)
	})

	/*
		Release: v1.19
		Testname: VolumeSubpathEnvExpansion, subpath ready from failed state
		Description: Verify that a failing subpath expansion can be modified during the lifecycle of a container.
	*/
	framework.ConformanceIt("should verify that a failing subpath expansion can be modified during the lifecycle of a container [Slow]", func() {

		envVars := []v1.EnvVar{
			{
				Name:  "POD_NAME",
				Value: "foo",
			},
			{
				Name: "ANNOTATION",
				ValueFrom: &v1.EnvVarSource{
					FieldRef: &v1.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.annotations['mysubpath']",
					},
				},
			},
		}
		mounts := []v1.VolumeMount{
			{
				Name:        "workdir1",
				MountPath:   "/subpath_mount",
				SubPathExpr: "$(ANNOTATION)/$(POD_NAME)",
			},
			{
				Name:      "workdir1",
				MountPath: "/volume_mount",
			},
		}
		volumes := []v1.Volume{
			{
				Name: "workdir1",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			},
		}
		pod := newPod([]string{"sh", "-c", "tail -f /dev/null"}, envVars, mounts, volumes)
		pod.ObjectMeta.Annotations = map[string]string{"notmysubpath": "mypath"}

		ginkgo.By("creating the pod with failed condition")
		var podClient *framework.PodClient = f.PodClient()
		pod = podClient.Create(pod)

		err := e2epod.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, pod.Namespace, framework.PodStartShortTimeout)
		framework.ExpectError(err, "while waiting for pod to be running")

		ginkgo.By("updating the pod")
		podClient.Update(pod.ObjectMeta.Name, func(pod *v1.Pod) {
			if pod.ObjectMeta.Annotations == nil {
				pod.ObjectMeta.Annotations = make(map[string]string)
			}
			pod.ObjectMeta.Annotations["mysubpath"] = "mypath"
		})

		ginkgo.By("waiting for pod running")
		err = e2epod.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, pod.Namespace, framework.PodStartShortTimeout)
		framework.ExpectNoError(err, "while waiting for pod to be running")

		ginkgo.By("deleting the pod gracefully")
		err = e2epod.DeletePodWithWait(f.ClientSet, pod)
		framework.ExpectNoError(err, "failed to delete pod")
	})

	/*
		Release: v1.19
		Testname: VolumeSubpathEnvExpansion, subpath test writes
		Description: Verify that a subpath expansion can be used to write files into subpaths.
		1.	valid subpathexpr starts a container running
		2.	test for valid subpath writes
		3.	successful expansion of the subpathexpr isn't required for volume cleanup

	*/
	framework.ConformanceIt("should succeed in writing subpaths in container [Slow]", func() {

		envVars := []v1.EnvVar{
			{
				Name:  "POD_NAME",
				Value: "foo",
			},
			{
				Name: "ANNOTATION",
				ValueFrom: &v1.EnvVarSource{
					FieldRef: &v1.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.annotations['mysubpath']",
					},
				},
			},
		}
		mounts := []v1.VolumeMount{
			{
				Name:        "workdir1",
				MountPath:   "/subpath_mount",
				SubPathExpr: "$(ANNOTATION)/$(POD_NAME)",
			},
			{
				Name:      "workdir1",
				MountPath: "/volume_mount",
			},
		}
		volumes := []v1.Volume{
			{
				Name: "workdir1",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			},
		}
		pod := newPod([]string{"sh", "-c", "tail -f /dev/null"}, envVars, mounts, volumes)
		pod.ObjectMeta.Annotations = map[string]string{"mysubpath": "mypath"}

		ginkgo.By("creating the pod")
		var podClient *framework.PodClient = f.PodClient()
		pod = podClient.Create(pod)

		ginkgo.By("waiting for pod running")
		err := e2epod.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, pod.Namespace, framework.PodStartShortTimeout)
		framework.ExpectNoError(err, "while waiting for pod to be running")

		ginkgo.By("creating a file in subpath")
		cmd := "touch /volume_mount/mypath/foo/test.log"
		_, _, err = f.ExecShellInPodWithFullOutput(pod.Name, cmd)
		if err != nil {
			framework.Failf("expected to be able to write to subpath")
		}

		ginkgo.By("test for file in mounted path")
		cmd = "test -f /subpath_mount/test.log"
		_, _, err = f.ExecShellInPodWithFullOutput(pod.Name, cmd)
		if err != nil {
			framework.Failf("expected to be able to verify file")
		}

		ginkgo.By("updating the annotation value")
		podClient.Update(pod.ObjectMeta.Name, func(pod *v1.Pod) {
			pod.ObjectMeta.Annotations["mysubpath"] = "mynewpath"
		})

		ginkgo.By("waiting for annotated pod running")
		err = e2epod.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, pod.Namespace, framework.PodStartShortTimeout)
		framework.ExpectNoError(err, "while waiting for annotated pod to be running")

		ginkgo.By("deleting the pod gracefully")
		err = e2epod.DeletePodWithWait(f.ClientSet, pod)
		framework.ExpectNoError(err, "failed to delete pod")
	})
})

func testPodFailSubpath(f *framework.Framework, pod *v1.Pod) {
	var podClient *framework.PodClient = f.PodClient()
	pod = podClient.Create(pod)

	defer func() {
		e2epod.DeletePodWithWait(f.ClientSet, pod)
	}()

	err := e2epod.WaitForPodContainerToFail(f.ClientSet, pod.Namespace, pod.Name, 0, "CreateContainerConfigError", framework.PodStartShortTimeout)
	framework.ExpectNoError(err, "while waiting for the pod container to fail")
}

func newPod(command []string, envVars []v1.EnvVar, mounts []v1.VolumeMount, volumes []v1.Volume) *v1.Pod {
	podName := "var-expansion-" + string(uuid.NewUUID())
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   podName,
			Labels: map[string]string{"name": podName},
		},
		Spec: v1.PodSpec{
			Containers:    []v1.Container{newContainer("dapi-container", command, envVars, mounts)},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes:       volumes,
		},
	}
}

func newContainer(containerName string, command []string, envVars []v1.EnvVar, mounts []v1.VolumeMount) v1.Container {
	return v1.Container{
		Name:         containerName,
		Image:        imageutils.GetE2EImage(imageutils.BusyBox),
		Command:      command,
		Env:          envVars,
		VolumeMounts: mounts,
	}
}
