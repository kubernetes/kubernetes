/*
Copyright 2017 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	appArmorProfilePrefix = "e2e-apparmor-test-"
	appArmorAllowedPath   = "/expect_allowed_write"
	appArmorDeniedPath    = "/expect_permission_denied"

	loaderLabelKey   = "name"
	loaderLabelValue = "e2e-apparmor-loader"
)

// AppArmorDistros are distros with AppArmor support
var AppArmorDistros = []string{"gci", "ubuntu"}

func IsAppArmorSupported() bool {
	return framework.NodeOSDistroIs(AppArmorDistros...)
}

func SkipIfAppArmorNotSupported() {
	framework.SkipUnlessNodeOSDistroIs(AppArmorDistros...)
}

func LoadAppArmorProfiles(f *framework.Framework) {
	createAppArmorProfileCM(f)
	createAppArmorProfileLoader(f)
}

// CreateAppArmorTestPod creates a pod that tests apparmor profile enforcement. The pod exits with
// an error code if the profile is incorrectly enforced. If runOnce is true the pod will exit after
// a single test, otherwise it will repeat the test every 1 second until failure.
func CreateAppArmorTestPod(f *framework.Framework, unconfined bool, runOnce bool) *v1.Pod {
	profile := "localhost/" + appArmorProfilePrefix + f.Namespace.Name
	testCmd := fmt.Sprintf(`
if touch %[1]s; then
  echo "FAILURE: write to %[1]s should be denied"
  exit 1
elif ! touch %[2]s; then
  echo "FAILURE: write to %[2]s should be allowed"
  exit 2
elif [[ $(< /proc/self/attr/current) != "%[3]s" ]]; then
  echo "FAILURE: not running with expected profile %[3]s"
  echo "found: $(cat /proc/self/attr/current)"
  exit 3
fi`, appArmorDeniedPath, appArmorAllowedPath, appArmorProfilePrefix+f.Namespace.Name)

	if unconfined {
		profile = apparmor.ProfileNameUnconfined
		testCmd = `
if cat /proc/sysrq-trigger 2>&1 | grep 'Permission denied'; then
  echo 'FAILURE: reading /proc/sysrq-trigger should be allowed'
  exit 1
elif [[ $(< /proc/self/attr/current) != "unconfined" ]]; then
  echo 'FAILURE: not running with expected profile unconfined'
  exit 2
fi`
	}

	if !runOnce {
		testCmd = fmt.Sprintf(`while true; do
%s
sleep 1
done`, testCmd)
	}

	loaderAffinity := &v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{{
				Namespaces: []string{f.Namespace.Name},
				LabelSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{loaderLabelKey: loaderLabelValue},
				},
				TopologyKey: "kubernetes.io/hostname",
			}},
		},
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-apparmor-",
			Annotations: map[string]string{
				apparmor.ContainerAnnotationKeyPrefix + "test": profile,
			},
			Labels: map[string]string{
				"test": "apparmor",
			},
		},
		Spec: v1.PodSpec{
			Affinity: loaderAffinity,
			Containers: []v1.Container{{
				Name:    "test",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"sh", "-c", testCmd},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	if runOnce {
		pod = f.PodClient().Create(pod)
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(
			f.ClientSet, pod.Name, f.Namespace.Name))
		var err error
		pod, err = f.PodClient().Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
	} else {
		pod = f.PodClient().CreateSync(pod)
		framework.ExpectNoError(f.WaitForPodReady(pod.Name))
	}

	// Verify Pod affinity colocated the Pods.
	loader := getRunningLoaderPod(f)
	framework.ExpectEqual(pod.Spec.NodeName, loader.Spec.NodeName)

	return pod
}

func createAppArmorProfileCM(f *framework.Framework) {
	profileName := appArmorProfilePrefix + f.Namespace.Name
	profile := fmt.Sprintf(`#include <tunables/global>
profile %s flags=(attach_disconnected) {
  #include <abstractions/base>

  file,

  deny %s w,
  audit %s w,
}
`, profileName, appArmorDeniedPath, appArmorAllowedPath)

	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "apparmor-profiles",
			Namespace: f.Namespace.Name,
		},
		Data: map[string]string{
			profileName: profile,
		},
	}
	_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(cm)
	framework.ExpectNoError(err, "Failed to create apparmor-profiles ConfigMap")
}

func createAppArmorProfileLoader(f *framework.Framework) {
	True := true
	One := int32(1)
	loader := &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "apparmor-loader",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &One,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{loaderLabelKey: loaderLabelValue},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Name:  "apparmor-loader",
						Image: imageutils.GetE2EImage(imageutils.AppArmorLoader),
						Args:  []string{"-poll", "10s", "/profiles"},
						SecurityContext: &v1.SecurityContext{
							Privileged: &True,
						},
						VolumeMounts: []v1.VolumeMount{{
							Name:      "sys",
							MountPath: "/sys",
							ReadOnly:  true,
						}, {
							Name:      "apparmor-includes",
							MountPath: "/etc/apparmor.d",
							ReadOnly:  true,
						}, {
							Name:      "profiles",
							MountPath: "/profiles",
							ReadOnly:  true,
						}},
					}},
					Volumes: []v1.Volume{{
						Name: "sys",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/sys",
							},
						},
					}, {
						Name: "apparmor-includes",
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: "/etc/apparmor.d",
							},
						},
					}, {
						Name: "profiles",
						VolumeSource: v1.VolumeSource{
							ConfigMap: &v1.ConfigMapVolumeSource{
								LocalObjectReference: v1.LocalObjectReference{
									Name: "apparmor-profiles",
								},
							},
						},
					}},
				},
			},
		},
	}
	_, err := f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(loader)
	framework.ExpectNoError(err, "Failed to create apparmor-loader ReplicationController")

	// Wait for loader to be ready.
	getRunningLoaderPod(f)
}

func getRunningLoaderPod(f *framework.Framework) *v1.Pod {
	label := labels.SelectorFromSet(labels.Set(map[string]string{loaderLabelKey: loaderLabelValue}))
	pods, err := e2epod.WaitForPodsWithLabelScheduled(f.ClientSet, f.Namespace.Name, label)
	framework.ExpectNoError(err, "Failed to schedule apparmor-loader Pod")
	pod := &pods.Items[0]
	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(f.ClientSet, pod), "Failed to run apparmor-loader Pod")
	return pod
}
