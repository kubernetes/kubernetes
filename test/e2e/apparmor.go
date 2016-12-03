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

package e2e

import (
	"fmt"

	api "k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	profilePrefix = "e2e-apparmor-test-"
	allowedPath   = "/expect_allowed_write"
	deniedPath    = "/expect_permission_denied"
)

var _ = framework.KubeDescribe("AppArmor", func() {
	f := framework.NewDefaultFramework("apparmor")

	BeforeEach(func() {
		SkipIfAppArmorNotSupported()
		LoadAppArmorProfiles(f)
	})

	It("should enforce an AppArmor profile", func() {
		profile := "localhost/" + profilePrefix + f.Namespace.Name
		testCmd := fmt.Sprintf(`
if touch %[1]s; then
  echo "FAILURE: write to %[1]s should be denied"
  exit 1
elif ! touch %[2]s; then
  echo "FAILURE: write to %[2]s should be allowed"
  exit 2
fi`, deniedPath, allowedPath)
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "test-apparmor",
				Annotations: map[string]string{
					apparmor.ContainerAnnotationKeyPrefix + "test": profile,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{{
					Name:    "test",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"sh", "-c", testCmd},
				}},
				RestartPolicy: api.RestartPolicyNever,
			},
		}
		f.PodClient().Create(pod)
		framework.ExpectNoError(framework.WaitForPodSuccessInNamespace(
			f.ClientSet, pod.Name, f.Namespace.Name))
		framework.LogFailedContainers(f.ClientSet, f.Namespace.Name, framework.Logf)
	})
})

func SkipIfAppArmorNotSupported() {
	framework.SkipUnlessNodeOSDistroIs("gci", "ubuntu")
}

func LoadAppArmorProfiles(f *framework.Framework) {
	_, err := createAppArmorProfileCM(f)
	framework.ExpectNoError(err)
	_, err = createAppArmorProfileLoader(f)
	framework.ExpectNoError(err)
}

func createAppArmorProfileCM(f *framework.Framework) (*api.ConfigMap, error) {
	profileName := profilePrefix + f.Namespace.Name
	profile := fmt.Sprintf(`#include <tunables/global>
profile %s flags=(attach_disconnected) {
  #include <abstractions/base>

  file,

  deny %s w,
  audit %s w,
}
`, profileName, deniedPath, allowedPath)

	cm := &api.ConfigMap{
		ObjectMeta: api.ObjectMeta{
			Name:      "apparmor-profiles",
			Namespace: f.Namespace.Name,
		},
		Data: map[string]string{
			profileName: profile,
		},
	}
	return f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Create(cm)
}

func createAppArmorProfileLoader(f *framework.Framework) (*extensions.DaemonSet, error) {
	True := true
	// Copied from https://github.com/kubernetes/contrib/blob/master/apparmor/loader/example-configmap.yaml
	loader := &extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{
			Name:      "apparmor-loader",
			Namespace: f.Namespace.Name,
		},
		Spec: extensions.DaemonSetSpec{
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"name": "apparmor-loader"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Name:  "apparmor-loader",
						Image: "gcr.io/google_containers/apparmor-loader:0.1",
						Args:  []string{"-poll", "10s", "/profiles"},
						SecurityContext: &api.SecurityContext{
							Privileged: &True,
						},
						VolumeMounts: []api.VolumeMount{{
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
					Volumes: []api.Volume{{
						Name: "sys",
						VolumeSource: api.VolumeSource{
							HostPath: &api.HostPathVolumeSource{
								Path: "/sys",
							},
						},
					}, {
						Name: "apparmor-includes",
						VolumeSource: api.VolumeSource{
							HostPath: &api.HostPathVolumeSource{
								Path: "/etc/apparmor.d",
							},
						},
					}, {
						Name: "profiles",
						VolumeSource: api.VolumeSource{
							ConfigMap: &api.ConfigMapVolumeSource{
								LocalObjectReference: api.LocalObjectReference{
									Name: "apparmor-profiles",
								},
							},
						},
					}},
				},
			},
		},
	}
	return f.ClientSet.Extensions().DaemonSets(f.Namespace.Name).Create(loader)
}
