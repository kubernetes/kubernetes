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

package storage

import (
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = utils.SIGDescribe("Subpath", func() {
	f := framework.NewDefaultFramework("subpath")

	Context("Atomic writer volumes", func() {
		var err error
		var privilegedSecurityContext bool = false

		BeforeEach(func() {
			By("Setting up data")
			secret := &v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "my-secret"}, Data: map[string][]byte{"secret-key": []byte("secret-value")}}
			secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(secret)
			if err != nil && !apierrors.IsAlreadyExists(err) {
				Expect(err).ToNot(HaveOccurred(), "while creating secret")
			}

			configmap := &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: "my-configmap"}, Data: map[string]string{"configmap-key": "configmap-value"}}
			configmap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(configmap)
			if err != nil && !apierrors.IsAlreadyExists(err) {
				Expect(err).ToNot(HaveOccurred(), "while creating configmap")
			}

		})

		/*
		  Release : v1.12
		  Testname: SubPath: Reading content from a secret volume.
		  Description: Containers in a pod can read content from a secret mounted volume which was configured with a subpath.
		  This test is marked LinuxOnly since Windows cannot mount individual files in Containers.
		*/
		framework.ConformanceIt("should support subpaths with secret pod [LinuxOnly]", func() {
			pod := testsuites.SubpathTestPod(f, "secret-key", "secret", &v1.VolumeSource{Secret: &v1.SecretVolumeSource{SecretName: "my-secret"}}, privilegedSecurityContext)
			testsuites.TestBasicSubpath(f, "secret-value", pod)
		})

		/*
		  Release : v1.12
		  Testname: SubPath: Reading content from a configmap volume.
		  Description: Containers in a pod can read content from a configmap mounted volume which was configured with a subpath.
		  This test is marked LinuxOnly since Windows cannot mount individual files in Containers.
		*/
		framework.ConformanceIt("should support subpaths with configmap pod [LinuxOnly]", func() {
			pod := testsuites.SubpathTestPod(f, "configmap-key", "configmap", &v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{LocalObjectReference: v1.LocalObjectReference{Name: "my-configmap"}}}, privilegedSecurityContext)
			testsuites.TestBasicSubpath(f, "configmap-value", pod)
		})

		/*
		  Release : v1.12
		  Testname: SubPath: Reading content from a configmap volume.
		  Description: Containers in a pod can read content from a configmap mounted volume which was configured with a subpath and also using a mountpath that is a specific file.
		  This test is marked LinuxOnly since Windows cannot mount individual files in Containers.
		*/
		framework.ConformanceIt("should support subpaths with configmap pod with mountPath of existing file [LinuxOnly]", func() {
			pod := testsuites.SubpathTestPod(f, "configmap-key", "configmap", &v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{LocalObjectReference: v1.LocalObjectReference{Name: "my-configmap"}}}, privilegedSecurityContext)
			file := "/etc/resolv.conf"
			pod.Spec.Containers[0].VolumeMounts[0].MountPath = file
			testsuites.TestBasicSubpathFile(f, "configmap-value", pod, file)
		})

		/*
		  Release : v1.12
		  Testname: SubPath: Reading content from a downwardAPI volume.
		  Description: Containers in a pod can read content from a downwardAPI mounted volume which was configured with a subpath.
		  This test is marked LinuxOnly since Windows cannot mount individual files in Containers.
		*/
		framework.ConformanceIt("should support subpaths with downward pod [LinuxOnly]", func() {
			pod := testsuites.SubpathTestPod(f, "downward/podname", "downwardAPI", &v1.VolumeSource{
				DownwardAPI: &v1.DownwardAPIVolumeSource{
					Items: []v1.DownwardAPIVolumeFile{{Path: "downward/podname", FieldRef: &v1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.name"}}},
				},
			}, privilegedSecurityContext)
			testsuites.TestBasicSubpath(f, pod.Name, pod)
		})

		/*
		  Release : v1.12
		  Testname: SubPath: Reading content from a projected volume.
		  Description: Containers in a pod can read content from a projected mounted volume which was configured with a subpath.
		  This test is marked LinuxOnly since Windows cannot mount individual files in Containers.
		*/
		framework.ConformanceIt("should support subpaths with projected pod [LinuxOnly]", func() {
			pod := testsuites.SubpathTestPod(f, "projected/configmap-key", "projected", &v1.VolumeSource{
				Projected: &v1.ProjectedVolumeSource{
					Sources: []v1.VolumeProjection{
						{ConfigMap: &v1.ConfigMapProjection{
							LocalObjectReference: v1.LocalObjectReference{Name: "my-configmap"},
							Items:                []v1.KeyToPath{{Path: "projected/configmap-key", Key: "configmap-key"}},
						}},
					},
				},
			}, privilegedSecurityContext)
			testsuites.TestBasicSubpath(f, "configmap-value", pod)
		})
	})
})
