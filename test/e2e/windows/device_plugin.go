/*
Copyright 2020 The Kubernetes Authors.

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
//	"context"
	"time"

//	v1 "k8s.io/api/core/v1"
//	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
//	appsv1 "k8s.io/api/apps/v1"
//	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Device Plugin", func() {
	f := framework.NewDefaultFramework("device-plugin")

//	var cs clientset.Interface

	ginkgo.BeforeEach(func() {
		//Only for Windows containers
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
//		cs = f.ClientSet
	})
	ginkgo.It("should be able to create a functioning device plugin for Windows", func() {
		ginkgo.By("creating Windows device plugin daemonset")
//		dsName := "directx-device-plugin"
//		daemonsetNameLabel := "daemonset-name"
//		image := "directxplugin"
//		mountName := "device-plugin"
//		mountPath := "/var/lib/kubelet/device-plugins"
//		privileged := true
/*		labels := map[string]string{
			daemonsetNameLabel: dsName,
		}*/
/*        	ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{
				Name: dsName,
				Namespace: "kube-system",
			},
			Spec: appsv1.DaemonSetSpec{
				Selector: &metav1.LabelSelector{
					MatchLabels: labels,
				},
				Template: v1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Annotations: map[string]string{
							"scheduler.alpha.kubernetes.io/critical-pod": "",
						},
						Labels: labels,
					},
					Spec: v1.PodSpec{
						Tolerations: []v1.Toleration{
							{
								Key: "CriticalAddonsOnly",
								Operator: "Exists",
							},
						},
						Containers: []v1.Container{
							{
								Name:  "hostdev",
								Image: image,
								SecurityContext: &v1.SecurityContext{
									Privileged: &privileged,
								},
								VolumeMounts: []v1.VolumeMount{
									{
										Name: mountName,
										MountPath: mountPath,
									},
								},
							},
						},
						Volumes: []v1.Volume{
							{
								Name: mountName,
								VolumeSource: v1.VolumeSource{
									HostPath: &v1.HostPathVolumeSource{
										Path: mountPath,
									},
								},
							},
						},
						NodeSelector: map[string]string{
							"kubernetes.io/os": "windows",
						},
					},
				},
			},
		}
*/
		ns := f.Namespace.Name
//		ds, err := cs.AppsV1().DaemonSets(ns).Create(context.TODO(), ds, metav1.CreateOptions{})
//		framework.ExpectNoError(err)

		ginkgo.By("creating Windows testing Pod")
		windowsPod := createTestPod(f, windowsBusyBoximage, windowsOS)
		windowsPod = f.PodClient().CreateSync(windowsPod)

		ginkgo.By("verifying device access in Windows testing Pod")
		command := []string{"cmd.exe", "/c", "dxdiag", "/t", "dxdiag_output.txt", "&", "type", "dxdiag_output.txt"}
		expectedString := "Todo: DirectX Version: DirectX 12"
		_, err := framework.LookForStringInPodExec(ns, windowsPod.Name, command, expectedString, time.Minute)
		framework.ExpectNoError(err, "failed: didn't find expected string in dxdiag output.")
	})
})
