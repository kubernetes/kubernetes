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
	"context"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

const (
	testSlowMultiplier = 60
)

var _ = SIGDescribe("[Feature:GPUDevicePlugin] Device Plugin", func() {
	f := framework.NewDefaultFramework("device-plugin")

	var cs clientset.Interface

	ginkgo.BeforeEach(func() {
		//Only for Windows containers
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
		cs = f.ClientSet
	})
	ginkgo.It("should be able to create a functioning device plugin for Windows", func() {
		ginkgo.By("creating Windows device plugin daemonset")
		dsName := "directx-device-plugin"
		daemonsetNameLabel := "daemonset-name"
		image := "e2eteam/k8s-directx-device-plugin:0.9.0-1809"
		mountName := "device-plugin"
		mountPath := "/var/lib/kubelet/device-plugins"
		labels := map[string]string{
			daemonsetNameLabel: dsName,
		}
		ds := &appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{
				Name:      dsName,
				Namespace: "kube-system",
			},
			Spec: appsv1.DaemonSetSpec{
				Selector: &metav1.LabelSelector{
					MatchLabels: labels,
				},
				Template: v1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: labels,
					},
					Spec: v1.PodSpec{
						PriorityClassName: "system-node-critical",
						Tolerations: []v1.Toleration{
							{
								Key:      "CriticalAddonsOnly",
								Operator: "Exists",
							},
						},
						Containers: []v1.Container{
							{
								Name:  "hostdev",
								Image: image,
								VolumeMounts: []v1.VolumeMount{
									{
										Name:      mountName,
										MountPath: mountPath,
									},
								},
								Env: []v1.EnvVar{
									{
										Name:  "DIRECTX_GPU_MATCH_NAME",
										Value: " ",
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

		sysNs := "kube-system"
		_, err := cs.AppsV1().DaemonSets(sysNs).Create(context.TODO(), ds, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("creating Windows testing Pod")
		windowsPod := createTestPod(f, imageutils.GetE2EImage(imageutils.WindowsServer), windowsOS)
		windowsPod.Spec.Containers[0].Args = []string{"powershell.exe", "Start-Sleep", "3600"}
		windowsPod.Spec.Containers[0].Resources.Limits = v1.ResourceList{
			"microsoft.com/directx": resource.MustParse("1"),
		}
		windowsPod, err = cs.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), windowsPod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		ginkgo.By("Waiting for the pod Running")
		err = e2epod.WaitTimeoutForPodRunningInNamespace(cs, windowsPod.Name, f.Namespace.Name, testSlowMultiplier*framework.PodStartTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("verifying device access in Windows testing Pod")
		dxdiagCommand := []string{"cmd.exe", "/c", "dxdiag", "/t", "dxdiag_output.txt", "&", "type", "dxdiag_output.txt"}
		//If DirectX version issues caused by supsequent windows releases occur, these tests need to do version checks
		//based on  the windows version running the test.
		dxdiagDirectxVersion := "DirectX Version: DirectX 12"
		defaultNs := f.Namespace.Name
		_, dxdiagDirectxVersionErr := framework.LookForStringInPodExec(defaultNs, windowsPod.Name, dxdiagCommand, dxdiagDirectxVersion, time.Minute)
		framework.ExpectNoError(dxdiagDirectxVersionErr, "failed: didn't find directX version dxdiag output.")

		dxdiagDdiVersion := "DDI Version: 12"
		_, dxdiagDdiVersionErr := framework.LookForStringInPodExec(defaultNs, windowsPod.Name, dxdiagCommand, dxdiagDdiVersion, time.Minute)
		framework.ExpectNoError(dxdiagDdiVersionErr, "failed: didn't find DDI version in dxdiag output.")

		dxdiagVendorID := "Vendor ID: 0x"
		_, dxdiagVendorIDErr := framework.LookForStringInPodExec(defaultNs, windowsPod.Name, dxdiagCommand, dxdiagVendorID, time.Minute)
		framework.ExpectNoError(dxdiagVendorIDErr, "failed: didn't find vendorID in dxdiag output.")

		envVarCommand := []string{"cmd.exe", "/c", "set", "DIRECTX_GPU_Name"}
		envVarDirectxGpuName := "DIRECTX_GPU_Name="
		_, envVarDirectxGpuNameErr := framework.LookForStringInPodExec(defaultNs, windowsPod.Name, envVarCommand, envVarDirectxGpuName, time.Minute)
		framework.ExpectNoError(envVarDirectxGpuNameErr, "failed: didn't find expected environment variable.")
	})
})
