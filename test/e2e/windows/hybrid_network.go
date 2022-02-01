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

	"k8s.io/kubernetes/pkg/features"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	linuxOS   = "linux"
	windowsOS = "windows"
)

var (
	windowsBusyBoximage = imageutils.GetE2EImage(imageutils.Agnhost)
	linuxBusyBoxImage   = imageutils.GetE2EImage(imageutils.Nginx)
)

var _ = SIGDescribe("Hybrid cluster network", func() {
	f := framework.NewDefaultFramework("hybrid-network")

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessNodeOSDistroIs("windows")
	})

	ginkgo.Context("for all supported CNIs", func() {

		ginkgo.It("should have stable networking for Linux and Windows pods", func() {

			linuxPod := createTestPod(f, linuxBusyBoxImage, linuxOS)
			ginkgo.By("creating a linux pod and waiting for it to be running")
			linuxPod = f.PodClient().CreateSync(linuxPod)

			windowsPod := createTestPod(f, windowsBusyBoximage, windowsOS)

			windowsPod.Spec.Containers[0].Args = []string{"test-webserver"}
			ginkgo.By("creating a windows pod and waiting for it to be running")
			windowsPod = f.PodClient().CreateSync(windowsPod)

			ginkgo.By("verifying pod external connectivity to the internet")

			ginkgo.By("checking connectivity to 8.8.8.8 53 (google.com) from Linux")
			assertConsistentConnectivity(f, linuxPod.ObjectMeta.Name, linuxOS, linuxCheck("8.8.8.8", 53))

			ginkgo.By("checking connectivity to www.google.com from Windows")
			assertConsistentConnectivity(f, windowsPod.ObjectMeta.Name, windowsOS, windowsCheck("www.google.com"))

			ginkgo.By("verifying pod internal connectivity to the cluster dataplane")

			ginkgo.By("checking connectivity from Linux to Windows")
			assertConsistentConnectivity(f, linuxPod.ObjectMeta.Name, linuxOS, linuxCheck(windowsPod.Status.PodIP, 80))

			ginkgo.By("checking connectivity from Windows to Linux")
			assertConsistentConnectivity(f, windowsPod.ObjectMeta.Name, windowsOS, windowsCheck(linuxPod.Status.PodIP))

		})

		ginkgo.It("should have stable networking for Linux and Windows with ClusterIP services", func() {
			serviceName := "stable-networking"
			ginkgo.By("creating service " + serviceName + " with type=ClusterIP in namespace " + f.Namespace.Name)
			jig := e2eservice.NewTestJig(f.ClientSet, f.Namespace.Name, serviceName)

			svc, err := jig.CreateTCPService(func(svc *v1.Service) {
				svc.Spec.Type = v1.ServiceTypeClusterIP
			})
			framework.ExpectNoError(err)

			ginkgo.By("creating some linux pods in the service")
			var linuxPod *v1.Pod
			for i := 0; i < 2; i++ {
				ginkgo.By("creating a linux pod and waiting for it to be running")
				linuxPod = createTestPod(f, linuxBusyBoxImage, linuxOS)
				linuxPod.ObjectMeta.Labels = svc.Labels
				linuxPod = f.PodClient().CreateSync(linuxPod)
			}

			ginkgo.By("creating some Windows pods in the service")
			var windowsPod *v1.Pod
			for i := 0; i < 2; i++ {
				ginkgo.By("creating a windows pod and waiting for it to be running")
				windowsPod = createTestPod(f, windowsBusyBoximage, windowsOS)
				windowsPod.Spec.Containers[0].Args = []string{"test-webserver"}
				windowsPod.ObjectMeta.Labels = svc.Labels
				windowsPod = f.PodClient().CreateSync(windowsPod)
			}

			ginkgo.By("verifying Service connectivity from pods outside the service")

			ginkgo.By("creating Linux testing Pod")
			linuxTestPod := createTestPod(f, linuxBusyBoxImage, linuxOS)
			linuxTestPod = f.PodClient().CreateSync(linuxTestPod)

			ginkgo.By("checking service connectivity from linux pod outside the service")
			assertConsistentConnectivity(f, linuxTestPod.ObjectMeta.Name, linuxOS, linuxCheck(svc.ObjectMeta.Name, 80))
			f.PodClient().Delete(context.Background(), linuxTestPod.ObjectMeta.Name, metav1.DeleteOptions{})

			ginkgo.By("creating Windows testing Pod")
			windowsTestPod := createTestPod(f, windowsBusyBoximage, windowsOS)
			windowsTestPod = f.PodClient().CreateSync(windowsTestPod)

			ginkgo.By("checking service connectivity from windows pod outside the service")
			assertConsistentConnectivity(f, windowsTestPod.ObjectMeta.Name, windowsOS, windowsCheck(svc.Spec.ClusterIP))
			f.PodClient().Delete(context.Background(), windowsTestPod.ObjectMeta.Name, metav1.DeleteOptions{})

			ginkgo.By("verifying Service connectivity from pods within the service")
			ginkgo.By("checking service connectivity from linux pod in the service")
			assertConsistentConnectivity(f, linuxPod.ObjectMeta.Name, linuxOS, linuxCheck(svc.Spec.ClusterIP, 80))

			ginkgo.By("checking service connectivity from windows pod in the service")
			assertConsistentConnectivity(f, windowsPod.ObjectMeta.Name, windowsOS, windowsCheck(svc.Spec.ClusterIP))

			ginkgo.By("verifying Service connectivity from linux host to service")

			ginkgo.By("creating Linux HostNetwork testing Pod")
			linuxHostNetworkpod := createTestPod(f, linuxBusyBoxImage, linuxOS)
			linuxHostNetworkpod.Spec.HostNetwork = true
			linuxHostNetworkpod = f.PodClient().CreateSync(linuxHostNetworkpod)

			ginkgo.By("checking service connectivity from linux pod with hostnetwork")
			assertConsistentConnectivity(f, linuxHostNetworkpod.ObjectMeta.Name, linuxOS, linuxCheck(svc.Spec.ClusterIP, 80))

			if framework.TestContext.FeatureGates[string(features.WindowsHostProcessContainers)] {
				ginkgo.By("verifying Service connectivity from windows host to service")

				ginkgo.By("creating Windows HostProcess testing Pod")
				windowsHostProcessTestPod := createTestPod(f, windowsBusyBoximage, windowsOS)
				makeHostProcess(windowsHostProcessTestPod)
				// need to override the anghost command to point to the hostprocess container mount point on the host when it is hostprocess
				windowsHostProcessTestPod.Spec.Containers[0].Command = []string{"%CONTAINER_SANDBOX_MOUNT_POINT%/agnhost", "pause"}
				windowsHostProcessTestPod = f.PodClient().CreateSync(windowsHostProcessTestPod)

				ginkgo.By("checking service connectivity from windows with hostnetwork")
				assertConsistentConnectivity(f, windowsHostProcessTestPod.ObjectMeta.Name, windowsOS, windowsCheck(svc.Spec.ClusterIP))
			} else {
				framework.Logf("Skipping verification of Windows host to service because HostProcess Containers feature is disabled in e2e framework")
			}
		})

	})
})

var (
	duration       = "10s"
	pollInterval   = "1s"
	timeoutSeconds = 10
)

func assertConsistentConnectivity(f *framework.Framework, podName string, os string, cmd []string) {
	connChecker := func() error {
		ginkgo.By(fmt.Sprintf("checking connectivity of %s-container in %s", os, podName))
		// TODO, we should be retrying this similar to what is done in DialFromNode, in the test/e2e/networking/networking.go tests
		stdout, stderr, err := f.ExecCommandInContainerWithFullOutput(podName, os+"-container", cmd...)
		if err != nil {
			framework.Logf("Encountered error while running command: %v.\nStdout: %s\nStderr: %s\nErr: %v", cmd, stdout, stderr, err)
		}
		return err
	}
	gomega.Eventually(connChecker, duration, pollInterval).ShouldNot(gomega.HaveOccurred())
	gomega.Consistently(connChecker, duration, pollInterval).ShouldNot(gomega.HaveOccurred())
}

func linuxCheck(address string, port int) []string {
	nc := fmt.Sprintf("nc -vz %s %v -w %v", address, port, timeoutSeconds)
	cmd := []string{"/bin/sh", "-c", nc}
	return cmd
}

func windowsCheck(address string) []string {
	curl := fmt.Sprintf("curl.exe %s --connect-timeout %v --fail", address, timeoutSeconds)
	cmd := []string{"cmd", "/c", curl}
	return cmd
}

func createTestPod(f *framework.Framework, image string, os string) *v1.Pod {
	containerName := fmt.Sprintf("%s-container", os)
	podName := "pod-" + string(uuid.NewUUID())
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  containerName,
					Image: image,
					Ports: []v1.ContainerPort{{ContainerPort: 80}},
				},
			},
			NodeSelector: map[string]string{
				"kubernetes.io/os": os,
			},
		},
	}
	if os == linuxOS {
		pod.Spec.Tolerations = []v1.Toleration{
			{
				Operator: v1.TolerationOpExists,
				Effect:   v1.TaintEffectNoSchedule,
			},
		}
	}
	return pod
}

func makeHostProcess(pod *v1.Pod) {
	pod.Spec.SecurityContext = &v1.PodSecurityContext{
		WindowsOptions: &v1.WindowsSecurityContextOptions{
			HostProcess:   &trueVar,
			RunAsUserName: &User_NTAuthorityLocalService,
		},
	}
	pod.Spec.HostNetwork = true
}
