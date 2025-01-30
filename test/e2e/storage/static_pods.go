/*
Copyright 2023 The Kubernetes Authors.

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
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Tests for static Pods + CSI volume reconstruction.
//
// NOTE: these tests *require* the cluster under test to be Kubernetes In Docker (kind)!
// Kind runs its API server as a static Pod, and we leverage it here
// to test kubelet starting without the API server.
var _ = utils.SIGDescribe("StaticPods", feature.Kind, func() {
	f := framework.NewDefaultFramework("static-pods-csi")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// global vars for the ginkgo.BeforeEach() / It()'s below
	var (
		driver     storageframework.DynamicPVTestDriver
		testConfig *storageframework.PerTestConfig
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		// Install a CSI driver. Using csi-driver-hospath here.
		driver = drivers.InitHostPathCSIDriver().(storageframework.DynamicPVTestDriver)
		testConfig = driver.PrepareTest(ctx, f)
	})

	// Test https://github.com/kubernetes/kubernetes/issues/117745
	// I.e. kubelet starts and it must start the API server as a static pod,
	// while there is a CSI volume mounted by the previous kubelet.
	f.It("should run after kubelet stopped with CSI volume mounted", f.WithDisruptive(), f.WithSerial(), func(ctx context.Context) {
		var timeout int64 = 5

		ginkgo.By("Provision a new CSI volume")
		res := storageframework.CreateVolumeResource(ctx, driver, testConfig, storageframework.DefaultFsDynamicPV, e2evolume.SizeRange{Min: "1", Max: "1Gi"})
		ginkgo.DeferCleanup(func(ctx context.Context) {
			res.CleanupResource(ctx)
		})

		ginkgo.By("Run a pod with the volume")
		podConfig := &e2epod.Config{
			NS:            f.Namespace.Name,
			PVCs:          []*v1.PersistentVolumeClaim{res.Pvc},
			NodeSelection: testConfig.ClientNodeSelection,
		}
		pod, err := e2epod.CreateSecPod(ctx, f.ClientSet, podConfig, f.Timeouts.PodStart)
		framework.ExpectNoError(err, "Creating a pod with a CSI volume")
		ginkgo.DeferCleanup(f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete, pod.Name, metav1.DeleteOptions{})

		ginkgo.By("Stop the control plane by removing the static pod manifests")
		err = stopControlPlane(ctx)
		framework.ExpectNoError(err, "Stopping the control plane")

		ginkgo.By("Wait for the API server to disappear")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			_, err := f.ClientSet.StorageV1().StorageClasses().List(ctx, metav1.ListOptions{TimeoutSeconds: &timeout})
			return err
		}).WithTimeout(time.Minute).Should(gomega.Not(gomega.Succeed()))

		ginkgo.By("Stop kubelet")
		err = stopKindKubelet(ctx)
		framework.ExpectNoError(err, "Stopping kubelet")

		ginkgo.By("Re-enable control plane")
		err = enableControlPlane(ctx)
		framework.ExpectNoError(err, "Re-enabling control plane by restoring the static pod manifests")

		ginkgo.By("Start kubelet")
		err = startKindKubelet(ctx)
		framework.ExpectNoError(err, "Starting kubelet")

		ginkgo.By("Wait for a static pod with the API server to start")
		const apiServerStartTimeout = 10 * time.Minute // It takes up to 2 minutes on my machine
		gomega.Eventually(ctx, func(ctx context.Context) error {
			_, err := f.ClientSet.StorageV1().StorageClasses().List(ctx, metav1.ListOptions{TimeoutSeconds: &timeout})
			return err
		}).WithTimeout(apiServerStartTimeout).Should(gomega.Succeed())
	})
})

func stopKindKubelet(ctx context.Context) error {
	return kubeletExec("systemctl", "stop", "kubelet")
}

func startKindKubelet(ctx context.Context) error {
	return kubeletExec("systemctl", "start", "kubelet")
}

func stopControlPlane(ctx context.Context) error {
	// Move all static pod manifests to a backup dir (incl. KCM, scheduler, API server and etcd)
	return kubeletExec("sh", "-c", "mkdir -p /etc/kubernetes/manifests-backup && mv /etc/kubernetes/manifests/* /etc/kubernetes/manifests-backup/")
}

func enableControlPlane(ctx context.Context) error {
	// Restore the static pod manifests from the backup dir.
	// Using `sh` to get `*` expansion.
	return kubeletExec("sh", "-c", "mv /etc/kubernetes/manifests-backup/* /etc/kubernetes/manifests/")
}

// Run a command in container with kubelet (and the whole control plane as containers)
func kubeletExec(command ...string) error {
	containerName := getKindContainerName()
	args := []string{"exec", containerName}
	args = append(args, command...)
	cmd := exec.Command("docker", args...)

	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command %q failed: %v\noutput:%s", prettyCmd(cmd), err, string(out))
	}

	framework.Logf("command %q succeeded:\n%s", prettyCmd(cmd), string(out))
	return nil
}

func getKindContainerName() string {
	clusterName := os.Getenv("KIND_CLUSTER_NAME")
	if clusterName == "" {
		clusterName = "kind"
	}
	return clusterName + "-control-plane"
}

func prettyCmd(cmd *exec.Cmd) string {
	return fmt.Sprintf("%s %s", cmd.Path, strings.Join(cmd.Args, " "))
}
