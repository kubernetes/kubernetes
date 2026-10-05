//go:build linux

/*
Copyright 2024 The Kubernetes Authors.

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

package e2enode

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletconfigpaths "k8s.io/kubernetes/pkg/kubelet/kubeletconfig"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e_node/services"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

var (
	customIDsPerPod int64 = 65536 * 2
	// kubelet user used for userns mapping.
	kubeletUserForUsernsMapping = "kubelet"
	getsubuidsBinary            = "getsubids"
)

var _ = SIGDescribe("user namespaces kubeconfig tests", "[LinuxOnly]", feature.UserNamespacesSupport, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("userns-kubeconfig")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	f.Context("test config using userNamespaces.idsPerPod", func() {
		ginkgo.BeforeEach(func() {
			if hasMappings, err := hasKubeletUsernsMappings(); err != nil {
				framework.Failf("failed to check kubelet user namespace mappings: %v", err)
			} else if hasMappings {
				// idsPerPod needs to be in sync with the kubelet's user namespace
				// mappings. Let's skip the test if there are mappings present.
				e2eskipper.Skipf("kubelet is configured with custom user namespace mappings, skipping test")
			}
		})

		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.UserNamespaces == nil {
				initialConfig.UserNamespaces = &kubeletconfig.UserNamespaces{}
			}
			initialConfig.UserNamespaces.IDsPerPod = &customIDsPerPod
		})
		f.It("honors idsPerPod in userns pods", func(ctx context.Context) {
			if !supportsUserNS(ctx, f) {
				e2eskipper.Skipf("runtime does not support user namespaces")
			}
			falseVar := false
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "userns-pod" + string(uuid.NewUUID())},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "container",
							Image: imageutils.GetE2EImage(imageutils.BusyBox),
							// The third field is the mapping length, that must be equal to idsPerPod.
							Command: []string{"awk", "NR != 1 { exit 1 } { print $3 }", "/proc/self/uid_map"},
						},
					},
					HostUsers:     &falseVar,
					RestartPolicy: v1.RestartPolicyNever,
				},
			}
			podClient := e2epod.NewPodClient(f)
			createdPod := podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) {
				ginkgo.By("delete the pod")
				podClient.DeleteSync(ctx, createdPod.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To(int64(0))}, f.Timeouts.PodDelete)
				// DeleteSync waits until the pod is deleted from the API server.
				// But we need the pod dir to removed from the node before we
				// continue. The pod dir is not deleted with the pod, but left for
				// the periodic run of the cleanup function to delete it later. So,
				// let's wait until the dir is removed.
				// The reason we need to wait for the dir to be removed is because
				// tempSetCurrentKubeletConfig's AfterEach will restart the kubelet
				// with the original idsPerPod. If the pod directory is still on
				// disk, the kubelet will fail to start as the new kubelet config
				// can't be honored if we have pods on disk with another config.
				podDir := filepath.Join(services.KubeletRootDirectory, kubeletconfigpaths.DefaultKubeletPodsDirName, string(createdPod.UID))
				gomega.Eventually(ctx, func() bool {
					_, err := os.Stat(podDir)
					return os.IsNotExist(err)
				}).WithTimeout(f.Timeouts.PodDelete).Should(gomega.BeTrueBecause("pod directory %s must be removed - kubelet can't restart", podDir))
			})

			err := e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, f.ClientSet, createdPod.Name, f.Namespace.Name, f.Timeouts.PodStart)
			framework.ExpectNoError(err)

			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, createdPod.Name, "container")
			framework.ExpectNoError(err)
			expected := strconv.FormatInt(customIDsPerPod, 10)
			gomega.Expect(logs).To(gomega.ContainSubstring(expected))

		})
	})
})

func hasKubeletUsernsMappings() (bool, error) {
	if _, err := user.Lookup(kubeletUserForUsernsMapping); err != nil {
		var e user.UnknownUserError
		if errors.As(err, &e) {
			err = nil
		}
		return false, err
	}
	cmdBin, err := exec.LookPath(getsubuidsBinary)
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			err = nil
		}
		return false, err
	}
	outUids, err := getsubids(cmdBin, kubeletUserForUsernsMapping)
	if err != nil {
		return false, err
	}
	if outUids == "" {
		return false, nil
	}
	outGids, err := getsubids(cmdBin, "-g", kubeletUserForUsernsMapping)
	if err != nil {
		return false, err
	}
	if string(outUids) != string(outGids) {
		return false, fmt.Errorf("user %q has different subuids and subgids: %q vs %q", kubeletUserForUsernsMapping, outUids, outGids)
	}
	return true, nil
}

// getsubids runs the getsubids command to fetch subuid mappings for a user.
// If the command fails with "Error fetching ranges", it returns an empty string
// to indicate that no subuid mappings were found, which is not considered an error.
// Otherwise, it returns the output of the command as a string.
// (e.g., "0: user 100000 65536")
func getsubids(cmdBin string, cmdArgs ...string) (string, error) {
	var stderr bytes.Buffer
	cmd := exec.Command(cmdBin, cmdArgs...)
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		if strings.TrimSpace(stderr.String()) == "Error fetching ranges" {
			return "", nil // No subuid mappings found, this is not an error
		}
		return "", fmt.Errorf("failed to run %v: %w (stderr=%q)", cmd.Args, err, stderr.String())
	}
	return strings.TrimSpace(string(out)), nil
}
