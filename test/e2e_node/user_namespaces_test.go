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
	"os/exec"
	"os/user"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var (
	customIDsPerPod int64 = 65536 * 2
	// kubelet user used for userns mapping.
	kubeletUserForUsernsMapping = "kubelet"
	getsubuidsBinary            = "getsubids"
)

var _ = SIGDescribe("UserNamespaces", "[LinuxOnly]", feature.UserNamespacesSupport, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("user-namespace-off-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	f.Context("when UserNamespacesSupport=false in the kubelet", func() {
		// Turn off UserNamespacesSupport for this test
		// TODO: once the UserNamespacesSupport feature is removed, this test should be removed too
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(kubefeatures.UserNamespacesSupport)] = false
			initialConfig.FeatureGates[string(kubefeatures.ProcMountType)] = false
		})
		f.It("will fail to create a hostUsers=false pod", func(ctx context.Context) {
			if on, ok := serviceFeatureGates[string(kubefeatures.UserNamespacesSupport)]; !ok || !on {
				e2eskipper.Skipf("services do not have user namespaces on")
			}
			falseVar := false
			podClient := e2epod.NewPodClient(f)
			pod, err := podClient.PodInterface.Create(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "userns-pod"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "test-container-1",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sleep"},
							Args:    []string{"10000"},
						},
					},
					HostUsers: &falseVar,
				},
			}, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			// Pod should stay in pending
			// Events would be a better way to tell this, as we could actually read the event,
			// but history proves events aren't reliable enough to base a test on.
			gomega.Consistently(ctx, func() error {
				p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				if p.Status.Phase != v1.PodPending {
					return fmt.Errorf("Pod phase isn't pending")
				}
				return nil
			}, 30*time.Second, 5*time.Second).ShouldNot(gomega.HaveOccurred())
		})
	})
})

var _ = SIGDescribe("user namespaces kubeconfig tests", "[LinuxOnly]", feature.UserNamespacesSupport, framework.WithFeatureGate(kubefeatures.UserNamespacesSupport), framework.WithSerial(), func() {
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
			expected := []string{strconv.FormatInt(customIDsPerPod, 10)}
			e2eoutput.TestContainerOutput(ctx, f, "idsPerPod is configured correctly", pod, 0, expected)
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
