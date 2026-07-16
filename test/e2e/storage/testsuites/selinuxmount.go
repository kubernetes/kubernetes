/*
Copyright 2025 The Kubernetes Authors.

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

package testsuites

import (
	"context"
	"fmt"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

type seLinuxMountTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

var (
	// Pod SELinux label used in the test pods.
	seLinuxLabel = &v1.SELinuxOptions{Level: "s0:c0,c1"}
	// The expected SELinux mount option, matching both Debian (system_u:object_r:svirt_lxc_net_t) and Fedore based distros (system_u:object_r:container_file_t)
	seLinuxMountOptionRE = "context=\"system_u:object_r:[^:]*:s0:c0,c1\""
)

// InitCustomSELinuxMountTestSuite returns seLinuxMountTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomSELinuxMountTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &seLinuxMountTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "seLinuxMount",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
			TestTags: []interface{}{
				feature.SELinux,
				framework.WithFeatureGate(features.SELinuxMountReadWriteOncePod),
				framework.WithFeatureGate(features.SELinuxChangePolicy),
			},
		},
	}
}

// InitSELinuxMountTestSuite returns seLinuxMountTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitSELinuxMountTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DefaultFsDynamicPV,
	}
	return InitCustomSELinuxMountTestSuite(patterns)
}

func (s *seLinuxMountTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return s.tsInfo
}

func (s *seLinuxMountTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	if !driver.GetDriverInfo().Capabilities[storageframework.CapSELinuxMount] {
		e2eskipper.Skipf("Driver %q does not support SELinuxMount - skipping", driver.GetDriverInfo().Name)
	}
}

func (s *seLinuxMountTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

		resource *storageframework.VolumeResource
		pod      *v1.Pod
	}
	var l local

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("selinux-mount", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context, accessMode v1.PersistentVolumeAccessMode) {
		l = local{}
		l.config = driver.PrepareTest(ctx, f)
		testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
		l.resource = storageframework.CreateVolumeResourceWithAccessModes(ctx, driver, l.config, pattern, testVolumeSizeRange, []v1.PersistentVolumeAccessMode{accessMode}, nil)
	}

	cleanup := func(ctx context.Context) {
		var errs []error
		if l.pod != nil {
			ginkgo.By("Deleting pod")
			err := e2epod.DeletePodWithWait(ctx, f.ClientSet, l.pod)
			errs = append(errs, err)
			l.pod = nil
		}

		if l.resource != nil {
			errs = append(errs, l.resource.CleanupResource(ctx))
			l.resource = nil
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
	}

	tests := []struct {
		name                     string
		accessMode               v1.PersistentVolumeAccessMode
		seLinuxChangePolicy      *v1.PodSELinuxChangePolicy
		featureGates             []featuregate.Feature
		expectSELinuxMountOption bool
	}{
		{
			name:                     "Pod with RWOP volume and the default policy",
			accessMode:               v1.ReadWriteOncePod,
			seLinuxChangePolicy:      nil, // implies mount option
			expectSELinuxMountOption: true,
		},
		{
			name:                     "Pod with RWO volume and mount policy",
			accessMode:               v1.ReadWriteOnce,
			seLinuxChangePolicy:      ptr.To(v1.SELinuxChangePolicyMountOption),
			featureGates:             []featuregate.Feature{features.SELinuxMount},
			expectSELinuxMountOption: true,
		},
		{
			name:                     "Pod with RWOP volume and recursive policy",
			accessMode:               v1.ReadWriteOncePod,
			seLinuxChangePolicy:      ptr.To(v1.SELinuxChangePolicyRecursive),
			expectSELinuxMountOption: false,
		},
		{
			name:                     "Pod with RWO and recursive policy",
			accessMode:               v1.ReadWriteOnce,
			seLinuxChangePolicy:      ptr.To(v1.SELinuxChangePolicyRecursive),
			expectSELinuxMountOption: false,
		},
	}
	for _, test := range tests {
		framework.Context(test.name, func() {
			// Compose framework.It arguments dynamically, because the feature gates are dynamic
			var args []interface{}
			// Test name
			if test.expectSELinuxMountOption {
				args = append(args, "should mount volumes with SELinux mount option")
			} else {
				args = append(args, "should mount volumes without SELinux mount option")
			}
			// Feature gates
			for _, fg := range test.featureGates {
				args = append(args, framework.WithFeatureGate(fg))
			}
			// The test body
			args = append(args, func(ctx context.Context) {
				init(ctx, test.accessMode)
				ginkgo.DeferCleanup(cleanup)

				ginkgo.By("Creating a pod with PVC")
				podConfig := e2epod.Config{
					NS:                     f.Namespace.Name,
					PVCs:                   []*v1.PersistentVolumeClaim{l.resource.Pvc},
					SeLinuxLabel:           seLinuxLabel,
					NodeSelection:          l.config.ClientNodeSelection,
					PodSELinuxChangePolicy: test.seLinuxChangePolicy,
				}
				var err error
				l.pod, err = e2epod.CreateSecPod(ctx, f.ClientSet, &podConfig, f.Timeouts.PodStart)
				framework.ExpectNoError(err, "while creating the pod")

				ginkgo.By("Waiting for pod to be ready")
				err = e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, l.pod)
				framework.ExpectNoError(err, "while waiting for the pod to be ready")

				ginkgo.By("Checking the volume mount options")
				mountOptions, err := getVolumeMountOptions(f, l.pod.Name, "write-pod", "/mnt/volume1")
				framework.ExpectNoError(err, "while getting the mount options")
				framework.Logf("Detected mount options: %s", mountOptions)
				if test.expectSELinuxMountOption {
					gomega.Expect(mountOptions).To(gomega.MatchRegexp(seLinuxMountOptionRE), "-o context mount option should be present")
				} else {
					gomega.Expect(mountOptions).NotTo(gomega.MatchRegexp(seLinuxMountOptionRE), "-o context mount option should not be present")
				}
			})
			framework.It(args...)
		})
	}
}

func getVolumeMountOptions(f *framework.Framework, podName string, containerName string, volumePath string) (string, error) {
	cmd := []string{"cat", "/proc/mounts"}
	stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, podName, containerName, cmd...)
	if err != nil {
		// Log all details about the call, except the error that will be logged by the caller
		framework.Logf("Executed: %s", strings.Join(cmd, " "))
		framework.Logf("stdout: %s", stdout)
		framework.Logf("stderr: %s", stderr)
		return "", err
	}

	for _, line := range strings.Split(stdout, "\n") {
		parts := strings.Fields(line)
		if len(parts) < 4 {
			continue
		}
		mountPath := parts[1]
		if mountPath == volumePath {
			return parts[3], nil
		}
	}
	return "", fmt.Errorf("volume path %s not found in /proc/mounts", volumePath)
}
