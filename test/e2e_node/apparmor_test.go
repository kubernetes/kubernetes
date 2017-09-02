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

package e2e_node

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("AppArmor [Feature:AppArmor]", func() {
	if isAppArmorEnabled() {
		BeforeEach(func() {
			By("Loading AppArmor profiles for testing")
			framework.ExpectNoError(loadTestProfiles(), "Could not load AppArmor test profiles")
		})
		Context("when running with AppArmor", func() {
			f := framework.NewDefaultFramework("apparmor-test")

			It("should reject an unloaded profile", func() {
				status := runAppArmorTest(f, false, apparmor.ProfileNamePrefix+"non-existant-profile")
				expectSoftRejection(status)
			})
			It("should enforce a profile blocking writes", func() {
				status := runAppArmorTest(f, true, apparmor.ProfileNamePrefix+apparmorProfilePrefix+"deny-write")
				if len(status.ContainerStatuses) == 0 {
					framework.Failf("Unexpected pod status: %s", spew.Sdump(status))
					return
				}
				state := status.ContainerStatuses[0].State.Terminated
				Expect(state).ToNot(BeNil(), "ContainerState: %+v", status.ContainerStatuses[0].State)
				Expect(state.ExitCode).To(Not(BeZero()), "ContainerStateTerminated: %+v", state)

			})
			It("should enforce a permissive profile", func() {
				status := runAppArmorTest(f, true, apparmor.ProfileNamePrefix+apparmorProfilePrefix+"audit-write")
				if len(status.ContainerStatuses) == 0 {
					framework.Failf("Unexpected pod status: %s", spew.Sdump(status))
					return
				}
				state := status.ContainerStatuses[0].State.Terminated
				Expect(state).ToNot(BeNil(), "ContainerState: %+v", status.ContainerStatuses[0].State)
				Expect(state.ExitCode).To(BeZero(), "ContainerStateTerminated: %+v", state)
			})
		})
	} else {
		Context("when running without AppArmor", func() {
			f := framework.NewDefaultFramework("apparmor-test")

			It("should reject a pod with an AppArmor profile", func() {
				status := runAppArmorTest(f, false, apparmor.ProfileRuntimeDefault)
				expectSoftRejection(status)
			})
		})
	}
})

const apparmorProfilePrefix = "e2e-node-apparmor-test-"
const testProfiles = `
#include <tunables/global>

profile e2e-node-apparmor-test-deny-write flags=(attach_disconnected) {
  #include <abstractions/base>

  file,

  # Deny all file writes.
  deny /** w,
}

profile e2e-node-apparmor-test-audit-write flags=(attach_disconnected) {
  #include <abstractions/base>

  file,

  # Only audit file writes.
  audit /** w,
}
`

func loadTestProfiles() error {
	f, err := ioutil.TempFile("/tmp", "apparmor")
	if err != nil {
		return fmt.Errorf("failed to open temp file: %v", err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	if _, err := f.WriteString(testProfiles); err != nil {
		return fmt.Errorf("failed to write profiles to file: %v", err)
	}

	// TODO(random-liu): The test is run as root now, no need to use sudo here.
	cmd := exec.Command("sudo", "apparmor_parser", "-r", "-W", f.Name())
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr
	out, err := cmd.Output()
	// apparmor_parser does not always return an error code, so consider any stderr output an error.
	if err != nil || stderr.Len() > 0 {
		if stderr.Len() > 0 {
			glog.Warning(stderr.String())
		}
		if len(out) > 0 {
			glog.Infof("apparmor_parser: %s", out)
		}
		return fmt.Errorf("failed to load profiles: %v", err)
	}
	glog.V(2).Infof("Loaded profiles: %v", out)
	return nil
}

func runAppArmorTest(f *framework.Framework, shouldRun bool, profile string) v1.PodStatus {
	pod := createPodWithAppArmor(f, profile)
	if shouldRun {
		// The pod needs to start before it stops, so wait for the longer start timeout.
		framework.ExpectNoError(framework.WaitTimeoutForPodNoLongerRunningInNamespace(
			f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
	} else {
		// Pod should remain in the pending state. Wait for the Reason to be set to "AppArmor".
		w, err := f.PodClient().Watch(metav1.SingleObject(metav1.ObjectMeta{Name: pod.Name}))
		framework.ExpectNoError(err)
		_, err = watch.Until(framework.PodStartTimeout, w, func(e watch.Event) (bool, error) {
			switch e.Type {
			case watch.Deleted:
				return false, errors.NewNotFound(schema.GroupResource{Resource: "pods"}, pod.Name)
			}
			switch t := e.Object.(type) {
			case *v1.Pod:
				if t.Status.Reason == "AppArmor" {
					return true, nil
				}
			}
			return false, nil
		})
		framework.ExpectNoError(err)
	}
	p, err := f.PodClient().Get(pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return p.Status
}

func createPodWithAppArmor(f *framework.Framework, profile string) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("test-apparmor-%s", strings.Replace(profile, "/", "-", -1)),
			Annotations: map[string]string{
				apparmor.ContainerAnnotationKeyPrefix + "test": profile,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:    "test",
				Image:   busyboxImage,
				Command: []string{"touch", "foo"},
			}},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	return f.PodClient().Create(pod)
}

func expectSoftRejection(status v1.PodStatus) {
	args := []interface{}{"PodStatus: %+v", status}
	Expect(status.Phase).To(Equal(v1.PodPending), args...)
	Expect(status.Reason).To(Equal("AppArmor"), args...)
	Expect(status.Message).To(ContainSubstring("AppArmor"), args...)
	Expect(status.ContainerStatuses[0].State.Waiting.Reason).To(Equal("Blocked"), args...)
}

func isAppArmorEnabled() bool {
	// TODO(tallclair): Pass this through the image setup rather than hardcoding.
	if strings.Contains(framework.TestContext.NodeName, "-gci-dev-") {
		gciVersionRe := regexp.MustCompile("-gci-dev-([0-9]+)-")
		matches := gciVersionRe.FindStringSubmatch(framework.TestContext.NodeName)
		if len(matches) == 2 {
			version, err := strconv.Atoi(matches[1])
			if err != nil {
				glog.Errorf("Error parsing GCI version from NodeName %q: %v", framework.TestContext.NodeName, err)
				return false
			}
			return version >= 54
		}
		return false
	}
	if strings.Contains(framework.TestContext.NodeName, "-ubuntu-") {
		return true
	}
	return apparmor.IsAppArmorEnabled()
}
