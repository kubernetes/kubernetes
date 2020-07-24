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

package e2enode

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"

	"github.com/davecgh/go-spew/spew"
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("AppArmor [Feature:AppArmor][NodeFeature:AppArmor]", func() {
	if isAppArmorEnabled() {
		ginkgo.BeforeEach(func() {
			ginkgo.By("Loading AppArmor profiles for testing")
			framework.ExpectNoError(loadTestProfiles(), "Could not load AppArmor test profiles")
		})
		ginkgo.Context("when running with AppArmor", func() {
			f := framework.NewDefaultFramework("apparmor-test")

			ginkgo.It("should reject an unloaded profile", func() {
				status := runAppArmorTest(f, false, v1.AppArmorBetaProfileNamePrefix+"non-existent-profile")
				expectSoftRejection(status)
			})
			ginkgo.It("should enforce a profile blocking writes", func() {
				status := runAppArmorTest(f, true, v1.AppArmorBetaProfileNamePrefix+apparmorProfilePrefix+"deny-write")
				if len(status.ContainerStatuses) == 0 {
					framework.Failf("Unexpected pod status: %s", spew.Sdump(status))
					return
				}
				state := status.ContainerStatuses[0].State.Terminated
				gomega.Expect(state).ToNot(gomega.BeNil(), "ContainerState: %+v", status.ContainerStatuses[0].State)
				gomega.Expect(state.ExitCode).To(gomega.Not(gomega.BeZero()), "ContainerStateTerminated: %+v", state)

			})
			ginkgo.It("should enforce a permissive profile", func() {
				status := runAppArmorTest(f, true, v1.AppArmorBetaProfileNamePrefix+apparmorProfilePrefix+"audit-write")
				if len(status.ContainerStatuses) == 0 {
					framework.Failf("Unexpected pod status: %s", spew.Sdump(status))
					return
				}
				state := status.ContainerStatuses[0].State.Terminated
				gomega.Expect(state).ToNot(gomega.BeNil(), "ContainerState: %+v", status.ContainerStatuses[0].State)
				gomega.Expect(state.ExitCode).To(gomega.BeZero(), "ContainerStateTerminated: %+v", state)
			})
		})
	} else {
		ginkgo.Context("when running without AppArmor", func() {
			f := framework.NewDefaultFramework("apparmor-test")

			ginkgo.It("should reject a pod with an AppArmor profile", func() {
				status := runAppArmorTest(f, false, v1.AppArmorBetaProfileRuntimeDefault)
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

	cmd := exec.Command("apparmor_parser", "-r", "-W", f.Name())
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr
	out, err := cmd.Output()
	// apparmor_parser does not always return an error code, so consider any stderr output an error.
	if err != nil || stderr.Len() > 0 {
		if stderr.Len() > 0 {
			klog.Warning(stderr.String())
		}
		if len(out) > 0 {
			klog.Infof("apparmor_parser: %s", out)
		}
		return fmt.Errorf("failed to load profiles: %v", err)
	}
	klog.V(2).Infof("Loaded profiles: %v", out)
	return nil
}

func runAppArmorTest(f *framework.Framework, shouldRun bool, profile string) v1.PodStatus {
	pod := createPodWithAppArmor(f, profile)
	if shouldRun {
		// The pod needs to start before it stops, so wait for the longer start timeout.
		framework.ExpectNoError(e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(
			f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
	} else {
		// Pod should remain in the pending state. Wait for the Reason to be set to "AppArmor".
		fieldSelector := fields.OneTermEqualSelector("metadata.name", pod.Name).String()
		w := &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.FieldSelector = fieldSelector
				return f.PodClient().List(context.TODO(), options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.FieldSelector = fieldSelector
				return f.PodClient().Watch(context.TODO(), options)
			},
		}
		preconditionFunc := func(store cache.Store) (bool, error) {
			_, exists, err := store.Get(&metav1.ObjectMeta{Namespace: pod.Namespace, Name: pod.Name})
			if err != nil {
				return true, err
			}
			if !exists {
				// We need to make sure we see the object in the cache before we start waiting for events
				// or we would be waiting for the timeout if such object didn't exist.
				return true, apierrors.NewNotFound(v1.Resource("pods"), pod.Name)
			}

			return false, nil
		}
		ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), framework.PodStartTimeout)
		defer cancel()
		_, err := watchtools.UntilWithSync(ctx, w, &v1.Pod{}, preconditionFunc, func(e watch.Event) (bool, error) {
			switch e.Type {
			case watch.Deleted:
				return false, apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, pod.Name)
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
	p, err := f.PodClient().Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return p.Status
}

func createPodWithAppArmor(f *framework.Framework, profile string) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("test-apparmor-%s", strings.Replace(profile, "/", "-", -1)),
			Annotations: map[string]string{
				v1.AppArmorBetaContainerAnnotationKeyPrefix + "test": profile,
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
	framework.ExpectEqual(status.Phase, v1.PodPending, args...)
	framework.ExpectEqual(status.Reason, "AppArmor", args...)
	gomega.Expect(status.Message).To(gomega.ContainSubstring("AppArmor"), args...)
	framework.ExpectEqual(status.ContainerStatuses[0].State.Waiting.Reason, "Blocked", args...)
}

func isAppArmorEnabled() bool {
	// TODO(tallclair): Pass this through the image setup rather than hardcoding.
	if strings.Contains(framework.TestContext.NodeName, "-gci-dev-") {
		gciVersionRe := regexp.MustCompile("-gci-dev-([0-9]+)-")
		matches := gciVersionRe.FindStringSubmatch(framework.TestContext.NodeName)
		if len(matches) == 2 {
			version, err := strconv.Atoi(matches[1])
			if err != nil {
				klog.Errorf("Error parsing GCI version from NodeName %q: %v", framework.TestContext.NodeName, err)
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
