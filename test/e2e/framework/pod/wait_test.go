/*
Copyright 2022 The Kubernetes Authors.

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

package pod_test

import (
	"context"
	"fmt"
	"regexp"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/output"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	_ "k8s.io/kubernetes/test/utils/format" // activate YAML object dumps
)

// The line number of the following code is checked in TestFailureOutput below.
// Be careful when moving it around or changing the import statements above.
// Here are some intentionally blank lines that can be removed to compensate
// for future additional import statements.
//
//
//
//
//
//
// This must be line #50.

var _ = ginkgo.Describe("pod", func() {
	ginkgo.It("not found, must exist", func(ctx context.Context) {
		gomega.Eventually(ctx, framework.HandleRetry(getNoSuchPod)).WithTimeout(timeout).Should(e2epod.BeInPhase(v1.PodRunning))
	})

	ginkgo.It("not found, retry", func(ctx context.Context) {
		framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, clientSet, "no-such-pod", "default", timeout))
	})

	ginkgo.It("not found, retry with wrappers", func(ctx context.Context) {
		gomega.Eventually(ctx, framework.RetryNotFound(framework.HandleRetry(getNoSuchPod))).WithTimeout(timeout).Should(e2epod.BeInPhase(v1.PodRunning))
	})

	ginkgo.It("not found, retry with inverted wrappers", func(ctx context.Context) {
		gomega.Eventually(ctx, framework.HandleRetry(framework.RetryNotFound(getNoSuchPod))).WithTimeout(timeout).Should(e2epod.BeInPhase(v1.PodRunning))
	})

	ginkgo.It("not running", func(ctx context.Context) {
		ginkgo.By(fmt.Sprintf("waiting for pod %s to run", podName))
		framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, clientSet, podName, podNamespace, timeout))
	})

	ginkgo.It("failed", func(ctx context.Context) {
		framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, clientSet, failedPodName, podNamespace, timeout))
	})

	ginkgo.It("gets reported with API error", func(ctx context.Context) {
		called := false
		getPod := func(ctx context.Context) (*v1.Pod, error) {
			if called {
				ginkgo.By("returning fake API error")
				return nil, apierrors.NewTooManyRequests("fake API error", 10)
			}
			called = true
			pod, err := clientSet.CoreV1().Pods(podNamespace).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
			ginkgo.By("returning pod")
			return pod, err
		}
		gomega.Eventually(ctx, framework.HandleRetry(getPod)).WithTimeout(5 * timeout).Should(e2epod.BeInPhase(v1.PodRunning))
	})
})

func getNoSuchPod(ctx context.Context) (*v1.Pod, error) {
	return clientSet.CoreV1().Pods("default").Get(ctx, "no-such-pod", metav1.GetOptions{})
}

const (
	podName       = "pending-pod"
	podNamespace  = "default"
	failedPodName = "failed-pod"
	timeout       = time.Second
)

var (
	clientSet = fake.NewSimpleClientset(
		&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: podNamespace}, Status: v1.PodStatus{Phase: v1.PodPending}},
		&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: failedPodName, Namespace: podNamespace}, Status: v1.PodStatus{Phase: v1.PodFailed}},
	)
)

func TestFailureOutput(t *testing.T) {

	expected := output.TestResult{
		NormalizeOutput: func(in string) string {
			return regexp.MustCompile(`wait.go:[[:digit:]]*`).ReplaceAllString(in, `wait.go`)
		},
		Suite: reporters.JUnitTestSuite{
			Tests:    7,
			Failures: 7,
			Errors:   0,
			Disabled: 0,
			Skipped:  0,
			TestCases: []reporters.JUnitTestCase{
				{
					Name:   "[It] pod not found, must exist",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] Told to stop trying after <after>.
Unexpected final error while getting *v1.Pod: pods "no-such-pod" not found
In [It] at: wait_test.go:54 <time>
`,
					},
					SystemErr: `> Enter [It] not found, must exist - wait_test.go:53 <time>
[FAILED] Told to stop trying after <after>.
Unexpected final error while getting *v1.Pod: pods "no-such-pod" not found
In [It] at: wait_test.go:54 <time>
< Exit [It] not found, must exist - wait_test.go:53 <time>
`,
				},
				{
					Name:   "[It] pod not found, retry",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] Timed out after <after>.
The function passed to Eventually returned the following error:
    <framework.transientError>: 
    pods "no-such-pod" not found
    {
        error: <*errors.StatusError>{
            ErrStatus: {
                TypeMeta: {Kind: "", APIVersion: ""},
                ListMeta: {
                    SelfLink: "",
                    ResourceVersion: "",
                    Continue: "",
                    RemainingItemCount: nil,
                },
                Status: "Failure",
                Message: "pods \"no-such-pod\" not found",
                Reason: "NotFound",
                Details: {Name: "no-such-pod", Group: "", Kind: "pods", UID: "", Causes: nil, RetryAfterSeconds: 0},
                Code: 404,
            },
        },
    }
In [It] at: wait_test.go:58 <time>
`,
					},
					SystemErr: `> Enter [It] not found, retry - wait_test.go:57 <time>
<klog> wait_test.go:58] Failed inside E2E framework:
    k8s.io/kubernetes/test/e2e/framework/pod.WaitTimeoutForPodRunningInNamespace()
    	wait.go
    k8s.io/kubernetes/test/e2e/framework/pod_test.glob..func1.2()
    	wait_test.go:58
[FAILED] Timed out after <after>.
The function passed to Eventually returned the following error:
    <framework.transientError>: 
    pods "no-such-pod" not found
    {
        error: <*errors.StatusError>{
            ErrStatus: {
                TypeMeta: {Kind: "", APIVersion: ""},
                ListMeta: {
                    SelfLink: "",
                    ResourceVersion: "",
                    Continue: "",
                    RemainingItemCount: nil,
                },
                Status: "Failure",
                Message: "pods \"no-such-pod\" not found",
                Reason: "NotFound",
                Details: {Name: "no-such-pod", Group: "", Kind: "pods", UID: "", Causes: nil, RetryAfterSeconds: 0},
                Code: 404,
            },
        },
    }
In [It] at: wait_test.go:58 <time>
< Exit [It] not found, retry - wait_test.go:57 <time>
`,
				},
				{
					Name:   "[It] pod not found, retry with wrappers",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] Timed out after <after>.
The function passed to Eventually returned the following error:
    <framework.transientError>: 
    pods "no-such-pod" not found
    {
        error: <*errors.StatusError>{
            ErrStatus: {
                TypeMeta: {Kind: "", APIVersion: ""},
                ListMeta: {
                    SelfLink: "",
                    ResourceVersion: "",
                    Continue: "",
                    RemainingItemCount: nil,
                },
                Status: "Failure",
                Message: "pods \"no-such-pod\" not found",
                Reason: "NotFound",
                Details: {Name: "no-such-pod", Group: "", Kind: "pods", UID: "", Causes: nil, RetryAfterSeconds: 0},
                Code: 404,
            },
        },
    }
In [It] at: wait_test.go:62 <time>
`,
					},
					SystemErr: `> Enter [It] not found, retry with wrappers - wait_test.go:61 <time>
[FAILED] Timed out after <after>.
The function passed to Eventually returned the following error:
    <framework.transientError>: 
    pods "no-such-pod" not found
    {
        error: <*errors.StatusError>{
            ErrStatus: {
                TypeMeta: {Kind: "", APIVersion: ""},
                ListMeta: {
                    SelfLink: "",
                    ResourceVersion: "",
                    Continue: "",
                    RemainingItemCount: nil,
                },
                Status: "Failure",
                Message: "pods \"no-such-pod\" not found",
                Reason: "NotFound",
                Details: {Name: "no-such-pod", Group: "", Kind: "pods", UID: "", Causes: nil, RetryAfterSeconds: 0},
                Code: 404,
            },
        },
    }
In [It] at: wait_test.go:62 <time>
< Exit [It] not found, retry with wrappers - wait_test.go:61 <time>
`,
				},
				{
					Name:   "[It] pod not found, retry with inverted wrappers",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Type: "failed",
						Description: `[FAILED] Timed out after <after>.
The function passed to Eventually returned the following error:
    <framework.transientError>: 
    pods "no-such-pod" not found
    {
        error: <*errors.StatusError>{
            ErrStatus: {
                TypeMeta: {Kind: "", APIVersion: ""},
                ListMeta: {
                    SelfLink: "",
                    ResourceVersion: "",
                    Continue: "",
                    RemainingItemCount: nil,
                },
                Status: "Failure",
                Message: "pods \"no-such-pod\" not found",
                Reason: "NotFound",
                Details: {Name: "no-such-pod", Group: "", Kind: "pods", UID: "", Causes: nil, RetryAfterSeconds: 0},
                Code: 404,
            },
        },
    }
In [It] at: wait_test.go:66 <time>
`,
					},
					SystemErr: `> Enter [It] not found, retry with inverted wrappers - wait_test.go:65 <time>
[FAILED] Timed out after <after>.
The function passed to Eventually returned the following error:
    <framework.transientError>: 
    pods "no-such-pod" not found
    {
        error: <*errors.StatusError>{
            ErrStatus: {
                TypeMeta: {Kind: "", APIVersion: ""},
                ListMeta: {
                    SelfLink: "",
                    ResourceVersion: "",
                    Continue: "",
                    RemainingItemCount: nil,
                },
                Status: "Failure",
                Message: "pods \"no-such-pod\" not found",
                Reason: "NotFound",
                Details: {Name: "no-such-pod", Group: "", Kind: "pods", UID: "", Causes: nil, RetryAfterSeconds: 0},
                Code: 404,
            },
        },
    }
In [It] at: wait_test.go:66 <time>
< Exit [It] not found, retry with inverted wrappers - wait_test.go:65 <time>
`,
				},
				{
					Name:   "[It] pod not running",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Description: `[FAILED] Timed out after <after>.
Expected Pod to be in <v1.PodPhase>: Running
Got instead:
    <*v1.Pod>: 
        metadata:
          name: pending-pod
          namespace: default
        spec:
          containers: null
        status:
          phase: Pending
In [It] at: wait_test.go:71 <time>
`,
						Type: "failed",
					},
					SystemErr: `> Enter [It] not running - wait_test.go:69 <time>
STEP: waiting for pod pending-pod to run - wait_test.go:70 <time>
<klog> wait_test.go:71] Failed inside E2E framework:
    k8s.io/kubernetes/test/e2e/framework/pod.WaitTimeoutForPodRunningInNamespace()
    	wait.go
    k8s.io/kubernetes/test/e2e/framework/pod_test.glob..func1.5()
    	wait_test.go:71
[FAILED] Timed out after <after>.
Expected Pod to be in <v1.PodPhase>: Running
Got instead:
    <*v1.Pod>: 
        metadata:
          name: pending-pod
          namespace: default
        spec:
          containers: null
        status:
          phase: Pending
In [It] at: wait_test.go:71 <time>
< Exit [It] not running - wait_test.go:69 <time>
`,
				},
				{
					Name:   "[It] pod failed",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Description: `[FAILED] Told to stop trying after <after>.
Expected pod to reach phase "Running", got final phase "Failed" instead:
    <*v1.Pod>: 
        metadata:
          name: failed-pod
          namespace: default
        spec:
          containers: null
        status:
          phase: Failed
In [It] at: wait_test.go:75 <time>
`,
						Type: "failed",
					},
					SystemErr: `> Enter [It] failed - wait_test.go:74 <time>
<klog> wait_test.go:75] Failed inside E2E framework:
    k8s.io/kubernetes/test/e2e/framework/pod.WaitTimeoutForPodRunningInNamespace()
    	wait.go
    k8s.io/kubernetes/test/e2e/framework/pod_test.glob..func1.6()
    	wait_test.go:75
[FAILED] Told to stop trying after <after>.
Expected pod to reach phase "Running", got final phase "Failed" instead:
    <*v1.Pod>: 
        metadata:
          name: failed-pod
          namespace: default
        spec:
          containers: null
        status:
          phase: Failed
In [It] at: wait_test.go:75 <time>
< Exit [It] failed - wait_test.go:74 <time>
`,
				},
				{
					Name:   "[It] pod gets reported with API error",
					Status: "failed",
					Failure: &reporters.JUnitFailure{
						Description: `[FAILED] Timed out after <after>.
The function passed to Eventually returned the following error:
    <*errors.StatusError>: 
    fake API error
    {
        ErrStatus: 
            code: 429
            details:
              retryAfterSeconds: 10
            message: fake API error
            metadata: {}
            reason: TooManyRequests
            status: Failure,
    }
At one point, however, the function did return successfully.
Yet, Eventually failed because the matcher was not satisfied:
Expected Pod to be in <v1.PodPhase>: Running
Got instead:
    <*v1.Pod>: 
        metadata:
          name: pending-pod
          namespace: default
        spec:
          containers: null
        status:
          phase: Pending
In [It] at: wait_test.go:93 <time>
`,
						Type: "failed",
					},
					SystemErr: `> Enter [It] gets reported with API error - wait_test.go:78 <time>
STEP: returning pod - wait_test.go:90 <time>
STEP: returning fake API error - wait_test.go:82 <time>
[FAILED] Timed out after <after>.
The function passed to Eventually returned the following error:
    <*errors.StatusError>: 
    fake API error
    {
        ErrStatus: 
            code: 429
            details:
              retryAfterSeconds: 10
            message: fake API error
            metadata: {}
            reason: TooManyRequests
            status: Failure,
    }
At one point, however, the function did return successfully.
Yet, Eventually failed because the matcher was not satisfied:
Expected Pod to be in <v1.PodPhase>: Running
Got instead:
    <*v1.Pod>: 
        metadata:
          name: pending-pod
          namespace: default
        spec:
          containers: null
        status:
          phase: Pending
In [It] at: wait_test.go:93 <time>
< Exit [It] gets reported with API error - wait_test.go:78 <time>
`,
				},
			},
		},
	}
	output.TestGinkgoOutput(t, expected)
}
