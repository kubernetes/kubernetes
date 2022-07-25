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
	"strings"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/output"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
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
//
//
//
//
//
//
//
//
// This must be line #52.

var _ = ginkgo.Describe("pod", func() {
	ginkgo.It("not found", func() {
		framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(clientSet, "no-such-pod", "default", timeout /* no explanation here to cover that code path */))
	})

	ginkgo.It("not running", func() {
		framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(clientSet, podName, podNamespace, timeout), "wait for pod %s running", podName /* tests printf formatting */)
	})
})

const (
	podName      = "pending-pod"
	podNamespace = "default"
	timeout      = 5 * time.Second
)

var (
	clientSet = fake.NewSimpleClientset(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: podNamespace}})
)

func TestFailureOutput(t *testing.T) {
	// Sorted by name!
	expected := output.SuiteResults{
		output.TestResult{
			Name: "pod not found",
			// "Ignoring NotFound..." will normally occur every two seconds,
			// but we reduce it to one line because it might occur less often
			// on a loaded system.
			Output: `INFO: Waiting up to 5s for pod "no-such-pod" in namespace "default" to be "running"
INFO: Ignoring NotFound error while getting pod default/no-such-pod
INFO: Unexpected error: 
    <*fmt.wrapError>: {
        msg: "error while waiting for pod default/no-such-pod to be running: pods \"no-such-pod\" not found",
        err: <*errors.StatusError>{
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
FAIL: error while waiting for pod default/no-such-pod to be running: pods "no-such-pod" not found

Full Stack Trace
k8s.io/kubernetes/test/e2e/framework/pod_test.glob..func1.1()
	wait_test.go:56
`,
			NormalizeOutput: func(output string) string {
				return trimDuplicateLines(output, "INFO: Ignoring NotFound error while getting pod default/no-such-pod")
			},
			Failure: `error while waiting for pod default/no-such-pod to be running: pods "no-such-pod" not found`,
			Stack: `k8s.io/kubernetes/test/e2e/framework/pod_test.glob..func1.1()
	wait_test.go:56`,
		},
		output.TestResult{
			Name: "pod not running",
			// "INFO: Pod ..." will normally occur every two seconds,
			// but we reduce it to one line because it might occur less often
			// on a loaded system.
			Output: `INFO: Waiting up to 5s for pod "pending-pod" in namespace "default" to be "running"
INFO: Pod "pending-pod": Phase="", Reason="", readiness=false. Elapsed: <elapsed>
INFO: Unexpected error: wait for pod pending-pod running: 
    <*pod.timeoutError>: {
        msg: "timed out while waiting for pod default/pending-pod to be running",
        observedObjects: [
            <*v1.Pod>{
                TypeMeta: {Kind: "", APIVersion: ""},
                ObjectMeta: {
                    Name: "pending-pod",
                    GenerateName: "",
                    Namespace: "default",
                    SelfLink: "",
                    UID: "",
                    ResourceVersion: "",
                    Generation: 0,
                    CreationTimestamp: {
                        Time: {wall: 0, ext: 0, loc: nil},
                    },
                    DeletionTimestamp: nil,
                    DeletionGracePeriodSeconds: nil,
                    Labels: nil,
                    Annotations: nil,
                    OwnerReferences: nil,
                    Finalizers: nil,
                    ManagedFields: nil,
                },
                Spec: {
                    Volumes: nil,
                    InitContainers: nil,
                    Containers: nil,
                    EphemeralContainers: nil,
                    RestartPolicy: "",
                    TerminationGracePeriodSeconds: nil,
                    ActiveDeadlineSeconds: nil,
                    DNSPolicy: "",
                    NodeSelector: nil,
                    ServiceAccountName: "",
                    DeprecatedServiceAccount: "",
                    AutomountServiceAccountToken: nil,
                    NodeName: "",
                    HostNetwork: false,
                    HostPID: false,
                    HostIPC: false,
                    ShareProcessNamespace: nil,
                    SecurityContext: nil,
                    ImagePullSecrets: nil,
                    Hostname: "",
                    Subdomain: "",
                    Affinity: nil,
                    SchedulerName: "",
                    Tolerations: nil,
                    HostAliases: nil,
                    PriorityClassName: "",
                    Priority: nil,
                    DNSConfig: nil,
                    ReadinessGates: nil,
                    RuntimeClassName: nil,
                    EnableServiceLinks: nil,
                    PreemptionPolicy: nil,
                    Overhead: nil,
                    TopologySpreadConstraints: nil,
                    SetHostnameAsFQDN: nil,
                    OS: nil,
                },
                Status: {
                    Phase: "",
                    Conditions: nil,
                    Message: "",
                    Reason: "",
                    NominatedNodeName: "",
                    HostIP: "",
                    PodIP: "",
                    PodIPs: nil,
                    StartTime: nil,
                    InitContainerStatuses: nil,
                    ContainerStatuses: nil,
                    QOSClass: "",
                    EphemeralContainerStatuses: nil,
                },
            },
        ],
    }
FAIL: wait for pod pending-pod running: timed out while waiting for pod default/pending-pod to be running

Full Stack Trace
k8s.io/kubernetes/test/e2e/framework/pod_test.glob..func1.2()
	wait_test.go:60
`,
			NormalizeOutput: func(output string) string {
				return trimDuplicateLines(output, `INFO: Pod "pending-pod": Phase="", Reason="", readiness=false. Elapsed: <elapsed>`)
			},
			Failure: `wait for pod pending-pod running: timed out while waiting for pod default/pending-pod to be running`,
			Stack: `k8s.io/kubernetes/test/e2e/framework/pod_test.glob..func1.2()
	wait_test.go:60`,
		},
	}

	output.TestGinkgoOutput(t, expected)
}

func trimDuplicateLines(output, prefix string) string {
	lines := strings.Split(output, "\n")
	trimming := false
	validLines := 0
	for i := 0; i < len(lines); i++ {
		if strings.HasPrefix(lines[i], prefix) {
			// Keep the first line, and only that one.
			if !trimming {
				trimming = true
				lines[validLines] = lines[i]
				validLines++
			}
		} else {
			trimming = false
			lines[validLines] = lines[i]
			validLines++
		}
	}
	return strings.Join(lines[0:validLines], "\n")
}
