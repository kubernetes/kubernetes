/*
Copyright 2017 The Kubernetes Authors.

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

package logging

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var loggingSoak struct {
	Scale            int           `default:"1" usage:"number of waves of pods"`
	TimeBetweenWaves time.Duration `default:"5000ms" usage:"time to wait before dumping the next wave of pods"`
}
var _ = e2econfig.AddOptions(&loggingSoak, "instrumentation.logging.soak")

var _ = instrumentation.SIGDescribe("Logging soak [Performance]", framework.WithSlow(), framework.WithDisruptive(), func() {

	f := framework.NewDefaultFramework("logging-soak")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// Not a global constant (irrelevant outside this test), also not a parameter (if you want more logs, use --scale=).
	kbRateInSeconds := 1 * time.Second
	totalLogTime := 2 * time.Minute

	// This test is designed to run and confirm that logs are being generated at a large scale, and that they can be grabbed by the kubelet.
	// By running it repeatedly in the background, you can simulate large collections of chatty containers.
	// This can expose problems in your docker configuration (logging), log searching infrastructure, to tune deployments to match high load
	// scenarios.  TODO jayunit100 add this to the kube CI in a follow on infra patch.

	ginkgo.It(fmt.Sprintf("should survive logging 1KB every %v seconds, for a duration of %v", kbRateInSeconds, totalLogTime), func(ctx context.Context) {
		ginkgo.By(fmt.Sprintf("scaling up to %v pods per node", loggingSoak.Scale))
		defer ginkgo.GinkgoRecover()
		var wg sync.WaitGroup
		wg.Add(loggingSoak.Scale)
		for i := 0; i < loggingSoak.Scale; i++ {
			go func(i int) {
				defer wg.Done()
				defer ginkgo.GinkgoRecover()
				wave := fmt.Sprintf("wave%v", strconv.Itoa(i))
				framework.Logf("Starting logging soak, wave = %v", wave)
				RunLogPodsWithSleepOf(ctx, f, kbRateInSeconds, wave, totalLogTime)
				framework.Logf("Completed logging soak, wave %v", i)
			}(i)
			// Niceness.
			time.Sleep(loggingSoak.TimeBetweenWaves)
		}
		framework.Logf("Waiting on all %v logging soak waves to complete", loggingSoak.Scale)
		wg.Wait()
	})
})

// RunLogPodsWithSleepOf creates a pod on every node, logs continuously (with "sleep" pauses), and verifies that the log string
// was produced in each and every pod at least once.  The final arg is the timeout for the test to verify all the pods got logs.
func RunLogPodsWithSleepOf(ctx context.Context, f *framework.Framework, sleep time.Duration, podname string, timeout time.Duration) {

	nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
	framework.ExpectNoError(err)
	totalPods := len(nodes.Items)
	gomega.Expect(nodes.Items).ToNot(gomega.BeEmpty())

	kilobyte := strings.Repeat("logs-123", 128) // 8*128=1024 = 1KB of text.

	appName := "logging-soak" + podname
	podlables := e2enode.CreatePodsPerNodeForSimpleApp(
		ctx,
		f.ClientSet,
		f.Namespace.Name,
		appName,
		func(n v1.Node) v1.PodSpec {
			return v1.PodSpec{
				Containers: []v1.Container{{
					Name:  "logging-soak",
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
					Args: []string{
						"/bin/sh",
						"-c",
						fmt.Sprintf("while true ; do echo %v ; sleep %v; done", kilobyte, sleep.Seconds()),
					},
				}},
				NodeName:      n.Name,
				RestartPolicy: v1.RestartPolicyAlways,
			}
		},
		totalPods,
	)

	logSoakVerification := f.NewClusterVerification(
		f.Namespace,
		framework.PodStateVerification{
			Selectors:   podlables,
			ValidPhases: []v1.PodPhase{v1.PodRunning, v1.PodSucceeded},
			// we don't validate total log data, since there is no guarantee all logs will be stored forever.
			// instead, we just validate that some logs are being created in std out.
			Verify: func(p v1.Pod) (bool, error) {
				s, err := e2eoutput.LookForStringInLog(f.Namespace.Name, p.Name, "logging-soak", "logs-123", 1*time.Second)
				return s != "", err
			},
		},
	)

	largeClusterForgiveness := time.Duration(len(nodes.Items)/5) * time.Second // i.e. a 100 node cluster gets an extra 20 seconds to complete.
	pods, err := logSoakVerification.WaitFor(ctx, totalPods, timeout+largeClusterForgiveness)

	if err != nil {
		framework.Failf("Error in wait... %v", err)
	} else if len(pods) < totalPods {
		framework.Failf("Only got %v out of %v", len(pods), totalPods)
	}
}
