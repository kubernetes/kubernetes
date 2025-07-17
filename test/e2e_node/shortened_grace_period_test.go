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
	"context"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe(framework.WithNodeConformance(), "Shortened Grace Period", func() {
	f := framework.NewDefaultFramework("shortened-grace-period")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	ginkgo.Context("When repeatedly deleting pods", func() {
		var podClient *e2epod.PodClient
		var dc dynamic.Interface
		var ns string
		var podName = "test-shortened-grace"
		var ctx = context.Background()
		var rcResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
		const gracePeriodShort = 20 // 20s for more room
		ginkgo.BeforeEach(func() {
			ns = f.Namespace.Name
			dc = f.DynamicClient
			podClient = e2epod.NewPodClient(f)
		})
		ginkgo.It("should deliver exactly two SIGTERM to the container and exit 0", func() {
			testRcNamespace := ns
			expectedWatchEvents := []watch.Event{
				{Type: watch.Modified},
				{Type: watch.Deleted},
			}
			callback := func(retryWatcher *watchtools.RetryWatcher) (actualWatchEvents []watch.Event) {
				start := time.Now()
				podClient.CreateSync(ctx, getGracePeriodTestPodSIGTERM(podName, testRcNamespace, 999))
				// Wait for the container to start
				time.Sleep(2 * time.Second)
				w, err := podClient.Watch(context.TODO(), metav1.ListOptions{LabelSelector: "test-shortened-grace=true"})
				framework.ExpectNoError(err, "failed to watch")
				// First Delete with 999s grace period
				err = podClient.Delete(ctx, podName, *metav1.NewDeleteOptions(999))
				framework.ExpectNoError(err, "failed to delete pod (first)")
				// Wait 1 second, then Delete again with 99s grace period
				time.Sleep(1 * time.Second)
				err = podClient.Delete(ctx, podName, *metav1.NewDeleteOptions(99))
				framework.ExpectNoError(err, "failed to delete pod (second)")
				// Wait 5 seconds to ensure signal handling and log output
				time.Sleep(5 * time.Second)
				// Retrieve logs from the pod's main container
				podLogs, err := podClient.GetLogs(podName, &v1.PodLogOptions{}).DoRaw(ctx)
				framework.ExpectNoError(err, "failed to get pod logs")
				framework.Logf("Pod logs: %q", string(podLogs))
				// Check logs: must contain SIGINT 1 and SIGINT 2
				if !strings.Contains(string(podLogs), "SIGINT 1") || !strings.Contains(string(podLogs), "SIGINT 2") {
					framework.Failf("unexpected pod logs: %q", string(podLogs))
				}
				// Wait for the pod to be fully deleted
				ctxUntil, cancel := context.WithTimeout(ctx, 30*time.Second)
				defer cancel()
				_, err = watchtools.UntilWithoutRetry(ctxUntil, w, func(watchEvent watch.Event) (bool, error) {
					actualWatchEvents = append(actualWatchEvents, watchEvent)
					return watchEvent.Type == watch.Deleted, nil
				})
				framework.ExpectNoError(err, "Wait until pod deleted should not return an error")
				// Check exit code
				status, err := podClient.Get(ctx, podName, metav1.GetOptions{})
				if err == nil && len(status.Status.ContainerStatuses) > 0 {
					exitCode := status.Status.ContainerStatuses[0].State.Terminated.ExitCode
					if exitCode != 0 {
						framework.Failf("unexpected exit code: %d", exitCode)
					}
				}
				// Log latency
				latency := time.Since(start)
				framework.Logf("Pod delete to final deletion latency: %v", latency)
				return expectedWatchEvents
			}
			framework.WatchEventSequenceVerifier(ctx, dc, rcResource, ns, podName, metav1.ListOptions{LabelSelector: "test-shortened-grace=true"}, expectedWatchEvents, callback, func() (err error) {
				return err
			})
		})
	})
})

// getGracePeriodTestPodSIGTERM returns a pod that traps SIGTERM and counts signals, exiting 0 after two, 1 if more.
func getGracePeriodTestPodSIGTERM(name, testRcNamespace string, gracePeriod int64) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"test-shortened-grace": "true",
			},
			Namespace: testRcNamespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   busyboxImage,
					Command: []string{"sh", "-c"},
					Args: []string{`
count=0
term_handler() {
  count=$((count+1))
  if [ "$count" -eq 1 ]; then
    echo "SIGINT 1"
  elif [ "$count" -eq 2 ]; then
    echo "SIGINT 2"
    sleep 10
    exit 0
  else
    echo "SIGINT $count"
    exit 1
  fi
}
trap term_handler TERM
echo "Container started"
while true; do sleep 1; done
`},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
	return pod
}
