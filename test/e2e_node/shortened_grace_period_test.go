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

package e2enode

import (
	"bytes"
	"context"
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
	"strings"
	"time"
)

var _ = SIGDescribe("Shortened Grace Period", func() {
	f := framework.NewDefaultFramework("shortened-grace-period")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
	ginkgo.Context("When repeatedly deleting pods", func() {
		var podClient *e2epod.PodClient
		var dc dynamic.Interface
		var ns string
		var podName = "test"
		var ctx = context.Background()
		var rcResource = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "Pod"}
		ginkgo.BeforeEach(func() {
			ns = f.Namespace.Name
			dc = f.DynamicClient
			podClient = e2epod.NewPodClient(f)
		})
		ginkgo.It("shorter grace period of a second command overrides the longer grace period of a first command", func() {
			expectedWatchEvents := []watch.Event{
				{Type: watch.Deleted},
			}
			var exitCode int32
			callback := func(retryWatcher *watchtools.RetryWatcher) (actualWatchEvents []watch.Event) {
				pod, err := podClient.Get(ctx, podName, metav1.GetOptions{})
				framework.ExpectNoError(err, "failed to get pod %q", podName)
				//Verify exit code
				exitCode = pod.Status.ContainerStatuses[0].State.Terminated.ExitCode
				framework.ExpectNoError(err, "failed to get most recent container exit code for pod %q", podName)
				framework.ExpectEqual(exitCode, int32(0), "unexpected container exit code for pod %q.code is %d", podName, exitCode)
				// Get pod logs.
				logs, err := podClient.GetLogs(podName, &v1.PodLogOptions{}).Stream(ctx)
				framework.ExpectNoError(err, "failed to get pod logs")
				defer logs.Close()
				buf := new(bytes.Buffer)
				buf.ReadFrom(logs)
				podLogs := buf.String()
				// Verify the number of SIGINT
				SIGINT1 := strings.Count("SIGINT 1", podLogs)
				SIGINT2 := strings.Count("SIGINT 2", podLogs)
				framework.ExpectEqual(SIGINT1, int32(1), "Unexpected SIGINT 1 exit volume")
				framework.ExpectEqual(SIGINT2, int32(1), "Unexpected SIGINT 1 exit volume")
				return nil
			}
			framework.WatchEventSequenceVerifier(ctx, dc, rcResource, ns, podName, metav1.ListOptions{LabelSelector: "test=true"}, expectedWatchEvents, callback, func() (err error) {
				const (
					gracePeriod      = 10000
					gracePeriodShort = 20
				)
				podClient.CreateSync(ctx, getGracePeriodTestPod(podName, gracePeriod))
				err = podClient.Delete(ctx, podName, *metav1.NewDeleteOptions(gracePeriod))
				podClient.DeleteSync(ctx, podName, *metav1.NewDeleteOptions(gracePeriodShort), gracePeriod*time.Second)
				return err
			})
		})
	})
})

func getGracePeriodTestPod(name string, gracePeriod int64) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"test": "true",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   busyboxImage,
					Command: []string{"sh", "-c"},
					Args: []string{`
term() {
  if [ "$COUNT" -eq 0 ]; then
    echo "SIGINT 1" >> /dev/termination-log
  elif [ "$COUNT" -eq 1 ]; then
    echo "SIGINT 2" >> /dev/termination-log
    sleep 5
    exit 0
  else
    echo "SIGINT $COUNT" >> /dev/termination-log
    exit 1
  fi
  COUNT=$((COUNT + 1))
}
COUNT=0
trap term SIGINT
while true; do
  sleep 1
done
`},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
	return pod
}
