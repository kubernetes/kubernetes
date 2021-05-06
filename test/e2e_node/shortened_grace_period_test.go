/*
Copyright 2021 The Kubernetes Authors.

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
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("Shortened Grace Period", func() {
	f := framework.NewDefaultFramework("shortened-grace-period")
	ginkgo.Context("When repeatedly deleting pods", func() {
		ginkgo.It("should be deleted immediately", func() {
			const (
				gracePeriod      = 600
				gracePeriodShort = 5
			)

			nodeName := getNodeName(f)
			podName := "test"

			f.PodClient().CreateSync(getGracePeriodTestPod(podName, nodeName, gracePeriod))
			err := f.PodClient().Delete(context.TODO(), podName, *metav1.NewDeleteOptions(gracePeriod))
			framework.ExpectNoError(err)

			start := time.Now()
			f.PodClient().DeleteSync(podName, *metav1.NewDeleteOptions(gracePeriodShort), gracePeriod*time.Second)
			framework.ExpectEqual(time.Since(start) < gracePeriod*time.Second, true, "cannot update GracePeriodSeconds")
		})
	})
})

func getGracePeriodTestPod(name string, node string, gracePeriod int64) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   busyboxImage,
					Command: []string{"sh", "-c"},
					Args: []string{`
_term() {
	echo "Caught SIGTERM signal!"
	sleep infinity
}
trap _term SIGTERM
sleep infinity
`},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
			NodeName:                      node,
		},
	}
	return pod
}
