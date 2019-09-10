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

package monitoring

import (
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/config"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"

	"github.com/onsi/ginkgo"
)

var cadvisor struct {
	MaxRetries    int           `default:"6"`
	SleepDuration time.Duration `default:"10000ms"`
}
var _ = config.AddOptions(&cadvisor, "instrumentation.monitoring.cadvisor")

var _ = instrumentation.SIGDescribe("Cadvisor", func() {

	f := framework.NewDefaultFramework("cadvisor")

	ginkgo.It("should be healthy on every node.", func() {
		CheckCadvisorHealthOnAllNodes(f.ClientSet, 5*time.Minute)
	})
})

// CheckCadvisorHealthOnAllNodes check cadvisor health via kubelet endpoint
func CheckCadvisorHealthOnAllNodes(c clientset.Interface, timeout time.Duration) {
	// It should be OK to list unschedulable Nodes here.
	ginkgo.By("getting list of nodes")
	nodeList, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err)
	var errors []error

	maxRetries := cadvisor.MaxRetries
	for {
		errors = []error{}
		for _, node := range nodeList.Items {
			// cadvisor is not accessible directly unless its port (4194 by default) is exposed.
			// Here, we access '/stats/' REST endpoint on the kubelet which polls cadvisor internally.
			statsResource := fmt.Sprintf("api/v1/nodes/%s/proxy/stats/", node.Name)
			ginkgo.By(fmt.Sprintf("Querying stats from node %s using url %s", node.Name, statsResource))
			_, err = c.CoreV1().RESTClient().Get().AbsPath(statsResource).Timeout(timeout).Do().Raw()
			if err != nil {
				errors = append(errors, err)
			}
		}
		if len(errors) == 0 {
			return
		}
		if maxRetries--; maxRetries <= 0 {
			break
		}
		framework.Logf("failed to retrieve kubelet stats -\n %v", errors)
		time.Sleep(cadvisor.SleepDuration)
	}
	framework.Failf("Failed after retrying %d times for cadvisor to be healthy on all nodes. Errors:\n%v", maxRetries, errors)
}
