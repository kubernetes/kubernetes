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

package e2e

import (
	"fmt"
	"time"

	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// TODO(crassirostris): Remove Flaky once test is stable
var _ = framework.KubeDescribe("Cluster level logging using GCL [Flaky]", func() {
	f := framework.NewDefaultFramework("gcl-logging")

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "gke")
	})

	It("should check that logs from containers are ingested in GCL", func() {
		podName := "synthlogger"

		gclLogsProvider, err := newGclLogsProvider(f)
		framework.ExpectNoError(err, "Failed to create GCL logs provider")

		err = gclLogsProvider.EnsureWorking()
		framework.ExpectNoError(err, "GCL is not working")

		By("Running synthetic logger")
		pod := createLoggingPod(f, podName, 100, 1*time.Second)
		defer f.PodClient().Delete(podName, &meta_v1.DeleteOptions{})
		err = framework.WaitForPodSuccessInNamespace(f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, fmt.Sprintf("Should've successfully waited for pod %s to succeed", podName))

		By("Waiting for logs to ingest")
		config := &loggingTestConfig{
			LogsProvider:              gclLogsProvider,
			Pods:                      []*loggingPod{pod},
			IngestionTimeout:          10 * time.Minute,
			MaxAllowedLostFraction:    0,
			MaxAllowedFluentdRestarts: 0,
		}
		err = waitForLogsIngestion(f, config)
		framework.ExpectNoError(err, "Failed to ingest logs")
	})
})
