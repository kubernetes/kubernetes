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

var _ = framework.KubeDescribe("Cluster level logging using Elasticsearch [Feature:Elasticsearch]", func() {
	f := framework.NewDefaultFramework("es-logging")

	BeforeEach(func() {
		// TODO: For now assume we are only testing cluster logging with Elasticsearch
		// on GCE. Once we are sure that Elasticsearch cluster level logging
		// works for other providers we should widen this scope of this test.
		framework.SkipUnlessProviderIs("gce")
	})

	It("should check that logs from containers are ingested into Elasticsearch", func() {
		podName := "synthlogger"
		esLogsProvider, err := newEsLogsProvider(f)
		framework.ExpectNoError(err, "Failed to create Elasticsearch logs provider")

		err = esLogsProvider.Init()
		defer esLogsProvider.Cleanup()
		framework.ExpectNoError(err, "Failed to init Elasticsearch logs provider")

		err = ensureSingleFluentdOnEachNode(f, esLogsProvider.FluentdApplicationName())
		framework.ExpectNoError(err, "Fluentd deployed incorrectly")

		By("Running synthetic logger")
		pod := createLoggingPod(f, podName, "", 10*60, 10*time.Minute)
		defer f.PodClient().Delete(podName, &meta_v1.DeleteOptions{})
		err = framework.WaitForPodNameRunningInNamespace(f.ClientSet, podName, f.Namespace.Name)
		framework.ExpectNoError(err, fmt.Sprintf("Should've successfully waited for pod %s to be running", podName))

		By("Waiting for logs to ingest")
		config := &loggingTestConfig{
			LogsProvider:              esLogsProvider,
			Pods:                      []*loggingPod{pod},
			IngestionTimeout:          10 * time.Minute,
			MaxAllowedLostFraction:    0,
			MaxAllowedFluentdRestarts: 0,
		}
		framework.ExpectNoError(waitForSomeLogs(f, config), "Failed to ingest logs")
	})
})
