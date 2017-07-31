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

package elasticsearch

import (
	"time"

	"k8s.io/kubernetes/test/e2e/framework"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"
	"k8s.io/kubernetes/test/e2e/instrumentation/logging/utils"

	"github.com/onsi/ginkgo"
)

var _ = instrumentation.SIGDescribe("Cluster level logging using Elasticsearch [Feature:Elasticsearch]", func() {
	f := framework.NewDefaultFramework("es-logging")

	ginkgo.BeforeEach(func() {
		// TODO: For now assume we are only testing cluster logging with Elasticsearch
		// on GCE. Once we are sure that Elasticsearch cluster level logging
		// works for other providers we should widen this scope of this test.
		framework.SkipUnlessProviderIs("gce")
	})

	ginkgo.It("should check that logs from containers are ingested into Elasticsearch", func() {
		ingestionInterval := 10 * time.Second
		ingestionTimeout := 10 * time.Minute

		p, err := newEsLogProvider(f)
		framework.ExpectNoError(err, "Failed to create Elasticsearch logs provider")

		err = p.Init()
		defer p.Cleanup()
		framework.ExpectNoError(err, "Failed to init Elasticsearch logs provider")

		err = utils.EnsureLoggingAgentDeployment(f, p.LoggingAgentName())
		framework.ExpectNoError(err, "Fluentd deployed incorrectly")

		pod, err := utils.StartAndReturnSelf(utils.NewRepeatingLoggingPod("synthlogger", "test"), f)
		framework.ExpectNoError(err, "Failed to start a pod")

		ginkgo.By("Waiting for logs to ingest")
		c := utils.NewLogChecker(p, utils.UntilFirstEntry, utils.JustTimeout, pod.Name())
		err = utils.WaitForLogs(c, ingestionInterval, ingestionTimeout)
		framework.ExpectNoError(err)
	})
})
