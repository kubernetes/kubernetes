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
	"context"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"

	"github.com/onsi/ginkgo"
)

var _ = instrumentation.SIGDescribe("Kibana Logging Instances Is Alive [Feature:Elasticsearch]", func() {
	f := framework.NewDefaultFramework("kibana-logging")

	ginkgo.BeforeEach(func() {
		// TODO: For now assume we are only testing cluster logging with Elasticsearch
		// and Kibana on GCE. Once we are sure that Elasticsearch and Kibana cluster level logging
		// works for other providers we should widen this scope of this test.
		e2eskipper.SkipUnlessProviderIs("gce")
	})

	ginkgo.It("should check that the Kibana logging instance is alive", func() {
		ClusterLevelLoggingWithKibana(f)
	})
})

const (
	kibanaKey   = "k8s-app"
	kibanaValue = "kibana-logging"
)

// ClusterLevelLoggingWithKibana is an end to end test that checks to see if Kibana is alive.
func ClusterLevelLoggingWithKibana(f *framework.Framework) {
	const pollingInterval = 10 * time.Second
	const pollingTimeout = 20 * time.Minute

	// Check for the existence of the Kibana service.
	ginkgo.By("Checking the Kibana service exists.")
	s := f.ClientSet.CoreV1().Services(metav1.NamespaceSystem)
	// Make a few attempts to connect. This makes the test robust against
	// being run as the first e2e test just after the e2e cluster has been created.
	err := wait.Poll(pollingInterval, pollingTimeout, func() (bool, error) {
		if _, err := s.Get("kibana-logging", metav1.GetOptions{}); err != nil {
			framework.Logf("Kibana is unreachable: %v", err)
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err)

	// Wait for the Kibana pod(s) to enter the running state.
	ginkgo.By("Checking to make sure the Kibana pods are running")
	label := labels.SelectorFromSet(labels.Set(map[string]string{kibanaKey: kibanaValue}))
	options := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).List(options)
	framework.ExpectNoError(err)
	for _, pod := range pods.Items {
		err = e2epod.WaitForPodRunningInNamespace(f.ClientSet, &pod)
		framework.ExpectNoError(err)
	}

	ginkgo.By("Checking to make sure we get a response from the Kibana UI.")
	err = wait.Poll(pollingInterval, pollingTimeout, func() (bool, error) {
		req, err := e2eservice.GetServicesProxyRequest(f.ClientSet, f.ClientSet.CoreV1().RESTClient().Get())
		if err != nil {
			framework.Logf("Failed to get services proxy request: %v", err)
			return false, nil
		}

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		_, err = req.Namespace(metav1.NamespaceSystem).
			Context(ctx).
			Name("kibana-logging").
			DoRaw()
		if err != nil {
			framework.Logf("Proxy call to kibana-logging failed: %v", err)
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err)
}
