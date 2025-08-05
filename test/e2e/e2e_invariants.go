/*
Copyright 2025 The Kubernetes Authors.

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
	"context"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"

	"github.com/onsi/ginkgo/v2"
)

// checks for api-server metrics that indicate an internal bug has occurred
// please speak to SIG-Testing leads before adding anything here

var _ = ginkgo.ReportAfterSuite("Invariant Metrics", func(report ginkgo.Report) {
	checkInvariants(context.TODO())
})

func checkInvariants(ctx context.Context) {
	// grab metrics
	config, err := framework.LoadConfig()
	if err != nil {
		framework.Failf("error loading client config: %v", err)
	}
	c, err := clientset.NewForConfig(config)
	if err != nil {
		framework.Failf("error loading client config: %v", err)
	}
	grabber, err := e2emetrics.NewMetricsGrabber(ctx, c, nil, config, false, false, false, true, false, false)
	if err != nil {
		framework.Failf("error creating metrics grabber: %v", err)
	}
	metrics, err := grabber.GrabFromAPIServer(ctx)
	if err != nil {
		framework.Failf("error grabbing api-server metrics: %v", err)
	}

	// check metrics
	// please speak to SIG-Testing leads before adding anything here
	checkDeclarativeValidation(metrics)
}

func checkDeclarativeValidation(metrics e2emetrics.APIServerMetrics) {
	samples := metrics["apiserver_validation_declarative_validation_mismatch_total"]
	for _, sample := range samples {
		// TODO: this is wrong, but we are just proving out the concept
		if sample.Value <= 0 {
			framework.Failf("apiserver_validation_declarative_validation_mismatch_total > 0: %v\nThis means we have a bug in apiserver, file an issue and /assign thockin", sample.Value)
		}
	}
}
