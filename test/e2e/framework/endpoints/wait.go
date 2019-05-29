/*
Copyright 2019 The Kubernetes Authors.

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

/*
This soak tests places a specified number of pods on each node and then
repeatedly sends queries to a service running on these pods via
a serivce
*/

package endpoints

import (
	"fmt"
	"time"

	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

const (
	// registerTimeout is how long to wait for an endpoint to be registered.
	registerTimeout = time.Minute
)

// WaitForEndpoint waits for the specified endpoint to be ready.
func WaitForEndpoint(c clientset.Interface, ns, name string) error {
	for t := time.Now(); time.Since(t) < registerTimeout; time.Sleep(framework.Poll) {
		endpoint, err := c.CoreV1().Endpoints(ns).Get(name, metav1.GetOptions{})
		if apierrs.IsNotFound(err) {
			e2elog.Logf("Endpoint %s/%s is not ready yet", ns, name)
			continue
		}
		framework.ExpectNoError(err, "Failed to get endpoints for %s/%s", ns, name)
		if len(endpoint.Subsets) == 0 || len(endpoint.Subsets[0].Addresses) == 0 {
			e2elog.Logf("Endpoint %s/%s is not ready yet", ns, name)
			continue
		}
		return nil
	}
	return fmt.Errorf("failed to get endpoints for %s/%s", ns, name)
}
