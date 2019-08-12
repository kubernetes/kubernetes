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

package service

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// WaitForServiceResponding waits for the service to be responding.
func WaitForServiceResponding(c clientset.Interface, ns, name string) error {
	ginkgo.By(fmt.Sprintf("trying to dial the service %s.%s via the proxy", ns, name))

	return wait.PollImmediate(framework.Poll, RespondingTimeout, func() (done bool, err error) {
		proxyRequest, errProxy := GetServicesProxyRequest(c, c.CoreV1().RESTClient().Get())
		if errProxy != nil {
			e2elog.Logf("Failed to get services proxy request: %v:", errProxy)
			return false, nil
		}

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		body, err := proxyRequest.Namespace(ns).
			Context(ctx).
			Name(name).
			Do().
			Raw()
		if err != nil {
			if ctx.Err() != nil {
				e2elog.Failf("Failed to GET from service %s: %v", name, err)
				return true, err
			}
			e2elog.Logf("Failed to GET from service %s: %v:", name, err)
			return false, nil
		}
		got := string(body)
		if len(got) == 0 {
			e2elog.Logf("Service %s: expected non-empty response", name)
			return false, err // stop polling
		}
		e2elog.Logf("Service %s: found nonempty answer: %s", name, got)
		return true, nil
	})
}
