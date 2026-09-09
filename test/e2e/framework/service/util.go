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
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
)

// TestReachableHTTP tests that the given host serves HTTP on the given port.
func TestReachableHTTP(ctx context.Context, host string, port int, timeout time.Duration) {
	TestReachableHTTPWithRetriableErrorCodes(ctx, host, port, []int{}, timeout)
}

// TestReachableHTTPWithRetriableErrorCodes tests that the given host serves HTTP on the given port with the given retriableErrCodes.
func TestReachableHTTPWithRetriableErrorCodes(ctx context.Context, host string, port int, retriableErrCodes []int, timeout time.Duration) {
	pollfn := func(ctx context.Context) (bool, error) {
		result := e2enetwork.PokeHTTP(host, port, "/echo?msg=hello",
			&e2enetwork.HTTPPokeParams{
				BodyContains:   "hello",
				RetriableCodes: retriableErrCodes,
			})
		if result.Status == e2enetwork.HTTPSuccess {
			return true, nil
		}
		return false, nil // caller can retry
	}

	if err := wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, true, pollfn); err != nil {
		if wait.Interrupted(err) {
			framework.Failf("Could not reach HTTP service through %v:%v after %v", host, port, timeout)
		} else {
			framework.Failf("Failed to reach HTTP service through %v:%v: %v", host, port, err)
		}
	}
}
