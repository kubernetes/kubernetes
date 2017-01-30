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

package test_internalclientset

import (
	"testing"

	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
)

func ClientSetRateLimiterTest(t *testing.T) {
	rateLimiter := flowcontrol.NewTokenBucketRateLimiter(1.0, 10)
	config := restclient.Config{
		RateLimiter: rateLimiter,
	}
	if err := restclient.SetKubernetesDefaults(&config); err != nil {
		t.Errorf("setting defaults failed for %#v: %v", config, err)
	}
	clientSet, err := NewForConfig(&config)
	if err != nil {
		t.Errorf("creating clientset for config %v failed: %v", config, err)
	}
	testGroupThrottler := clientSet.Testgroup().RESTClient().GetRateLimiter()

	if rateLimiter != testGroupThrottler {
		t.Errorf("Clients in client set should use rateLimiter passed in config:\noriginal: %v\ntestGroup: %v", rateLimiter, testGroupThrottler)
	}
}
