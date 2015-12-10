/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e_node

import (
	"time"

	client "k8s.io/kubernetes/pkg/client/unversioned"
)

type RetryFn func(cl *client.Client) error

func Retry(maxWait time.Duration, wait time.Duration, cl *client.Client, retry RetryFn) []error {
	errs := []error{}
	for start := time.Now(); time.Now().Before(start.Add(maxWait)); {
		if err := retry(cl); err != nil {
			errs = append(errs, err)
		} else {
			return []error{}
		}
	}
	return errs
}
