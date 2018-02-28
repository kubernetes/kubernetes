/*
Copyright 2018 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/kubectl"
)

const (
	// Parameters for retrying updates/waits with linear backoff.
	// TODO: Try to move this to exponential backoff by modifying kubectl.Scale().
	updateRetryInterval = 5 * time.Second
	updateRetryTimeout  = 1 * time.Minute
	waitRetryInterval   = 5 * time.Second
	waitRetryTImeout    = 5 * time.Minute
)

func ScaleResourceWithRetries(scaler kubectl.Scaler, namespace, name string, size uint) error {
	waitForScale := kubectl.NewRetryParams(updateRetryInterval, updateRetryTimeout)
	waitForReplicas := kubectl.NewRetryParams(waitRetryInterval, waitRetryTImeout)
	if err := scaler.Scale(namespace, name, size, nil, waitForScale, waitForReplicas); err != nil {
		return fmt.Errorf("Error while scaling %s to %d replicas: %v", name, size, err)
	}
	return nil
}
