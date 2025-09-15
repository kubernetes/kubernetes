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
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/kubectl/pkg/scale"
)

const (
	// Parameters for retrying updates/waits with linear backoff.
	// TODO: Try to move this to exponential backoff by modifying scale.Scale().
	updateRetryInterval = 5 * time.Second
	updateRetryTimeout  = 1 * time.Minute
	waitRetryInterval   = 5 * time.Second
	waitRetryTimeout    = 5 * time.Minute
)

func RetryErrorCondition(condition wait.ConditionWithContextFunc) wait.ConditionWithContextFunc {
	return func(ctx context.Context) (bool, error) {
		done, err := condition(ctx)
		return done, err
	}
}

func ScaleResourceWithRetries(scalesGetter scaleclient.ScalesGetter, namespace, name string, size uint, gvr schema.GroupVersionResource) error {
	scaler := scale.NewScaler(scalesGetter)
	preconditions := &scale.ScalePrecondition{
		Size:            -1,
		ResourceVersion: "",
	}
	waitForReplicas := scale.NewRetryParams(waitRetryInterval, waitRetryTimeout)
	cond := RetryErrorCondition(scale.ScaleCondition(scaler, preconditions, namespace, name, size, nil, gvr, false))
	err := wait.PollUntilContextTimeout(context.Background(), updateRetryInterval, updateRetryTimeout, true, cond)
	if err == nil {
		err = scale.WaitForScaleHasDesiredReplicas(scalesGetter, gvr.GroupResource(), name, namespace, size, waitForReplicas)
	}
	if err != nil {
		return fmt.Errorf("error while scaling %s to %d replicas: %v", name, size, err)
	}
	return nil
}
