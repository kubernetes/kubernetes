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

package events

import (
	"context"
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
)

// Action is a function to be performed by the system.
type Action func() error

// WaitTimeoutForEvent waits the given timeout duration for an event to occur.
// Please note delivery of events is not guaranteed. Asserting on events can lead to flaky tests.
func WaitTimeoutForEvent(ctx context.Context, c clientset.Interface, namespace, eventSelector, msg string, timeout time.Duration) error {
	interval := 2 * time.Second
	return wait.PollUntilContextTimeout(ctx, interval, timeout, true, eventOccurred(c, namespace, eventSelector, msg))
}

func eventOccurred(c clientset.Interface, namespace, eventSelector, msg string) wait.ConditionWithContextFunc {
	options := metav1.ListOptions{FieldSelector: eventSelector}
	return func(ctx context.Context) (bool, error) {
		events, err := c.CoreV1().Events(namespace).List(ctx, options)
		if err != nil {
			return false, fmt.Errorf("got error while getting events: %w", err)
		}
		for _, event := range events.Items {
			if strings.Contains(event.Message, msg) {
				return true, nil
			}
		}
		return false, nil
	}
}
