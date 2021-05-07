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

package rest

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	flowcontrolv1beta1 "k8s.io/api/flowcontrol/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/client-go/kubernetes/fake"
)

func TestShouldEnsurePredefinedSettings(t *testing.T) {
	testCases := []struct {
		name                  string
		existingPriorityLevel *flowcontrolv1beta1.PriorityLevelConfiguration
		expected              bool
	}{
		{
			name:                  "should ensure if exempt priority-level is absent",
			existingPriorityLevel: nil,
			expected:              true,
		},
		{
			name:                  "should not ensure if exempt priority-level is present",
			existingPriorityLevel: bootstrap.MandatoryPriorityLevelConfigurationExempt,
			expected:              false,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			c := fake.NewSimpleClientset()
			if testCase.existingPriorityLevel != nil {
				c.FlowcontrolV1beta1().PriorityLevelConfigurations().Create(context.TODO(), testCase.existingPriorityLevel, metav1.CreateOptions{})
			}
			should, err := shouldCreateSuggested(c.FlowcontrolV1beta1())
			assert.NoError(t, err)
			assert.Equal(t, testCase.expected, should)
		})
	}
}

func TestContextFromChannelAndMaxWaitDurationWithChannelClosed(t *testing.T) {
	stopCh := make(chan struct{})
	ctx, cancel := contextFromChannelAndMaxWaitDuration(stopCh, time.Hour)
	defer cancel()

	select {
	case <-ctx.Done():
		t.Fatalf("Expected the derived context to be not cancelled, but got: %v", ctx.Err())
	default:
	}

	close(stopCh)

	<-ctx.Done()
	if ctx.Err() != context.Canceled {
		t.Errorf("Expected the context to be canceled with: %v, but got: %v", context.Canceled, ctx.Err())
	}
}

func TestContextFromChannelAndMaxWaitDurationWithMaxWaitElapsed(t *testing.T) {
	stopCh := make(chan struct{})
	ctx, cancel := contextFromChannelAndMaxWaitDuration(stopCh, 100*time.Millisecond)
	defer cancel()

	<-ctx.Done()

	if ctx.Err() != context.Canceled {
		t.Errorf("Expected the context to be canceled with: %v, but got: %v", context.Canceled, ctx.Err())
	}
}
