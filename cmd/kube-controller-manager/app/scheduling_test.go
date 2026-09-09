/*
Copyright The Kubernetes Authors.

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

package app

import (
	"context"
	"testing"
	"time"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"

	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/features"
)

func TestPodGroupProtectionController(t *testing.T) {
	tests := []struct {
		name              string
		enableFeatureGate bool
		expectController  bool
	}{
		{
			name:              "controller runs when GenericWorkload is enabled",
			enableFeatureGate: true,
			expectController:  true,
		},
		{
			name:              "controller is disabled when GenericWorkload is disabled",
			enableFeatureGate: false,
			expectController:  false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericWorkload, test.enableFeatureGate)
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			clientset := fakeclientset.NewClientset()
			controllerCtx := ControllerContext{
				ClientBuilder:   TestClientBuilder{clientset: clientset},
				InformerFactory: informers.NewSharedInformerFactoryWithOptions(clientset, time.Duration(1)),
			}
			controllerCtx.ComponentConfig.Generic.Controllers = []string{names.PodGroupProtectionController}

			// Verify the constructor produces the expected result.
			ctrl, err := newPodGroupProtectionController(ctx, controllerCtx, names.PodGroupProtectionController)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if test.expectController && ctrl == nil {
				t.Errorf("expected a controller, got nil")
			} else if !test.expectController && ctrl != nil {
				t.Errorf("expected no controller, got %v", ctrl)
			}

			// Verify the descriptor gates the controller correctly.
			initFuncCalled := false
			descriptor := NewControllerDescriptors()[names.PodGroupProtectionController]
			descriptor.constructor = func(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
				initFuncCalled = true
				return newControllerLoop(func(ctx context.Context) {}, controllerName), nil
			}

			var healthChecks mockHealthCheckAdder
			if err := runControllers(ctx, controllerCtx, map[string]*ControllerDescriptor{
				names.PodGroupProtectionController: descriptor,
			}, &healthChecks); err != nil {
				t.Fatalf("unexpected error starting controller: %v", err)
			}
			if test.expectController != initFuncCalled {
				t.Errorf("constructor call: expected=%v, got=%v", test.expectController, initFuncCalled)
			}
			hasHealthCheck := len(healthChecks.Checks) > 0
			if test.expectController != hasHealthCheck {
				t.Errorf("health check: expected=%v, got=%v", test.expectController, hasHealthCheck)
			}
		})
	}
}
