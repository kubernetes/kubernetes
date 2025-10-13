/*
Copyright 2025 The Kubernetes Authors.

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

package controller

import (
	"context"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
)

// This file contains types and functions for wrapping controller implementations from downstream packages.
// Every controller wrapper implements the Controller interface,
// which is then associated with a Descriptor, which holds additional static metadata
// needed so that the manager can manage Controllers properly.

// Controller defines the base interface that all controller wrappers must implement.
type Controller interface {
	// Name returns the controller's canonical Name.
	Name() string

	// Run runs the controller loop.
	// When there is anything to be done, it blocks until the context is cancelled.
	// Run must ensure all goroutines are terminated before returning.
	Run(context.Context)
}

// Constructor is a Constructor for a controller.
// A nil Controller returned means that the associated controller is disabled.
type Constructor func(ctx context.Context, controllerContext Context, controllerName string) (Controller, error)

type Descriptor struct {
	Name                      string
	Constructor               Constructor
	RequiredFeatureGates      []featuregate.Feature
	Aliases                   []string
	IsDisabledByDefault       bool
	IsCloudProviderController bool
	RequiresSpecialHandling   bool
}

// BuildController creates a controller based on the given descriptor.
// The associated controller's Constructor is called at the end, so the same contract applies for the return values here.
func (r *Descriptor) BuildController(ctx context.Context, controllerCtx Context) (Controller, error) {
	logger := klog.FromContext(ctx)
	controllerName := r.Name

	for _, featureGate := range r.RequiredFeatureGates {
		if !utilfeature.DefaultFeatureGate.Enabled(featureGate) {
			logger.Info("Controller is disabled by a feature gate",
				"controller", controllerName,
				"requiredFeatureGates", r.RequiredFeatureGates)
			return nil, nil
		}
	}

	if r.IsCloudProviderController {
		logger.Info("Skipping a cloud provider controller", "controller", controllerName)
		return nil, nil
	}

	ctx = klog.NewContext(ctx, klog.LoggerWithName(logger, controllerName))
	return r.Constructor(ctx, controllerCtx, controllerName)
}
