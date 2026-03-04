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

package app

import (
	"context"
	"fmt"
	"sort"

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
)

// This file contains types and functions for wrapping controller implementations from downstream packages.
// Every controller wrapper implements the Controller interface,
// which is then associated with a ControllerDescriptor, which holds additional static metadata
// needed so that the manager can manage Controllers properly.

// Controller defines the base interface that all controller wrappers must implement.
type Controller interface {
	// Name returns the controller's canonical name.
	Name() string

	// Run runs the controller loop.
	// When there is anything to be done, it blocks until the context is cancelled.
	// Run must ensure all goroutines are terminated before returning.
	Run(context.Context)
}

// ControllerConstructor is a constructor for a controller.
// A nil Controller returned means that the associated controller is disabled.
type ControllerConstructor func(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error)

type ControllerDescriptor struct {
	name                      string
	constructor               ControllerConstructor
	requiredFeatureGates      []featuregate.Feature
	aliases                   []string
	isDisabledByDefault       bool
	isCloudProviderController bool
	requiresSpecialHandling   bool
}

func (r *ControllerDescriptor) Name() string {
	return r.name
}

func (r *ControllerDescriptor) GetControllerConstructor() ControllerConstructor {
	return r.constructor
}

func (r *ControllerDescriptor) GetRequiredFeatureGates() []featuregate.Feature {
	return append([]featuregate.Feature(nil), r.requiredFeatureGates...)
}

// GetAliases returns aliases to ensure backwards compatibility and should never be removed!
// Only addition of new aliases is allowed, and only when a canonical name is changed (please see CHANGE POLICY of controller names)
func (r *ControllerDescriptor) GetAliases() []string {
	return append([]string(nil), r.aliases...)
}

func (r *ControllerDescriptor) IsDisabledByDefault() bool {
	return r.isDisabledByDefault
}

func (r *ControllerDescriptor) IsCloudProviderController() bool {
	return r.isCloudProviderController
}

// RequiresSpecialHandling should return true only in a special non-generic controllers like ServiceAccountTokenController
func (r *ControllerDescriptor) RequiresSpecialHandling() bool {
	return r.requiresSpecialHandling
}

// BuildController creates a controller based on the given descriptor.
// The associated controller's constructor is called at the end, so the same contract applies for the return values here.
func (r *ControllerDescriptor) BuildController(ctx context.Context, controllerCtx ControllerContext) (Controller, error) {
	logger := klog.FromContext(ctx)
	controllerName := r.Name()

	for _, featureGate := range r.GetRequiredFeatureGates() {
		if !utilfeature.DefaultFeatureGate.Enabled(featureGate) {
			logger.Info("Controller is disabled by a feature gate",
				"controller", controllerName,
				"requiredFeatureGates", r.GetRequiredFeatureGates())
			return nil, nil
		}
	}

	if r.IsCloudProviderController() {
		logger.Info("Skipping a cloud provider controller", "controller", controllerName)
		return nil, nil
	}

	ctx = klog.NewContext(ctx, klog.LoggerWithName(logger, controllerName))
	return r.GetControllerConstructor()(ctx, controllerCtx, controllerName)
}

// KnownControllers returns all known controllers' name
func KnownControllers() []string {
	return sets.StringKeySet(NewControllerDescriptors()).List()
}

// ControllerAliases returns a mapping of aliases to canonical controller names
func ControllerAliases() map[string]string {
	aliases := map[string]string{}
	for name, c := range NewControllerDescriptors() {
		for _, alias := range c.GetAliases() {
			aliases[alias] = name
		}
	}
	return aliases
}

func ControllersDisabledByDefault() []string {
	var controllersDisabledByDefault []string

	for name, c := range NewControllerDescriptors() {
		if c.IsDisabledByDefault() {
			controllersDisabledByDefault = append(controllersDisabledByDefault, name)
		}
	}

	sort.Strings(controllersDisabledByDefault)

	return controllersDisabledByDefault
}

// NewControllerDescriptors is a public map of named controller groups (you can start more than one in an init func)
// paired to their ControllerDescriptor wrapper object that includes the associated controller constructor.
// This allows for structured downstream composition and subdivision.
func NewControllerDescriptors() map[string]*ControllerDescriptor {
	controllers := map[string]*ControllerDescriptor{}
	aliases := sets.NewString()

	// All the controllers must fulfil common constraints, or else we will explode.
	register := func(controllerDesc *ControllerDescriptor) {
		if controllerDesc == nil {
			panic("received nil controller for a registration")
		}
		name := controllerDesc.Name()
		if len(name) == 0 {
			panic("received controller without a name for a registration")
		}
		if _, found := controllers[name]; found {
			panic(fmt.Sprintf("controller name %q was registered twice", name))
		}
		if controllerDesc.GetControllerConstructor() == nil {
			panic(fmt.Sprintf("controller %q does not have a constructor specified", name))
		}

		for _, alias := range controllerDesc.GetAliases() {
			if aliases.Has(alias) {
				panic(fmt.Sprintf("controller %q has a duplicate alias %q", name, alias))
			}
			aliases.Insert(alias)
		}

		controllers[name] = controllerDesc
	}

	// First add "special" controllers that aren't initialized normally. These controllers cannot be initialized
	// in the main controller loop initialization, so we add them here only for the metadata and duplication detection.
	// app.ControllerDescriptor#RequiresSpecialHandling should return true for such controllers
	// The only known special case is the ServiceAccountTokenController which *must* be started
	// first to ensure that the SA tokens for future controllers will exist. Think very carefully before adding new
	// special controllers.
	register(newServiceAccountTokenControllerDescriptor(nil))

	register(newEndpointsControllerDescriptor())
	register(newEndpointSliceControllerDescriptor())
	register(newEndpointSliceMirroringControllerDescriptor())
	register(newReplicationControllerDescriptor())
	register(newPodGarbageCollectorControllerDescriptor())
	register(newResourceQuotaControllerDescriptor())
	register(newNamespaceControllerDescriptor())
	register(newServiceAccountControllerDescriptor())
	register(newGarbageCollectorControllerDescriptor())
	register(newDaemonSetControllerDescriptor())
	register(newJobControllerDescriptor())
	register(newDeploymentControllerDescriptor())
	register(newReplicaSetControllerDescriptor())
	register(newHorizontalPodAutoscalerControllerDescriptor())
	register(newDisruptionControllerDescriptor())
	register(newStatefulSetControllerDescriptor())
	register(newCronJobControllerDescriptor())
	register(newCertificateSigningRequestSigningControllerDescriptor())
	register(newCertificateSigningRequestApprovingControllerDescriptor())
	register(newCertificateSigningRequestCleanerControllerDescriptor())
	register(newPodCertificateRequestCleanerControllerDescriptor())
	register(newTTLControllerDescriptor())
	register(newBootstrapSignerControllerDescriptor())
	register(newTokenCleanerControllerDescriptor())
	register(newNodeIpamControllerDescriptor())
	register(newNodeLifecycleControllerDescriptor())

	register(newServiceLBControllerDescriptor())          // cloud provider controller
	register(newNodeRouteControllerDescriptor())          // cloud provider controller
	register(newCloudNodeLifecycleControllerDescriptor()) // cloud provider controller

	register(newPersistentVolumeBinderControllerDescriptor())
	register(newPersistentVolumeAttachDetachControllerDescriptor())
	register(newPersistentVolumeExpanderControllerDescriptor())
	register(newClusterRoleAggregrationControllerDescriptor())
	register(newPersistentVolumeClaimProtectionControllerDescriptor())
	register(newPersistentVolumeProtectionControllerDescriptor())
	register(newVolumeAttributesClassProtectionControllerDescriptor())
	register(newTTLAfterFinishedControllerDescriptor())
	register(newRootCACertificatePublisherControllerDescriptor())
	register(newKubeAPIServerSignerClusterTrustBundledPublisherDescriptor())
	register(newEphemeralVolumeControllerDescriptor())

	// feature gated
	register(newStorageVersionGarbageCollectorControllerDescriptor())
	register(newResourceClaimControllerDescriptor())
	register(newDeviceTaintEvictionControllerDescriptor())
	register(newLegacyServiceAccountTokenCleanerControllerDescriptor())
	register(newValidatingAdmissionPolicyStatusControllerDescriptor())
	register(newTaintEvictionControllerDescriptor())
	register(newServiceCIDRsControllerDescriptor())
	register(newStorageVersionMigratorControllerDescriptor())
	register(newSELinuxWarningControllerDescriptor())

	for _, alias := range aliases.UnsortedList() {
		if _, ok := controllers[alias]; ok {
			panic(fmt.Sprintf("alias %q conflicts with a controller name", alias))
		}
	}

	return controllers
}
