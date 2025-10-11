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
	"fmt"
	"sort"

	"k8s.io/apimachinery/pkg/util/sets"

	"k8s.io/kubernetes/cmd/kube-controller-manager/internal/controller"
)

// KnownControllers returns all known controllers' name
func KnownControllers() []string {
	return sets.StringKeySet(NewControllerDescriptors()).List()
}

// ControllerAliases returns a mapping of aliases to canonical controller names
func ControllerAliases() map[string]string {
	aliases := map[string]string{}
	for name, c := range NewControllerDescriptors() {
		for _, alias := range c.Aliases {
			aliases[alias] = name
		}
	}
	return aliases
}

func ControllersDisabledByDefault() []string {
	var controllersDisabledByDefault []string

	for name, c := range NewControllerDescriptors() {
		if c.IsDisabledByDefault {
			controllersDisabledByDefault = append(controllersDisabledByDefault, name)
		}
	}

	sort.Strings(controllersDisabledByDefault)

	return controllersDisabledByDefault
}

// NewControllerDescriptors is a public map of named controller groups (you can start more than one in an init func)
// paired to their ControllerDescriptor wrapper object that includes the associated controller constructor.
// This allows for structured downstream composition and subdivision.
func NewControllerDescriptors() map[string]*controller.Descriptor {
	controllers := map[string]*controller.Descriptor{}
	aliases := sets.NewString()

	// All the controllers must fulfil common constraints, or else we will explode.
	register := func(controllerDesc *controller.Descriptor) {
		if controllerDesc == nil {
			panic("received nil controller for a registration")
		}
		name := controllerDesc.Name
		if len(name) == 0 {
			panic("received controller without a name for a registration")
		}
		if _, found := controllers[name]; found {
			panic(fmt.Sprintf("controller name %q was registered twice", name))
		}
		if controllerDesc.Constructor == nil {
			panic(fmt.Sprintf("controller %q does not have a constructor specified", name))
		}

		for _, alias := range controllerDesc.Aliases {
			if aliases.Has(alias) {
				panic(fmt.Sprintf("controller %q has a duplicate alias %q", name, alias))
			}
			aliases.Insert(alias)
		}

		controllers[name] = controllerDesc
	}

	// First add "special" controllers that aren't initialized normally. These controllers cannot be initialized
	// in the main controller loop initialization, so we add them here only for the metadata and duplication detection.
	// ControllerDescriptor#RequiresSpecialHandling should return true for such controllers
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
