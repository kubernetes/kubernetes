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

package apidefinitions

import (
	"context"
	"slices"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

// TestAllowCreateOnUpdate verifies that APIs prefer AllowCreateOnUpdate() = false.
//
// Rationale: POST is exclusively for creation, and PUT is strictly for updating existing
// resources to ensure safe and predictable lifecycles. Allowing PUT to create objects
// can lead to accidental object creation.
func TestAllowCreateOnUpdate(t *testing.T) {
	exempt := sets.New(
		// Leases are maintained exclusively via PUT requests.
		"leasecandidates.coordination.k8s.io",
		"leases.coordination.k8s.io",

		// Grandfathered APIs with no clear rationale:
		"clusterrolebindings.rbac.authorization.k8s.io",
		"clusterroles.rbac.authorization.k8s.io",
		"endpoints",
		"events.events.k8s.io",
		"events",
		"limitranges",
		"rolebindings.rbac.authorization.k8s.io",
		"roles.rbac.authorization.k8s.io",
		"runtimeclasses.node.k8s.io",
		"services",
	)
	TestAllDefinitions(t, "allow-create-on-update", func(t *testing.T, api Definition) {
		if !api.HasVerb("create") || !api.HasVerb("get") || !api.HasVerb("update") || !api.HasVerb("delete") {
			t.Skip("Resource does not support create, get, update, and delete")
		}
		client := api.ResourceClient()
		obj := TestObj(t, api.StorageData.Stub, "{}", api.Mapping.GroupVersionKind)
		obj.SetResourceVersion("")
		obj.SetUID("")
		_, err := client.Update(context.TODO(), obj, metav1.UpdateOptions{})
		assertDefault(t, api.Mapping.Resource, "AllowCreateOnUpdate must be false", errors.IsNotFound(err), exempt)
	})
}

// TestGenerateName verifies that APIs honor metadata.generateName.
//
// Rationale: generateName should be provided by APIs unless new resources of that API require the name to be derived
// from particular field values.
func TestGenerateName(t *testing.T) {
	exempt := sets.New(
		// APIs with specific naming requirements that to not support generateName:
		"apiservices.apiregistration.k8s.io",
		"customresourcedefinitions.apiextensions.k8s.io",
		"ipaddresses.networking.k8s.io",
	)
	TestAllDefinitions(t, "generate-name", func(t *testing.T, api Definition) {
		if !api.HasVerb("create") || !api.HasVerb("get") || !api.HasVerb("update") || !api.HasVerb("delete") {
			t.Skip("Resource does not support create, get, update, and delete")
		}
		client := api.ResourceClient()
		obj := TestObj(t, api.StorageData.Stub, "{}", api.Mapping.GroupVersionKind)
		obj.SetName("")
		obj.SetGenerateName("default-behaviors-")
		obj.SetResourceVersion("")
		obj.SetUID("")
		created, err := client.Create(context.TODO(), obj, metav1.CreateOptions{})
		generated := err == nil && strings.HasPrefix(created.GetName(), "default-behaviors-")
		assertDefault(t, api.Mapping.Resource, "metadata.generateName must produce a server-generated name", generated, exempt)
	})
}

// TestAllowUnconditionalUpdate verifies that APIs prefer AllowUnconditionalUpdate() = false.
//
// Rationale: Unconditional updates can result in accidental overwrites. APIs should encourage
// users to either use conditional updates or patch operations.
func TestAllowUnconditionalUpdate(t *testing.T) {
	exempt := sets.New(
		// Grandfathered APIs:
		"apiservices.apiregistration.k8s.io",
		"certificatesigningrequests.v1.certificates.k8s.io",
		"clusterrolebindings.v1.rbac.authorization.k8s.io",
		"clusterroles.v1.rbac.authorization.k8s.io",
		"clustertrustbundles.certificates.k8s.io",
		"configmaps.v1",
		"controllerrevisions.v1.apps",
		"cronjobs.v1.batch",
		"csidrivers.storage.k8s.io",
		"csinodes.storage.k8s.io",
		"csistoragecapacities.storage.k8s.io",
		"customresourcedefinitions.apiextensions.k8s.io",
		"daemonsets.v1.apps",
		"deployments.v1.apps",
		"deviceclasses.v1.resource.k8s.io",
		"endpoints.v1",
		"endpointslices.v1.discovery.k8s.io",
		"events.v1.events.k8s.io",
		"events.v1",
		"flowschemas.v1.flowcontrol.apiserver.k8s.io",
		"foos.cr.bar.com",
		"horizontalpodautoscalers.v1.autoscaling",
		"horizontalpodautoscalers.v2.autoscaling",
		"ingressclasses.v1.networking.k8s.io",
		"ingresses.v1.networking.k8s.io",
		"integers.random.numbers.com",
		"ipaddresses.v1.networking.k8s.io",
		"jobs.v1.batch",
		"leasecandidates.coordination.k8s.io",
		"leases.coordination.k8s.io",
		"limitranges.v1",
		"mutatingadmissionpolicies.admissionregistration.k8s.io",
		"mutatingadmissionpolicybindings.admissionregistration.k8s.io",
		"mutatingwebhookconfigurations.admissionregistration.k8s.io",
		"namespaces.v1",
		"networkpolicies.v1.networking.k8s.io",
		"nodes.v1",
		"pandas.awesome.bears.com",
		"pants.custom.fancy.com",
		"persistentvolumeclaims.v1",
		"persistentvolumes.v1",
		"podcertificaterequests.certificates.k8s.io",
		"poddisruptionbudgets.policy",
		"pods.v1",
		"podtemplates.v1",
		"priorityclasses.v1.scheduling.k8s.io",
		"prioritylevelconfigurations.v1.flowcontrol.apiserver.k8s.io",
		"replicasets.v1.apps",
		"replicationcontrollers.v1",
		"resourceclaims.v1.resource.k8s.io",
		"resourceclaimtemplates.v1.resource.k8s.io",
		"resourcequotas.v1",
		"resourceslices.v1.resource.k8s.io",
		"rolebindings.v1.rbac.authorization.k8s.io",
		"roles.v1.rbac.authorization.k8s.io",
		"runtimeclasses.node.k8s.io",
		"secrets.v1",
		"serviceaccounts.v1",
		"servicecidrs.v1.networking.k8s.io",
		"services.v1",
		"statefulsets.v1.apps",
		"storageclasses.v1.storage.k8s.io",
		"storageversionmigrations.storagemigration.k8s.io",
		"storageversions.internal.apiserver.k8s.io",
		"validatingadmissionpolicies.admissionregistration.k8s.io",
		"validatingadmissionpolicybindings.admissionregistration.k8s.io",
		"validatingwebhookconfigurations.admissionregistration.k8s.io",
		"volumeattachments.storage.k8s.io",
		"volumeattributesclasses.v1.storage.k8s.io",
	)
	TestAllDefinitions(t, "unconditional-update", func(t *testing.T, api Definition) {
		if !api.HasVerb("create") || !api.HasVerb("get") || !api.HasVerb("update") || !api.HasVerb("delete") {
			t.Skip("Resource does not support create, get, update, and delete")
		}
		client := api.ResourceClient()
		obj := TestObj(t, api.StorageData.Stub, "{}", api.Mapping.GroupVersionKind)

		created, err := client.Create(context.TODO(), obj, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create: %v", err)
		}

		created.SetResourceVersion("")
		_, err = client.Update(context.TODO(), created, metav1.UpdateOptions{})

		assertDefault(t, api.Mapping.Resource, "AllowUnconditionalUpdate must be false", errors.IsConflict(err), exempt)
	})
}

// TestDefaultGarbageCollectionPolicy verifies that resources prefer the default
// GarbageCollectionPolicy strategy of background deletion.
//
// Rationale: Orphaning dependents by default proved to be confusing to
// controller authors and users and prone to resource leaks. Foreground deletion
// allows dependents to block resource deletion.
func TestDefaultGarbageCollectionPolicy(t *testing.T) {
	// APIs that use OrphanDependents garbage collection policy
	orphanDependentsExempt := sets.New(
		// Grandfathered APIs that use OrphanDependants for backward compatibility
		"jobs.batch",
		"replicationcontrollers",

		// non-GA APIs that use OrphanDependants, such as cronjobs, exist but is no longer served
	)
	// APIs that use DeleteDependents garbage collection policy
	foregroundExempt := sets.New[string]()

	// APIs that use Unsupported garbage collection policy
	unsupportedExempt := sets.New(
		// Events are intended to be high-volume leaf nodes.
		"events",
		"events.events.k8s.io",
	)
	TestAllDefinitions(t, "default-gc-policy", func(t *testing.T, api Definition) {
		if !api.HasVerb("create") || !api.HasVerb("get") || !api.HasVerb("update") || !api.HasVerb("delete") {
			t.Skip("Resource does not support create, get, update, and delete")
		}
		client := api.ResourceClient()
		obj := TestObj(t, api.StorageData.Stub, "{}", api.Mapping.GroupVersionKind)
		name := obj.GetName()

		if _, err := client.Create(context.TODO(), obj, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create: %v", err)
		}
		latest, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get latest: %v", err)
		}

		// A blocking finalizer is added first, so the object is not removed
		// before we can observe the result of each delete.
		latest.SetFinalizers(append(latest.GetFinalizers(), "test.k8s.io/block"))
		if _, err := client.Update(context.TODO(), latest, metav1.UpdateOptions{}); err != nil {
			t.Logf("Could not set test finalizer, skipping GC policy check: %v", err)
			return
		}
		if err := client.Delete(context.TODO(), name, metav1.DeleteOptions{}); err != nil {
			t.Fatalf("Failed to delete: %v", err)
		}
		afterDefault, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get after default delete: %v", err)
		}
		hasOrphan := slices.Contains(afterDefault.GetFinalizers(), metav1.FinalizerOrphanDependents)
		hasForegroundDeletion := slices.Contains(afterDefault.GetFinalizers(), metav1.FinalizerDeleteDependents)
		isBackground := !hasOrphan && !hasForegroundDeletion

		assertDefault(t, api.Mapping.Resource, "DefaultGarbageCollectionPolicy must be unset, to indicate that dependents are background deleted", isBackground, orphanDependentsExempt.Union(foregroundExempt))
		if orphanDependentsExempt.Has(api.Mapping.Resource.GroupResource().String()) {
			if !hasOrphan {
				t.Errorf("%s: DefaultGarbageCollectionPolicy expected to be OrphanDependents", api.Mapping.Resource.GroupResource().String())
			}
		}
		if foregroundExempt.Has(api.Mapping.Resource.GroupResource().String()) {
			if !hasForegroundDeletion {
				t.Errorf("%s: DefaultGarbageCollectionPolicy expected to be DeleteDependents", api.Mapping.Resource.GroupResource().String())
			}
		}
	})

	TestAllDefinitions(t, "default-gc-policy-foreground", func(t *testing.T, api Definition) {
		if !api.HasVerb("create") || !api.HasVerb("get") || !api.HasVerb("update") || !api.HasVerb("delete") {
			t.Skip("Resource does not support create, get, update, and delete")
		}
		client := api.ResourceClient()
		obj := TestObj(t, api.StorageData.Stub, "{}", api.Mapping.GroupVersionKind)
		name := obj.GetName()

		if _, err := client.Create(context.TODO(), obj, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create: %v", err)
		}
		latest, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get latest: %v", err)
		}

		// A blocking finalizer is added first, so the object is not removed
		// before we can observe the result of each delete.
		latest.SetFinalizers(append(latest.GetFinalizers(), "test.k8s.io/block"))
		if _, err := client.Update(context.TODO(), latest, metav1.UpdateOptions{}); err != nil {
			t.Logf("Could not set test finalizer, skipping GC policy check: %v", err)
			return
		}

		// Also check for APIs using Unsupported garbage collection policy
		foreground := metav1.DeletePropagationForeground
		if err := client.Delete(context.TODO(), name, metav1.DeleteOptions{PropagationPolicy: &foreground}); err != nil {
			t.Fatalf("Failed to delete with foreground propagation: %v", err)
		}
		afterForeground, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get after foreground delete: %v", err)
		}
		hasForegroundDeletion := slices.Contains(afterForeground.GetFinalizers(), metav1.FinalizerDeleteDependents)
		assertDefault(t, api.Mapping.Resource, "DefaultGarbageCollectionPolicy must not return Unsupported, propagationPolicy=Foreground must add FinalizerDeleteDependents", hasForegroundDeletion, unsupportedExempt)
	})
}

// TestCheckGracefulDelete verifies that APIs do not implement deletion grace period support.
//
// Rationale: The use of finalizers followed by immediate deletion is more predictable to clients.
func TestCheckGracefulDelete(t *testing.T) {
	exempt := sets.New(
		// Pods support a grace period window to send SIGTERM and let containers shut down
		// cleanly before the object is removed.
		"pods",
	)
	TestAllDefinitions(t, "check-graceful-delete", func(t *testing.T, api Definition) {
		if !api.HasVerb("create") || !api.HasVerb("get") || !api.HasVerb("update") || !api.HasVerb("delete") {
			t.Skip("Resource does not support create, get, update, and delete")
		}
		rsc := api.ResourceClient()
		obj := TestObj(t, api.StorageData.Stub, "{}", api.Mapping.GroupVersionKind)
		name := obj.GetName()

		if api.Mapping.Resource.GroupResource().String() == "pods" {
			obj.Object["spec"].(map[string]any)["nodeName"] = "fake-node"
		}

		if _, err := rsc.Create(context.TODO(), obj, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create: %v", err)
		}
		gracePeriod := int64(30)
		if err := rsc.Delete(context.TODO(), name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod}); err != nil {
			t.Fatalf("Failed to delete with grace period: %v", err)
		}
		after, err := rsc.Get(context.TODO(), name, metav1.GetOptions{})

		isGraceful := false
		if err == nil {
			if after.GetDeletionGracePeriodSeconds() != nil && *after.GetDeletionGracePeriodSeconds() > 0 {
				isGraceful = true
			}
		}

		assertDefault(t, api.Mapping.Resource, "CheckGracefulDelete must not be implemented", !isGraceful, exempt)
	})
}

// assertDefault checks that a default behavior holds, with an allowlist for known exceptions.
func assertDefault(t *testing.T, gvr schema.GroupVersionResource, msg string, conforms bool, allowed sets.Set[string]) {
	t.Helper()
	name := ResourceString(gvr)
	if matchesException(gvr, allowed) {
		if conforms {
			t.Errorf("%s: %s unexpectedly conforms. Remove it from the exception list.", name, msg)
		}
	} else if !conforms {
		t.Errorf("%s: %s", name, msg)
	}
}
