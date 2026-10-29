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

package resourceslice

import (
	"fmt"
	"strings"
	"unicode/utf8"

	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
)

// validateDriverResources identifies problems that cannot be caught by
// the server-side validation and that would prevent using the published
// ResourceSlices, like incorrect cross-references. We do validation here
// so any issues can be discovered as early as possible. This is also
// checked in the allocator before allocating any devices.
func validateDriverResources(driverName string, resources *DriverResources) error {
	for poolName, pool := range resources.Pools {
		if err := validatePool(driverName, poolName, pool); err != nil {
			return err
		}
	}
	return nil
}

// validatePool checks that there aren't any pool-wide issues that
// can't be caught in the API-server per-ResourceSlice validation.
//
// An empty driverName skips validateUnqualifiedName checks (used when the
// caller opted out via [Options.ValidateQualifiedNames]).
//
// This logic is very similar to what we do in the allocator when we
// gather the pools. We might want to see if there is a good way to
// put this logic in one place.
func validatePool(driverName, name string, pool Pool) error {
	counterSets := make(map[string]resourceapi.CounterSet)
	for _, slice := range pool.Slices {
		for _, counterSet := range slice.SharedCounters {
			if _, found := counterSets[counterSet.Name]; found {
				return fmt.Errorf("found duplicate counter set %q in pool %q", counterSet.Name, name)
			}
			counterSets[counterSet.Name] = counterSet
		}
	}

	if err := validatePoolSharingAffinity(name, pool); err != nil {
		return err
	}

	devices := sets.New[string]()
	for _, slice := range pool.Slices {
		for _, device := range slice.Devices {
			if _, found := devices[device.Name]; found {
				return fmt.Errorf("found duplicate device %q in pool %q", device.Name, name)
			}
			devices.Insert(device.Name)

			if driverName != "" {
				for attrName := range device.Attributes {
					if err := validateUnqualifiedName(driverName, attrName); err != nil {
						return fmt.Errorf("pool %q: device %q: attribute %q: %w", name, device.Name, attrName, err)
					}
				}
				for capName := range device.Capacity {
					if err := validateUnqualifiedName(driverName, capName); err != nil {
						return fmt.Errorf("pool %q: device %q: capacity %q: %w", name, device.Name, capName, err)
					}
				}
			}

			for _, dcc := range device.ConsumesCounters {
				counterSet, found := counterSets[dcc.CounterSet]
				if !found {
					return fmt.Errorf("counter set %q referenced by device %q not found", dcc.CounterSet, device.Name)
				}
				for counterName := range dcc.Counters {
					if _, found := counterSet.Counters[counterName]; !found {
						return fmt.Errorf("counter %q referenced by device %q not found in counter set %q", counterName, device.Name, counterSet.Name)
					}
				}
			}
		}
	}
	return nil
}

// Per-slice field validation runs separately against the matched stored objects.
// Pool validation uses the allocator's compilation environment and cost checks.
func validatePoolSharingAffinity(poolName string, pool Pool) error {
	var all []resourceapi.SharingAffinityExtractor
	names := sets.New[string]()
	for _, slice := range pool.Slices {
		for _, extractor := range slice.SharingAffinity {
			if names.Has(extractor.Name) {
				return fmt.Errorf("pool %q: duplicate sharing affinity key %q", poolName, extractor.Name)
			}
			names.Insert(extractor.Name)
		}
		all = append(all, slice.SharingAffinity...)
	}
	if err := sharingaffinity.ValidateExpressions(all); err != nil {
		return fmt.Errorf("pool %q: sharing affinity: %w", poolName, err)
	}
	return nil
}

// Ratcheting applies only to unchanged values stored on the ResourceSlice being updated.
func validateSliceSharingAffinity(slice Slice, old *resourceapi.ResourceSlice) error {
	if slice.SharingAffinity != nil && (slice.Devices != nil || slice.SharedCounters != nil) {
		return fmt.Errorf("sharingAffinity cannot be combined with devices or sharedCounters in one slice")
	}
	var oldExtractors []resourceapi.SharingAffinityExtractor
	if old != nil {
		oldExtractors = old.Spec.SharingAffinity
	}
	if oldExtractors != nil && apiequality.Semantic.DeepEqual(slice.SharingAffinity, oldExtractors) {
		return nil
	}
	if len(slice.SharingAffinity) > resourceapi.SharingAffinityMaxEntries {
		return fmt.Errorf("sharing affinity entries %d exceeds %d", len(slice.SharingAffinity), resourceapi.SharingAffinityMaxEntries)
	}
	oldByName := make(map[string]*resourceapi.SharingAffinityExtractor, len(oldExtractors))
	for i := range oldExtractors {
		oldByName[oldExtractors[i].Name] = &oldExtractors[i]
	}
	for i, entry := range slice.SharingAffinity {
		if err := validateSharingAffinityDeclaration(entry, oldByName[entry.Name]); err != nil {
			return fmt.Errorf("sharing affinity #%d (%q): %w", i, entry.Name, err)
		}
	}
	return nil
}

// validateSharingAffinityDeclaration validates new or changed fields without
// compiling CEL. For ratcheting, old must be the same-name extractor from the
// ResourceSlice being updated.
func validateSharingAffinityDeclaration(entry resourceapi.SharingAffinityExtractor, old *resourceapi.SharingAffinityExtractor) error {
	if old == nil {
		if errs := validation.IsDNS1123Label(entry.Name); len(errs) > 0 {
			return fmt.Errorf("invalid name: %s", strings.Join(errs, ", "))
		}
	}
	if old == nil || old.APIGroup != entry.APIGroup {
		if err := validateSharingAffinityAPIGroupMatcher(entry.APIGroup); err != nil {
			return fmt.Errorf("invalid apiGroup: %w", err)
		}
	}
	if old == nil || old.Kind != entry.Kind {
		if err := validateSharingAffinityKindMatcher(entry.Kind); err != nil {
			return fmt.Errorf("invalid kind: %w", err)
		}
	}
	if old == nil || old.Expression != entry.Expression {
		if entry.Expression == "" {
			return fmt.Errorf("expression is required")
		}
		if len(entry.Expression) > resourceapi.SharingAffinityCELExpressionMaxLength {
			return fmt.Errorf("expression length %d exceeds %d bytes", len(entry.Expression), resourceapi.SharingAffinityCELExpressionMaxLength)
		}
	}
	return nil
}

// validateSharingAffinityAPIGroupMatcher validates an API group matcher, allowing
// "*" as a wildcard or a lowercase DNS subdomain.
func validateSharingAffinityAPIGroupMatcher(apiGroup string) error {
	if apiGroup == "" {
		return fmt.Errorf("value is required")
	}
	if apiGroup == "*" {
		return nil
	}
	const maxLength = validation.DNS1123SubdomainMaxLength
	if utf8.RuneCountInString(apiGroup) > maxLength {
		return fmt.Errorf("value exceeds %d characters", maxLength)
	}
	if errs := validation.IsDNS1123Subdomain(apiGroup); len(errs) > 0 {
		return fmt.Errorf("%s", strings.Join(errs, ", "))
	}
	return nil
}

// validateSharingAffinityKindMatcher validates a kind matcher, allowing "*" as a
// wildcard or a DNS1035 label with uppercase letters allowed.
func validateSharingAffinityKindMatcher(kind string) error {
	if kind == "" {
		return fmt.Errorf("value is required")
	}
	if kind == "*" {
		return nil
	}
	const maxLength = validation.DNS1035LabelMaxLength
	if utf8.RuneCountInString(kind) > maxLength {
		return fmt.Errorf("value exceeds %d characters", maxLength)
	}
	if errs := validation.IsDNS1035Label(strings.ToLower(kind)); len(errs) > 0 {
		return fmt.Errorf("%s", strings.Join(errs, ", "))
	}
	return nil
}

// validateUnqualifiedName rejects an attribute or capacity name that is explicitly
// qualified with the driver's own domain (e.g. "<driverName>/foo"). Such a name is
// always equivalent to the same name given without a domain ("foo"), because the
// driver's domain is implicit for its own ResourceSlices. Publishing both forms, or
// only the qualified form, is redundant and at worst creates two conflicting entries.
//
// The apiserver does not validate this. The scheduling code in 1.37 prefers
// the fully-qualified name for constraints, but CEL lookup is random.
// Therefore DRA drivers should not publish both.
//
// Callers must not invoke this with an empty driverName; see [validatePool].
func validateUnqualifiedName(driverName string, name resourceapi.QualifiedName) error {
	domain, id, hasDomain := strings.Cut(string(name), "/")
	if hasDomain && domain == driverName {
		return fmt.Errorf("use unqualified %q, the driver name is added implicitly", id)
	}
	return nil
}
