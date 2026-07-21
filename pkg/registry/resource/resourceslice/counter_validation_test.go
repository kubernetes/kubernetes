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

package resourceslice

import (
	"context"
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apiserverfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	resourceapi "k8s.io/kubernetes/pkg/apis/resource"
	_ "k8s.io/kubernetes/pkg/apis/resource/install"
	"k8s.io/kubernetes/pkg/features"
)

// TestCounterRequirementValidation preserves admission and ratcheting across feature-gate changes.
func TestCounterRequirementValidation(t *testing.T) {
	zero := resource.MustParse("0")
	one := resource.MustParse("1")
	negative := resource.MustParse("-1")
	none := resourceapi.CounterRequirement{}
	fixedZero := resourceapi.CounterRequirement{Value: &zero}
	fixed := resourceapi.CounterRequirement{Value: &one}
	from := resourceapi.CounterRequirement{ValueFrom: &resourceapi.CounterValueFrom{CapacityName: "driver.example.com/bandwidth"}}
	both := resourceapi.CounterRequirement{Value: &zero, ValueFrom: from.ValueFrom}
	makeSlice := func(requirement resourceapi.CounterRequirement) *resourceapi.ResourceSlice {
		obj := slice.DeepCopy()
		obj.UID = "counter-validation"
		obj.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{{
			CounterSet: "set", Counters: map[string]resourceapi.CounterRequirement{"memory": requirement},
		}}
		return obj
	}
	forms := []struct {
		name            string
		requirement     resourceapi.CounterRequirement
		wantOn, wantOff bool
	}{
		{"neither", none, false, false},
		{"explicit-zero", fixedZero, true, true},
		{"positive", fixed, true, true},
		{"negative", resourceapi.CounterRequirement{Value: &negative}, false, false},
		{"valueFrom", from, true, false},
		{"unqualified-capacity-name", resourceapi.CounterRequirement{ValueFrom: &resourceapi.CounterValueFrom{CapacityName: "bandwidth"}}, true, false},
		{"empty-capacity-name", resourceapi.CounterRequirement{ValueFrom: &resourceapi.CounterValueFrom{}}, false, false},
		{"invalid-capacity-name", resourceapi.CounterRequirement{ValueFrom: &resourceapi.CounterValueFrom{CapacityName: "bad/name/extra"}}, false, false},
		{"both", both, false, true},
		{"both-with-negative", resourceapi.CounterRequirement{Value: &negative, ValueFrom: from.ValueFrom}, false, false},
	}
	type scenario struct {
		name     string
		obj, old *resourceapi.ResourceSlice
		allowed  bool
	}
	for _, version := range []string{"v1", "v1beta1", "v1beta2"} {
		devicePath := field.NewPath("spec", "devices").Index(0)
		if version == "v1beta1" {
			devicePath = devicePath.Child("basic")
		}
		counterPath := devicePath.Child("consumesCounters").Index(0).Child("counters").Key("memory")
		for _, beta := range []bool{false, true} {
			for _, shared := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/beta=%t/shared=%t", version, beta, shared), func(t *testing.T) {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						apiserverfeatures.DeclarativeValidationBeta: beta,
						features.DRAPartitionableDevices:            true,
						features.DRAConsumableCapacity:              true,
						features.DRASharedConsumableCapacity:        shared,
					})
					var scenarios []scenario
					for _, form := range forms {
						allowed := form.wantOff
						if shared {
							allowed = form.wantOn
						}
						scenarios = append(scenarios,
							scenario{"create/" + form.name, makeSlice(form.requirement), nil, allowed},
							scenario{"update/" + form.name, makeSlice(form.requirement), makeSlice(fixed), allowed},
						)
					}
					for _, original := range []struct {
						name        string
						requirement resourceapi.CounterRequirement
					}{
						{"neither", none}, {"both", both}, {"empty-capacity-name", resourceapi.CounterRequirement{ValueFrom: &resourceapi.CounterValueFrom{}}},
					} {
						old := makeSlice(original.requirement)
						unchanged := old.DeepCopy()
						unchanged.Labels = map[string]string{"test": "unchanged-devices"}
						changed := old.DeepCopy()
						changed.Spec.Devices[0].Name = "renamed-device"
						repaired := makeSlice(fixedZero)
						scenarios = append(scenarios,
							scenario{"stored-invalid/" + original.name + "/metadata-only", unchanged, old, true},
							scenario{"stored-invalid/" + original.name + "/changed-devices", changed, old, false},
							scenario{"stored-invalid/" + original.name + "/repair-to-zero", repaired, old, true},
						)
					}
					oldFrom := makeSlice(from)
					changedFrom := oldFrom.DeepCopy()
					changedFrom.Spec.Devices[0].Name = "renamed-device"
					scenarios = append(scenarios, scenario{"existing-valueFrom/changed-device", changedFrom, oldFrom, true})
					for _, tc := range scenarios {
						t.Run(tc.name, func(t *testing.T) {
							obj := tc.obj.DeepCopy()
							var old *resourceapi.ResourceSlice
							verb := "create"
							if tc.old != nil {
								old = tc.old.DeepCopy()
								old.ResourceVersion, obj.ResourceVersion = "1", "1"
								verb = "update"
							}
							ctx := genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
								APIGroup: "resource.k8s.io", APIVersion: version, Resource: "resourceslices", Verb: verb, IsResourceRequest: true,
							})
							var errs field.ErrorList
							if old == nil {
								Strategy.PrepareForCreate(ctx, obj)
								errs = rest.ValidateCreate(ctx, obj, Strategy)
							} else {
								Strategy.PrepareForUpdate(ctx, obj, old)
								errs = rest.ValidateUpdate(ctx, obj, old, Strategy)
							}
							if (len(errs) == 0) != tc.allowed {
								t.Fatalf("admission allowed=%t, want %t: %v", len(errs) == 0, tc.allowed, errs)
							}
							if shared && (tc.name == "create/neither" || tc.name == "create/both") {
								if len(errs) != 1 || errs[0].Type != field.ErrorTypeInvalid || errs[0].Origin != "union" || errs[0].Field != counterPath.String() || errs[0].IsAlpha() || errs[0].IsBeta() {
									t.Fatalf("expected one authoritative union error at %s, got %v", counterPath, errs)
								}
							}
							if shared && (tc.name == "create/empty-capacity-name" || tc.name == "update/empty-capacity-name") {
								capacityNamePath := counterPath.Child("valueFrom", "capacityName")
								if len(errs) != 1 || errs[0].Type != field.ErrorTypeRequired || errs[0].Field != capacityNamePath.String() || errs[0].IsAlpha() || errs[0].IsBeta() {
									t.Fatalf("expected one authoritative required error at %s, got %v", capacityNamePath, errs)
								}
							}
						})
					}
				})
			}
		}
	}
}

// TestCounterValueValidation checks quantity limits and the generated ratcheting boundaries through admission.
func TestCounterValueValidation(t *testing.T) {
	for _, shared := range []bool{true, false} {
		makeSlice := func(value string) *resourceapi.ResourceSlice {
			obj := slice.DeepCopy()
			obj.UID = "counter-values"
			quantity := resource.MustParse(value)
			if shared {
				obj.Spec.Devices = nil
				obj.Spec.SharedCounters = []resourceapi.CounterSet{{
					Name: "set", Counters: map[string]resourceapi.Counter{"memory": {Value: quantity}},
				}}
			} else {
				obj.Spec.Devices[0].ConsumesCounters = []resourceapi.DeviceCounterConsumption{{
					CounterSet: "set", Counters: map[string]resourceapi.CounterRequirement{"memory": {Value: &quantity}},
				}}
			}
			return obj
		}
		for _, apiVersion := range []string{"v1", "v1beta1", "v1beta2"} {
			valuePath := field.NewPath("spec", "sharedCounters").Index(0).Child("counters").Key("memory").Child("value")
			if !shared {
				valuePath = field.NewPath("spec", "devices").Index(0)
				if apiVersion == "v1beta1" {
					valuePath = valuePath.Child("basic")
				}
				valuePath = valuePath.Child("consumesCounters").Index(0).Child("counters").Key("memory").Child("value")
			}
			for _, beta := range []bool{false, true} {
				t.Run(fmt.Sprintf("shared=%t/%s/beta=%t", shared, apiVersion, beta), func(t *testing.T) {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						apiserverfeatures.DeclarativeValidationBeta: beta,
						features.DRAPartitionableDevices:            true,
						features.DRAConsumableCapacity:              true,
						features.DRASharedConsumableCapacity:        true,
					})
					type scenario struct {
						name         string
						obj, old     *resourceapi.ResourceSlice
						wantNegative bool
					}
					var scenarios []scenario
					for _, value := range []struct {
						quantity string
						negative bool
					}{
						{"0", false}, {"1", false}, {"0.001", false},
						{"9223372036854775807", false}, {"10000000000000000000", false},
						{"-1", true}, {"-500m", true}, {"-10000000000000000000", true},
					} {
						scenarios = append(scenarios,
							scenario{"create/" + value.quantity, makeSlice(value.quantity), nil, value.negative},
							scenario{"update/" + value.quantity, makeSlice(value.quantity), makeSlice("1"), value.negative},
						)
					}
					old := makeSlice("-1")
					unchanged := old.DeepCopy()
					unchanged.Labels = map[string]string{"test": "unchanged-counter"}
					generation := old.DeepCopy()
					generation.Spec.Pool.Generation++
					renamed := old.DeepCopy()
					sibling := old.DeepCopy()
					if shared {
						renamed.Spec.SharedCounters[0].Name = "renamed-set"
						sibling.Spec.SharedCounters[0].Counters["other"] = resourceapi.Counter{Value: resource.MustParse("1")}
					} else {
						renamed.Spec.Devices[0].Name = "renamed-device"
						sibling.Spec.Devices[0].ConsumesCounters[0].Counters["other"] = resourceapi.CounterRequirement{Value: new(resource.MustParse("1"))}
					}
					scenarios = append(scenarios,
						scenario{"stored-negative/label-only", unchanged, old, false},
						scenario{"stored-negative/pool-generation", generation, old, false},
						scenario{"stored-negative/renamed-list-item", renamed, old, true},
						scenario{"stored-negative/sibling-change", sibling, old, !shared},
						scenario{"stored-negative/changed-value", makeSlice("-2"), old, true},
						scenario{"stored-negative/repair-to-zero", makeSlice("0"), old, false},
					)
					for _, tc := range scenarios {
						t.Run(tc.name, func(t *testing.T) {
							obj := tc.obj.DeepCopy()
							verb := "create"
							if tc.old != nil {
								verb = "update"
							}
							ctx := genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
								APIGroup: "resource.k8s.io", APIVersion: apiVersion, Resource: "resourceslices", Verb: verb, IsResourceRequest: true,
							})
							var errs field.ErrorList
							if tc.old == nil {
								Strategy.PrepareForCreate(ctx, obj)
								errs = rest.ValidateCreate(ctx, obj, Strategy)
							} else {
								old := tc.old.DeepCopy()
								old.ResourceVersion, obj.ResourceVersion = "1", "1"
								Strategy.PrepareForUpdate(ctx, obj, old)
								errs = rest.ValidateUpdate(ctx, obj, old, Strategy)
							}
							if !tc.wantNegative {
								if len(errs) != 0 {
									t.Fatalf("expected admission to succeed, got %v", errs)
								}
							} else if len(errs) != 1 || errs[0].Type != field.ErrorTypeInvalid || errs[0].Origin != "minimum" || errs[0].Field != valuePath.String() || errs[0].IsAlpha() || errs[0].IsBeta() {
								t.Fatalf("expected one standard minimum error at %s, got %v", valuePath, errs)
							}
						})
					}
				})
			}
		}
	}
}

// TestSharedCounterPolicyRatcheting preserves stored policies after fractional validation is disabled.
func TestSharedCounterPolicyRatcheting(t *testing.T) {
	for _, apiVersion := range []string{"v1", "v1beta1", "v1beta2"} {
		for _, beta := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/beta=%t", apiVersion, beta), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					apiserverfeatures.DeclarativeValidationBeta: beta,
					features.DRAPartitionableDevices:            true,
					features.DRAConsumableCapacity:              true,
					features.DRASharedConsumableCapacity:        true,
					features.DRAFractionalCapacityRange:         true,
				})
				old := sliceWithPartitionableDevicesSharedCounters.DeepCopy()
				old.Spec.SharedCounters[0].Counters["memory"] = resourceapi.Counter{
					Value: resource.MustParse("1"),
					ConsumptionPolicy: &resourceapi.CapacityRequestPolicy{
						Default: new(resource.MustParse("500m")), ValidValues: []resource.Quantity{resource.MustParse("500m"), resource.MustParse("1")},
					},
				}
				createCtx := genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
					APIGroup: "resource.k8s.io", APIVersion: apiVersion, Resource: "resourceslices", Verb: "create", IsResourceRequest: true,
				})
				Strategy.PrepareForCreate(createCtx, old)
				if errs := rest.ValidateCreate(createCtx, old, Strategy); len(errs) != 0 {
					t.Fatalf("expected fractional policy to be valid before gate change, got %v", errs)
				}
				old.ResourceVersion = "1"
				for _, fractional := range []bool{false, true} {
					t.Run(fmt.Sprintf("fractional=%t", fractional), func(t *testing.T) {
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAFractionalCapacityRange, fractional)
						for _, change := range []string{"label-only", "pool-generation", "sibling-counter", "policy-change", "repair-policy"} {
							t.Run(change, func(t *testing.T) {
								obj := old.DeepCopy()
								counter := obj.Spec.SharedCounters[0].Counters["memory"]
								switch change {
								case "label-only":
									obj.Labels = map[string]string{"test": "unchanged-policy"}
								case "pool-generation":
									obj.Spec.Pool.Generation++
								case "sibling-counter":
									obj.Spec.SharedCounters[0].Counters["other"] = resourceapi.Counter{Value: resource.MustParse("1")}
								case "policy-change":
									counter.ConsumptionPolicy.Default = new(resource.MustParse("1"))
								case "repair-policy":
									counter.ConsumptionPolicy.Default = new(resource.MustParse("1"))
									counter.ConsumptionPolicy.ValidValues = []resource.Quantity{resource.MustParse("1")}
								}
								obj.Spec.SharedCounters[0].Counters["memory"] = counter
								ctx := genericapirequest.WithRequestInfo(context.Background(), &genericapirequest.RequestInfo{
									APIGroup: "resource.k8s.io", APIVersion: apiVersion, Resource: "resourceslices", Verb: "update", IsResourceRequest: true,
								})
								Strategy.PrepareForUpdate(ctx, obj, old)
								errs := rest.ValidateUpdate(ctx, obj, old, Strategy)
								if !fractional && (change == "sibling-counter" || change == "policy-change") {
									if len(errs) != 1 || errs[0].Type != field.ErrorTypeDuplicate || errs[0].Field != "spec.sharedCounters[0].counters[memory].consumptionPolicy.validValues[1]" {
										t.Fatalf("expected policy revalidation to reject rounded duplicate values, got %v", errs)
									}
								} else if len(errs) != 0 {
									t.Fatalf("expected policy update to succeed, got %v", errs)
								}
								if !fractional && (obj.Spec.SharedCounters[0].Counters["memory"].ConsumptionPolicy == nil) {
									t.Fatal("stored consumption policy was dropped after the fractional gate changed")
								}
							})
						}
					})
				}
			})
		}
	}
}
