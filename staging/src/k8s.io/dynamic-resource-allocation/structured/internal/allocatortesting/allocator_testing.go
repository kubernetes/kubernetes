/*
Copyright 2024 The Kubernetes Authors.

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

package allocatortesting

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apitypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/structured/internal"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

type Allocator = internal.Allocator
type DeviceClassLister = internal.DeviceClassLister
type Features = internal.Features
type DeviceID = internal.DeviceID

// types_experimental
type SharedDeviceID = internal.SharedDeviceID
type ConsumedCapacityCollection = internal.ConsumedCapacityCollection
type ConsumedCapacity = internal.ConsumedCapacity
type AllocatedState = internal.AllocatedState

func MakeDeviceID(driver, pool, device string) DeviceID {
	return internal.MakeDeviceID(driver, pool, device)
}

const (
	region1     = "region-1"
	region2     = "region-2"
	node1       = "node-1"
	node2       = "node-2"
	classA      = "class-a"
	classB      = "class-b"
	driverA     = "driver-a"
	driverB     = "driver-b"
	pool1       = "pool-1"
	pool2       = "pool-2"
	pool3       = "pool-3"
	pool4       = "pool-4"
	req0        = "req-0"
	req1        = "req-1"
	req2        = "req-2"
	req3        = "req-3"
	subReq0     = "subReq-0"
	subReq1     = "subReq-1"
	req0SubReq0 = "req-0/subReq-0"
	req0SubReq1 = "req-0/subReq-1"
	req1SubReq0 = "req-1/subReq-0"
	req1SubReq1 = "req-1/subReq-1"
	req2SubReq0 = "req-2/subReq-0"
	claim0      = "claim-0"
	claim1      = "claim-1"
	slice1      = "slice-1"
	slice2      = "slice-2"
	slice3      = "slice-3"
	slice4      = "slice-4"
	device0     = "device-0"
	device1     = "device-1"
	device2     = "device-2"
	device3     = "device-3"
	device4     = "device-4"
	counterSet1 = "counter-set-1"
	counterSet2 = "counter-set-2"
	capacity0   = "capacity-0"
	capacity1   = "capacity-1"
)

var (
	fixedShareID = apitypes.UID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

	zero  = *resource.NewQuantity(0, resource.BinarySI)
	one   = *resource.NewQuantity(1, resource.BinarySI)
	two   = *resource.NewQuantity(2, resource.BinarySI)
	three = *resource.NewQuantity(3, resource.BinarySI)
	four  = *resource.NewQuantity(4, resource.BinarySI)
)

func init() {
	// Bump up the default verbosity for testing. Allocate uses very
	// high thresholds because it is used in the scheduler's per-node
	// filter operation.
	ktesting.DefaultConfig = ktesting.NewConfig(ktesting.Verbosity(7))
	ktesting.DefaultConfig.AddFlags(flag.CommandLine)
}

// Test objects generators

const (
	fieldNameKey     = "metadata.name"
	regionKey        = "region"
	planetKey        = "planet"
	planetValueEarth = "earth"
)

// generate a node object given a name and a region
func node(name, region string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				regionKey: region,
				planetKey: planetValueEarth,
			},
		},
	}
}

// generate a DeviceClass object with the given name and a driver CEL selector.
// driver name is assumed to be the same as the class name.
func class(name, driver string) *resourceapi.DeviceClass {
	return &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.DeviceClassSpec{
			Selectors: []resourceapi.DeviceSelector{
				{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.driver == "%s"`, driver),
					},
				},
			},
		},
	}
}

// generate a DeviceClass object with the given name and a driver CEL selector.
// driver name is assumed to be the same as the class name.
// shared condition is explicitly set.
func classWithAllowMultipleAllocations(name, driver string, multialloc bool) *resourceapi.DeviceClass {
	return &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.DeviceClassSpec{
			Selectors: []resourceapi.DeviceSelector{
				{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.driver == "%s" && device.allowMultipleAllocations == %v`, driver, multialloc),
					},
				},
			},
		},
	}
}

// generate a DeviceConfiguration object with the given driver and attribute.
func deviceConfiguration(driver, attribute string) resourceapi.DeviceConfiguration {
	return resourceapi.DeviceConfiguration{
		Opaque: &resourceapi.OpaqueDeviceConfiguration{
			Driver: driver,
			Parameters: runtime.RawExtension{
				Raw: []byte(fmt.Sprintf("{\"%s\":\"%s\"}", attribute, attribute+"Value")),
			},
		},
	}
}

// generate a DeviceClass object with the given name and attribute.
// attribute is used to generate device configuration parameters in a form of JSON {attribute: attributeValue}.
func classWithConfig(name, driver, attribute string) *resourceapi.DeviceClass {
	class := class(name, driver)
	class.Spec.Config = []resourceapi.DeviceClassConfiguration{
		{
			DeviceConfiguration: deviceConfiguration(driver, attribute),
		},
	}
	return class
}

// generate a ResourceClaim object with the given name and device requests.
func claimWithRequests(name string, constraints []resourceapi.DeviceConstraint, requests ...resourceapi.DeviceRequest) wrapResourceClaim {
	return wrapResourceClaim{&resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests:    requests,
				Constraints: constraints,
			},
		},
	}}
}

// generate a DeviceRequest object with the given name, class and selectors.
func request(name, class string, count int64, selectors ...resourceapi.DeviceSelector) resourceapi.DeviceRequest {
	return resourceapi.DeviceRequest{
		Name: name,
		Exactly: &resourceapi.ExactDeviceRequest{
			Count:           count,
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
			DeviceClassName: class,
			Selectors:       selectors,
		},
	}
}

func deviceRequest(name, class string, count int64) wrapDeviceRequest {
	return wrapDeviceRequest{
		request(name, class, count),
	}
}

func allDeviceRequest(name, class string) wrapDeviceRequest {
	return wrapDeviceRequest{
		resourceapi.DeviceRequest{
			Name: name,
			Exactly: &resourceapi.ExactDeviceRequest{
				AllocationMode:  resourceapi.DeviceAllocationModeAll,
				DeviceClassName: class,
			},
		},
	}
}

func subRequest(name, class string, count int64, fields ...any) wrapDeviceSubRequest {
	subRequest := resourceapi.DeviceSubRequest{
		Name:            name,
		Count:           count,
		AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
		DeviceClassName: class,
	}

	for _, field := range fields {
		switch typedField := field.(type) {
		case resourceapi.DeviceSelector:
			subRequest.Selectors = append(subRequest.Selectors, typedField)
		case resourceapi.DeviceToleration:
			subRequest.Tolerations = append(subRequest.Tolerations, typedField)
		default:
			panic(fmt.Sprintf("unsupported field for DeviceSubRequest: %T", field))
		}
	}

	return wrapDeviceSubRequest{subRequest}
}

type wrapDeviceRequest struct{ resourceapi.DeviceRequest }

func (in wrapDeviceRequest) obj() resourceapi.DeviceRequest {
	return in.DeviceRequest
}

func (in wrapDeviceRequest) withCapacityRequest(quantity *resource.Quantity) wrapDeviceRequest {
	out := in.DeepCopy()
	out.Exactly.Capacity = capacityRequests(quantity)
	return wrapDeviceRequest{*out}
}

func (in wrapDeviceRequest) withSelectors(selectors ...resourceapi.DeviceSelector) wrapDeviceRequest {
	out := in.DeepCopy()
	out.Exactly.Selectors = selectors
	return wrapDeviceRequest{*out}
}

type wrapDeviceSubRequest struct{ resourceapi.DeviceSubRequest }

func (in wrapDeviceSubRequest) obj() resourceapi.DeviceSubRequest {
	return in.DeviceSubRequest
}

func (in wrapDeviceSubRequest) withAllocationMode(mode resourceapi.DeviceAllocationMode) wrapDeviceSubRequest {
	out := in.DeepCopy()
	out.AllocationMode = mode
	return wrapDeviceSubRequest{*out}
}

func (in wrapDeviceSubRequest) withCapacityRequest(quantity *resource.Quantity) wrapDeviceSubRequest {
	out := in.DeepCopy()
	out.Capacity = capacityRequests(quantity)
	return wrapDeviceSubRequest{*out}
}

// genereate a DeviceRequest with the given name and list of prioritized requests.
func requestWithPrioritizedList(name string, prioritizedRequests ...wrapDeviceSubRequest) wrapDeviceRequest {
	var subreqs []resourceapi.DeviceSubRequest
	for _, subreq := range prioritizedRequests {
		subreqs = append(subreqs, subreq.obj())
	}
	return wrapDeviceRequest{resourceapi.DeviceRequest{
		Name:           name,
		FirstAvailable: subreqs,
	}}
}

// generate a ResourceClaim object with the given name and with a single request with given class and constraints.
func claimWithRequest(name, req, class string, constraints ...resourceapi.DeviceConstraint) wrapResourceClaim {
	claim := claimWithRequests(name, constraints, request(req, class, 1))
	return claim
}

// generate a ResourceClaim object with the given name.
func claim(name string) wrapResourceClaim {
	return wrapResourceClaim{&resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}}
}

type wrapResourceClaim struct{ *resourceapi.ResourceClaim }

func (in wrapResourceClaim) obj() *resourceapi.ResourceClaim {
	return in.ResourceClaim
}

func (in wrapResourceClaim) withTolerations(tolerations ...resourceapi.DeviceToleration) wrapResourceClaim {
	out := in.DeepCopy()
	for i := range out.Spec.Devices.Requests {
		out.Spec.Devices.Requests[i].Exactly.Tolerations = append(out.Spec.Devices.Requests[i].Exactly.Tolerations, tolerations...)
	}
	return wrapResourceClaim{out}
}

func (in wrapResourceClaim) withRequests(requests ...wrapDeviceRequest) wrapResourceClaim {
	out := in.DeepCopy()
	out.Spec.Devices.Requests = make([]resourceapi.DeviceRequest, len(requests))
	for i, request := range requests {
		out.Spec.Devices.Requests[i] = resourceapi.DeviceRequest(request.DeviceRequest)
	}
	return wrapResourceClaim{out}
}

func (in wrapResourceClaim) withConstraints(constraints ...resourceapi.DeviceConstraint) wrapResourceClaim {
	out := in.DeepCopy()
	out.Spec.Devices.Constraints = constraints
	return wrapResourceClaim{out}
}

func (in wrapResourceClaim) withConfigs(configs []resourceapi.DeviceClaimConfiguration) wrapResourceClaim {
	out := in.DeepCopy()
	out.Spec.Devices.Config = configs
	return wrapResourceClaim{out}
}

// generate a ResourceClaim object with the given name, request, class, and attribute.
// attribute is used to generate parameters in a form of JSON {attribute: attributeValue}.
func claimWithDeviceConfig(name, request, class, driver, attribute string) wrapResourceClaim {
	claim := claimWithRequest(name, request, class)
	claim.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{
		{
			DeviceConfiguration: deviceConfiguration(driver, attribute),
		},
	}
	return claim
}

func deviceClaimConfig(requests []string, deviceConfig resourceapi.DeviceConfiguration) resourceapi.DeviceClaimConfiguration {
	return resourceapi.DeviceClaimConfiguration{
		Requests:            requests,
		DeviceConfiguration: deviceConfig,
	}
}

const (
	fromCounters = "fromCounters"
)

// generate a Device object with the given name, capacity and attributes.
func device(name string, capacity any, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) wrapDevice {
	device := resourceapi.Device{
		Name:       name,
		Attributes: attributes,
	}

	var capacityFromCounters bool
	switch capacity := capacity.(type) {
	case map[resourceapi.QualifiedName]resource.Quantity:
		device.Capacity = toDeviceCapacity(capacity)
	case string:
		if capacity == fromCounters {
			capacityFromCounters = true
		} else {
			panic(fmt.Sprintf("unexpected capacity value %q", capacity))
		}
	case nil:
		// nothing to do
	default:
		panic(fmt.Sprintf("unexpected capacity type %T: %+v", capacity, capacity))
	}

	return wrapDevice{Device: device, capacityFromCounters: capacityFromCounters}
}

type wrapDevice struct {
	resourceapi.Device
	capacityFromCounters bool
}

func (in wrapDevice) obj() resourceapi.Device {
	return in.Device
}

func (in wrapDevice) withTaints(taints ...resourceapi.DeviceTaint) wrapDevice {
	inDevice := resourceapi.Device(in.Device)
	device := inDevice.DeepCopy()
	device.Taints = append(device.Taints, taints...)
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withDeviceCounterConsumption(deviceCounterConsumption ...resourceapi.DeviceCounterConsumption) wrapDevice {
	inDevice := in.Device
	device := inDevice.DeepCopy()
	device.ConsumesCounters = append(device.ConsumesCounters, deviceCounterConsumption...)
	if in.capacityFromCounters {
		c := make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity)
		for _, dcc := range device.ConsumesCounters {
			for name, cap := range dcc.Counters {
				ccap := resourceapi.DeviceCapacity{
					Value: cap.Value,
				}
				c[resourceapi.QualifiedName(name)] = ccap
			}
		}
		device.Capacity = c
	}
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withNodeSelection(nodeSelection any) wrapDevice {
	inDevice := in.Device
	device := inDevice.DeepCopy()
	switch nodeSelection := nodeSelection.(type) {
	case *v1.NodeSelector:
		device.NodeSelector = nodeSelection
	case string:
		if nodeSelection == nodeSelectionAll {
			device.AllNodes = func() *bool {
				r := true
				return &r
			}()
		} else if nodeSelection == nodeSelectionPerDevice {
			panic("nodeSelectionPerDevice is not supported for devices")
		} else {
			device.NodeName = &nodeSelection
		}
	default:
		panic(fmt.Sprintf("unexpected nodeSelection type %T: %+v", nodeSelection, nodeSelection))
	}
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withBindingConditions(bindingConditions, bindingFailureConditions []string) wrapDevice {
	inDevice := in.Device
	device := inDevice.DeepCopy()
	device.BindingConditions = bindingConditions
	device.BindingFailureConditions = bindingFailureConditions
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withBindsToNode(bindsToNode bool) wrapDevice {
	inDevice := in.Device
	device := inDevice.DeepCopy()
	device.BindsToNode = ptr.To(bindsToNode)
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withAllowMultipleAllocations() wrapDevice {
	inDevice := in.Device
	device := inDevice.DeepCopy()
	device.AllowMultipleAllocations = ptr.To(true)
	return wrapDevice{Device: *device}
}

// withCapacityRequestPolicyRange adds capacity with default requestPolicy (2,2,4)
func (in wrapDevice) withCapacityRequestPolicyRange(capacity map[resourceapi.QualifiedName]resource.Quantity) wrapDevice {
	inDevice := in.Device
	device := inDevice.DeepCopy()
	if device.Capacity == nil {
		device.Capacity = make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, len(capacity))
	}
	for name, quantity := range capacity {
		device.Capacity[name] = resourceapi.DeviceCapacity{
			Value: quantity,
			RequestPolicy: &resourceapi.CapacityRequestPolicy{
				Default: ptr.To(two),
				ValidRange: &resourceapi.CapacityRequestPolicyRange{
					Min:  ptr.To(two),
					Step: ptr.To(two),
					Max:  ptr.To(four),
				},
			},
		}
	}
	return wrapDevice{Device: *device}
}

// withCapacityRequestPolicyValidValues adds capacity with default valid values (1)
func (in wrapDevice) withCapacityRequestPolicyValidValues(defaultValue resource.Quantity, capacity map[resourceapi.QualifiedName]resource.Quantity, additionalValidValues []resource.Quantity) wrapDevice {
	inDevice := in.Device
	device := inDevice.DeepCopy()
	if device.Capacity == nil {
		device.Capacity = make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, len(capacity))
	}
	for name, quantity := range capacity {
		validValues := []resource.Quantity{defaultValue}            // default is minimum
		validValues = append(validValues, additionalValidValues...) // append with additional valid values
		device.Capacity[name] = resourceapi.DeviceCapacity{
			Value: quantity,
			RequestPolicy: &resourceapi.CapacityRequestPolicy{
				Default:     ptr.To(defaultValue),
				ValidValues: validValues,
			},
		}
	}
	return wrapDevice{Device: *device}
}

func deviceCounterConsumption(counterSet string, counters map[string]resource.Quantity) resourceapi.DeviceCounterConsumption {
	return resourceapi.DeviceCounterConsumption{
		CounterSet: counterSet,
		Counters:   toCounters(counters),
	}
}

const (
	nodeSelectionAll       = "nodeSelectionAll"
	nodeSelectionPerDevice = "nodeSelectionPerDevice"
)

// generate a ResourceSlice object with the given name, node,
// driver and pool names and generation.
// The nodeSelection parameter may be a string with the value
// nodeSelectionAll for all nodes, the value nodeSelectionPerDevice
// for per device node selection, or any other value to set the
// node name. Providing a node selectors sets the NodeSelector field.
func slice(name string, nodeSelection, pool any, driver string) *resourceapi.ResourceSlice {
	var resourcePool resourceapi.ResourcePool
	switch poolSelection := pool.(type) {
	case resourceapi.ResourcePool:
		resourcePool = poolSelection
	case string:
		resourcePool = resourceapi.ResourcePool{
			Name:               poolSelection,
			ResourceSliceCount: 1,
			Generation:         1,
		}
	default:
		panic(fmt.Sprintf("unexpected pool type %T: %+v", poolSelection, poolSelection))
	}

	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver: driver,
			Pool:   resourcePool,
		},
	}
	switch nodeSelection := nodeSelection.(type) {
	case *v1.NodeSelector:
		slice.Spec.NodeSelector = nodeSelection
	case string:
		if nodeSelection == nodeSelectionAll {
			slice.Spec.AllNodes = ptr.To(true)
		} else if nodeSelection == nodeSelectionPerDevice {
			slice.Spec.PerDeviceNodeSelection = func() *bool {
				r := true
				return &r
			}()
		} else {
			slice.Spec.NodeName = ptr.To(nodeSelection)
		}
	default:
		panic(fmt.Sprintf("unexpected nodeSelection type %T: %+v", nodeSelection, nodeSelection))
	}

	return slice
}

func sliceWithDevices(name string, nodeSelection, pool any, driver string, devices ...wrapDevice) wrapResourceSliceWithDevices {
	slice := slice(name, nodeSelection, pool, driver)
	for _, device := range devices {
		slice.Spec.Devices = append(slice.Spec.Devices, resourceapi.Device(device.Device))
	}
	return wrapResourceSliceWithDevices{ResourceSlice: slice}
}

type wrapResourceSliceWithDevices struct {
	*resourceapi.ResourceSlice
}

func (in wrapResourceSliceWithDevices) obj() *resourceapi.ResourceSlice {
	return in.ResourceSlice
}

func sliceWithCounterSets(name string, nodeSelection, pool any, driver string, counterSets ...resourceapi.CounterSet) wrapResourceSliceWithCounterSets {
	slice := slice(name, nodeSelection, pool, driver)
	slice.Spec.SharedCounters = counterSets
	return wrapResourceSliceWithCounterSets{ResourceSlice: slice}
}

type wrapResourceSliceWithCounterSets struct {
	*resourceapi.ResourceSlice
}

func (in wrapResourceSliceWithCounterSets) obj() *resourceapi.ResourceSlice {
	return in.ResourceSlice
}

func resourcePool(name string, count int64) resourceapi.ResourcePool {
	return resourceapi.ResourcePool{
		Name:               name,
		ResourceSliceCount: count,
		Generation:         1,
	}
}

func deviceAllocationResult(request, driver, pool, device string, adminAccess bool, tolerations ...resourceapi.DeviceToleration) resourceapi.DeviceRequestAllocationResult {
	r := resourceapi.DeviceRequestAllocationResult{
		Request:     request,
		Driver:      driver,
		Pool:        pool,
		Device:      device,
		Tolerations: tolerations,
	}
	if adminAccess {
		r.AdminAccess = &adminAccess
	}
	return r
}

// deviceRequestAllocationResult can replace deviceAllocationResult
func deviceRequestAllocationResult(request, driver, pool, device string) wrapDeviceRequestAllocationResult {
	return wrapDeviceRequestAllocationResult{
		&resourceapi.DeviceRequestAllocationResult{
			Request: request,
			Driver:  driver,
			Pool:    pool,
			Device:  device,
		},
	}
}

type wrapDeviceRequestAllocationResult struct {
	*resourceapi.DeviceRequestAllocationResult
}

func (in wrapDeviceRequestAllocationResult) obj() *resourceapi.DeviceRequestAllocationResult {
	return in.DeviceRequestAllocationResult
}

func (in wrapDeviceRequestAllocationResult) withConsumedCapacity(shareID *apitypes.UID,
	consumedCapacity map[resourceapi.QualifiedName]resource.Quantity) resourceapi.DeviceRequestAllocationResult {
	out := in.DeepCopy()
	out.ShareID = shareID
	out.ConsumedCapacity = consumedCapacity
	return *out
}

func capacityRequests(request *resource.Quantity) *resourceapi.CapacityRequirements {
	return &resourceapi.CapacityRequirements{
		Requests: requirements(request),
	}
}

func requirements(request *resource.Quantity) map[resourceapi.QualifiedName]resource.Quantity {
	r := make(map[resourceapi.QualifiedName]resource.Quantity, 0)
	if request != nil {
		r[capacity0] = *request
	}
	return r
}

func multipleDeviceAllocationResults(request, driver, pool string, count, startIndex int) []resourceapi.DeviceRequestAllocationResult {
	var results []resourceapi.DeviceRequestAllocationResult
	for i := startIndex; i < startIndex+count; i++ {
		results = append(results, deviceAllocationResult(request, driver, pool, fmt.Sprintf("device-%d", i), false))
	}
	return results
}

// nodeLabelSelector creates a node selector with a label match for "key" in "values".
func nodeLabelSelector(key string, values ...string) *v1.NodeSelector {
	requirements := []v1.NodeSelectorRequirement{{
		Key:      key,
		Operator: v1.NodeSelectorOpIn,
		Values:   values,
	}}
	selector := &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{MatchExpressions: requirements}}}
	return selector
}

// localNodeSelector returns a node selector for a specific node.
func localNodeSelector(nodeName string) *v1.NodeSelector {
	selector := nodeLabelSelector(fieldNameKey, nodeName)
	// Swap the requirements: we need to select by field, not label.
	selector.NodeSelectorTerms[0].MatchFields, selector.NodeSelectorTerms[0].MatchExpressions = selector.NodeSelectorTerms[0].MatchExpressions, selector.NodeSelectorTerms[0].MatchFields
	return selector
}

// allocationResult returns a matcher for one AllocationResult pointer with a list of
// embedded device allocation results. The order of those results doesn't matter.
func allocationResult(selector *v1.NodeSelector, results ...resourceapi.DeviceRequestAllocationResult) types.GomegaMatcher {
	return gstruct.MatchFields(0, gstruct.Fields{
		"Devices": gstruct.MatchFields(0, gstruct.Fields{
			"Results": gomega.ConsistOf(results), // Order is irrelevant.
			"Config":  gomega.BeNil(),
		}),
		"NodeSelector":        matchNodeSelector(selector),
		"AllocationTimestamp": gomega.BeNil(),
	})
}

// matchNodeSelector returns a matcher for a node selector. The order
// of terms, requirements, and values is irrelevant.
func matchNodeSelector(selector *v1.NodeSelector) types.GomegaMatcher {
	if selector == nil {
		return gomega.BeNil()
	}
	return gomega.HaveField("NodeSelectorTerms", matchNodeSelectorTerms(selector.NodeSelectorTerms))
}

func matchNodeSelectorTerms(terms []v1.NodeSelectorTerm) types.GomegaMatcher {
	var matchTerms []types.GomegaMatcher
	for _, term := range terms {
		matchTerms = append(matchTerms, matchNodeSelectorTerm(term))
	}
	return gomega.ConsistOf(matchTerms)
}

func matchNodeSelectorTerm(term v1.NodeSelectorTerm) types.GomegaMatcher {
	return gstruct.MatchFields(0, gstruct.Fields{
		"MatchExpressions": matchNodeSelectorRequirements(term.MatchExpressions),
		"MatchFields":      matchNodeSelectorRequirements(term.MatchFields),
	})
}

func matchNodeSelectorRequirements(requirements []v1.NodeSelectorRequirement) types.GomegaMatcher {
	var matchRequirements []types.GomegaMatcher
	for _, requirement := range requirements {
		matchRequirements = append(matchRequirements, matchNodeSelectorRequirement(requirement))
	}
	return gomega.ConsistOf(matchRequirements)
}

func matchNodeSelectorRequirement(requirement v1.NodeSelectorRequirement) types.GomegaMatcher {
	return gstruct.MatchFields(0, gstruct.Fields{
		"Key":      gomega.Equal(requirement.Key),
		"Operator": gomega.Equal(requirement.Operator),
		"Values":   gomega.ConsistOf(requirement.Values),
	})
}

func allocationResultWithConfig(selector *v1.NodeSelector, driver string, source resourceapi.AllocationConfigSource, attribute string, results ...resourceapi.DeviceRequestAllocationResult) resourceapi.AllocationResult {
	return resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: results,
			Config: []resourceapi.DeviceAllocationConfiguration{
				{
					Source:              source,
					DeviceConfiguration: deviceConfiguration(driver, attribute),
				},
			},
		},
		NodeSelector: selector,
	}
}

func allocationResultWithConfigs(selector *v1.NodeSelector, results []resourceapi.DeviceRequestAllocationResult, configs []resourceapi.DeviceAllocationConfiguration) resourceapi.AllocationResult {
	return resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: results,
			Config:  configs,
		},
		NodeSelector: selector,
	}
}

// Helpers

// convert a list of objects to a slice
func objects[T any](objs ...T) []T {
	return objs
}

// convert a list of wrapper objects to a slice
func unwrap[T any, O wrapper[T]](objs ...O) []T {
	out := make([]T, len(objs))
	for i, obj := range objs {
		out[i] = obj.obj()
	}
	return out
}

type wrapper[T any] interface {
	obj() T
}

// convert a list of wrapper objects to a slice
func unwrapResourceSlices(objs ...resourceSliceWrapper) []*resourceapi.ResourceSlice {
	out := make([]*resourceapi.ResourceSlice, len(objs))
	for i, obj := range objs {
		out[i] = obj.obj()
	}
	return out
}

type resourceSliceWrapper interface {
	obj() *resourceapi.ResourceSlice
}

// generate a ResourceSlice object with the given parameters and no devices
func sliceWithNoDevices(name string, nodeSelection, pool any, driver string) wrapResourceSliceWithDevices {
	return sliceWithDevices(name, nodeSelection, pool, driver)
}

// generate a ResourceSlice object with the given parameters and one device "device-1"
func sliceWithOneDevice(name string, nodeSelection, pool any, driver string) wrapResourceSliceWithDevices {
	return sliceWithDevices(name, nodeSelection, pool, driver, device(device1, nil, nil))
}

// generate a ResourceSclie object with the given parameters and the specified number of devices.
func sliceWithMultipleDevices(name string, nodeSelection, pool any, driver string, count int) wrapResourceSliceWithDevices {
	var devices []wrapDevice
	for i := 0; i < count; i++ {
		devices = append(devices, device(fmt.Sprintf("device-%d", i), nil, nil))
	}
	return sliceWithDevices(name, nodeSelection, pool, driver, devices...)
}

func counterSet(name string, counters map[string]resource.Quantity) resourceapi.CounterSet {
	return resourceapi.CounterSet{
		Name:     name,
		Counters: toCounters(counters),
	}
}

func toDeviceCapacity(capacity map[resourceapi.QualifiedName]resource.Quantity) map[resourceapi.QualifiedName]resourceapi.DeviceCapacity {
	out := make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, len(capacity))
	for name, quantity := range capacity {
		out[name] = resourceapi.DeviceCapacity{Value: quantity}
	}
	return out
}

func toCounters(counters map[string]resource.Quantity) map[string]resourceapi.Counter {
	out := make(map[string]resourceapi.Counter, len(counters))
	for name, quantity := range counters {
		out[string(name)] = resourceapi.Counter{Value: quantity}
	}
	return out
}

// deviceRequestAllocationResultWithBindingConditions returns an DeviceRequestAllocationResult object for testing purposes,
// specifying the driver, pool, device, usage restriction, binding conditions,
// binding failure conditions, and binding timeout.
func deviceRequestAllocationResultWithBindingConditions(request, driver, pool, device string, bindingConditions, bindingFailureConditions []string, tolerations ...resourceapi.DeviceToleration) resourceapi.DeviceRequestAllocationResult {
	return resourceapi.DeviceRequestAllocationResult{
		Request:                  request,
		Driver:                   driver,
		Pool:                     pool,
		Device:                   device,
		BindingConditions:        bindingConditions,
		BindingFailureConditions: bindingFailureConditions,
		Tolerations:              tolerations,
	}
}

// TestAllocator runs as many of the shared tests against a specific allocator implementation as possible.
// Test cases which depend on features that are not supported by the implementation are silently skipped.
func TestAllocator(t *testing.T,
	supportedFeatures Features,
	newAllocator func(
		ctx context.Context,
		features Features,
		allocateState AllocatedState,
		classLister DeviceClassLister,
		slices []*resourceapi.ResourceSlice,
		celCache *cel.Cache,
	) (Allocator, error)) {
	nonExistentAttribute := resourceapi.FullyQualifiedName(driverA + "/" + "NonExistentAttribute")
	boolAttribute := resourceapi.FullyQualifiedName(driverA + "/" + "boolAttribute")
	stringAttribute := resourceapi.FullyQualifiedName(driverA + "/" + "stringAttribute")
	versionAttribute := resourceapi.FullyQualifiedName(driverA + "/" + "driverVersion")
	intAttribute := resourceapi.FullyQualifiedName(driverA + "/" + "numa")
	taintKey := "taint-key"
	taintValue := "taint-value"
	taintValue2 := "taint-value-2"
	taintNone := resourceapi.DeviceTaint{
		Key:    taintKey,
		Value:  taintValue,
		Effect: resourceapi.DeviceTaintEffectNone,
	}
	taintNoSchedule := resourceapi.DeviceTaint{
		Key:    taintKey,
		Value:  taintValue,
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	}
	taintNoExecute := resourceapi.DeviceTaint{
		Key:    taintKey,
		Value:  taintValue2,
		Effect: resourceapi.DeviceTaintEffectNoExecute,
	}
	tolerationNoSchedule := resourceapi.DeviceToleration{
		Operator: resourceapi.DeviceTolerationOpExists,
		Key:      taintKey,
		Effect:   resourceapi.DeviceTaintEffectNoSchedule,
	}
	tolerationNoExecute := resourceapi.DeviceToleration{
		Operator: resourceapi.DeviceTolerationOpEqual,
		Key:      taintKey,
		Value:    taintValue2,
		Effect:   resourceapi.DeviceTaintEffectNoExecute,
	}
	tolerationNoScheduleKey1 := resourceapi.DeviceToleration{
		Operator: resourceapi.DeviceTolerationOpExists,
		Key:      "key1",
		Effect:   resourceapi.DeviceTaintEffectNoSchedule,
	}

	testcases := map[string]struct {
		features                 Features
		claimsToAllocate         []wrapResourceClaim
		allocatedDevices         []DeviceID
		allocatedSharedDeviceIDs sets.Set[SharedDeviceID]
		allocatedCapacityDevices ConsumedCapacityCollection
		classes                  []*resourceapi.DeviceClass
		slices                   []*resourceapi.ResourceSlice
		node                     *v1.Node

		expectResults []any
		expectError   types.GomegaMatcher // can be used to check for no error or match specific error

		// Test case setting expectNumAllocateOneInvocations do not run against the "stable" variant of the allocator,
		// which doesn't provide the stats and also falls over with excessive runtime for them.
		expectNumAllocateOneInvocations int64
	}{

		"empty": {},
		"simple": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"other-node": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, node1, pool1, driverB),
				sliceWithOneDevice(slice2, node2, pool2, driverA),
			),
			node: node(node2, region2),

			expectResults: []any{allocationResult(
				localNodeSelector(node2),
				deviceAllocationResult(req0, driverA, pool2, device1, false),
			)},
		},
		"small-and-large": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				nil,
				request(req0, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("1Gi")) >= 0`, driverA),
					}}),
				request(req1, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
					}}),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("1Gi"),
				}, nil),
				device(device2, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("2Gi"),
				}, nil),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1, driverA, pool1, device2, false),
			)},
		},
		"small-and-large-backtrack-requests": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				nil,
				request(req0, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("1Gi")) >= 0`, driverA),
					}}),
				request(req1, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
					}}),
			)),
			classes: objects(class(classA, driverA)),
			// Reversing the order in which the devices are listed causes the "large" device to
			// be allocated for the "small" request, leaving the "large" request unsatisfied.
			// The initial decision needs to be undone before a solution is found.
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device2, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("2Gi"),
				}, nil),
				device(device1, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("1Gi"),
				}, nil),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1, driverA, pool1, device2, false),
			)},
		},
		"small-and-large-backtrack-claims": {
			claimsToAllocate: objects(
				claimWithRequests(
					claim0,
					nil,
					request(req0, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("1Gi")) >= 0`, driverA),
						}})),
				claimWithRequests(
					claim1,
					nil,
					request(req1, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
						}}),
				)),
			classes: objects(class(classA, driverA)),
			// Reversing the order in which the devices are listed causes the "large" device to
			// be allocated for the "small" request, leaving the "large" request unsatisfied.
			// The initial decision needs to be undone before a solution is found.
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device2, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("2Gi"),
				}, nil),
				device(device1, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("1Gi"),
				}, nil),
			)),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverA, pool1, device1, false)),
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req1, driverA, pool1, device2, false)),
			},
		},
		"devices-split-across-different-slices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					Count:           2,
					AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
					DeviceClassName: classA,
				},
			})),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, node1, pool1, driverA),
				sliceWithOneDevice(slice2, node1, pool2, driverA),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool2, device1, false),
			)},
		},
		"obsolete-slice": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices("slice-1-obsolete", node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil),
				),
				func() wrapResourceSliceWithDevices {
					slice := sliceWithDevices("slice-1-obsolete", node1, resourcePool(pool1, 2), driverA,
						device(device2, nil, nil),
					)
					// This makes the other slice obsolete.
					slice.Spec.Pool.Generation++
					return slice
				}(),
			),
			node: node(node1, region1),

			expectResults: []any{},
		},
		"duplicate-slice": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: func() []*resourceapi.ResourceSlice {
				// This simulates the problem that can
				// (theoretically) occur when the resource
				// slice controller wants to publish a pool
				// with two slices but ends up creating some
				// identical slices under different names
				// because its informer cache was out-dated on
				// another sync (see
				// resourceslicecontroller.go).
				sliceA := sliceWithOneDevice(slice1, node1, pool1, driverA).obj()
				sliceA.Spec.Pool.ResourceSliceCount = 2
				sliceB := sliceA.DeepCopy()
				sliceB.Name += "-2"
				return []*resourceapi.ResourceSlice{sliceA, sliceB}
			}(),
			node: node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("invalid resource pools were encountered")),
		},
		"no-slices": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           nil,
			node:             node(node1, region1),

			expectResults: nil,
		},
		"not-enough-suitable-devices": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA), claimWithRequest(claim0, req1, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),

			node: node(node1, region1),

			expectResults: nil,
		},
		"no-classes": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          nil,
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("could not retrieve device class class-a")),
		},
		"unknown-class": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, "unknown-class")),
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("could not retrieve device class unknown-class")),
		},
		"empty-class": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, "")),
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: missing device class name (unsupported request type?)")),
		},
		"no-claims-to-allocate": {
			claimsToAllocate: nil,
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: nil,
		},
		"all-devices-single": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					DeviceClassName: classA,
				},
			})),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"all-devices-many": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					DeviceClassName: classA,
				},
			})),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, node1, pool1, driverA),
				sliceWithOneDevice(slice1, node1, pool2, driverA),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool2, device1, false),
			)},
		},
		"all-devices-of-the-incomplete-pool": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					Count:           1,
					DeviceClassName: classA,
				},
			})),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				func() wrapResourceSliceWithDevices {
					slice := sliceWithOneDevice(slice1, node1, pool1, driverA)
					// This makes the pool incomplete, one other slice is missing.
					slice.Spec.Pool.ResourceSliceCount++
					return slice
				}(),
			),
			node: node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: asks for all devices, but resource pool driver-a/pool-1 is currently being updated")),
		},
		"all-devices-plus-another": {
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeAll,
						DeviceClassName: classA,
					},
				}),
				claimWithRequests(claim1, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           1,
						DeviceClassName: classB,
					},
				}),
			),
			classes: objects(
				class(classA, driverA),
				class(classB, driverB),
			),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, node1, pool1, driverA),
				sliceWithOneDevice(slice1, node1, pool1, driverB),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverB, pool1, device1, false),
				),
			},
		},
		"all-devices-plus-another-reversed": {
			claimsToAllocate: objects(
				claimWithRequests(claim1, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           1,
						DeviceClassName: classB,
					},
				}),
				claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeAll,
						DeviceClassName: classA,
					},
				}),
			),
			classes: objects(
				class(classA, driverA),
				class(classB, driverB),
			),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, node1, pool1, driverA),
				sliceWithOneDevice(slice1, node1, pool1, driverB),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverB, pool1, device1, false),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
				),
			},
		},
		"all-devices-many-plus-another": {
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeAll,
						DeviceClassName: classA,
					},
				}),
				claimWithRequests(claim1, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           1,
						DeviceClassName: classB,
					},
				}),
			),
			classes: objects(
				class(classA, driverA),
				class(classB, driverB),
			),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil),
					device(device2, nil, nil),
				),
				sliceWithOneDevice(slice1, node1, pool1, driverB),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
					deviceAllocationResult(req0, driverA, pool1, device2, false),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverB, pool1, device1, false),
				),
			},
		},
		"all-devices-many-plus-another-reversed": {
			claimsToAllocate: objects(
				claimWithRequests(claim1, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           1,
						DeviceClassName: classB,
					},
				}),
				claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeAll,
						DeviceClassName: classA,
					},
				}),
			),
			classes: objects(
				class(classA, driverA),
				class(classB, driverB),
			),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil),
					device(device2, nil, nil),
				),
				sliceWithOneDevice(slice1, node1, pool1, driverB),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverB, pool1, device1, false),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
					deviceAllocationResult(req0, driverA, pool1, device2, false),
				),
			},
		},
		"all-devices-no-solution": {
			// One device, two claims both trying to allocate it.
			claimsToAllocate: objects(
				claimWithRequests(claim1, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           1,
						DeviceClassName: classA,
					},
				}),
				claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeAll,
						DeviceClassName: classA,
					},
				}),
			),
			classes: objects(
				class(classA, driverA),
			),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, node1, pool1, driverA),
			),
			node: node(node1, region1),
		},
		"all-devices-no-solution-reversed": {
			// One device, two claims both trying to allocate it.
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeAll,
						DeviceClassName: classA,
					},
				}),
				claimWithRequests(claim1, nil, resourceapi.DeviceRequest{
					Name: req0,
					Exactly: &resourceapi.ExactDeviceRequest{
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           1,
						DeviceClassName: classA,
					},
				}),
			),
			classes: objects(
				class(classA, driverA),
			),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, node1, pool1, driverA),
			),
			node: node(node1, region1),
		},
		"all-devices-slice-without-devices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					DeviceClassName: classA,
				},
			})),
			classes:       objects(class(classA, driverA)),
			slices:        unwrapResourceSlices(sliceWithNoDevices(slice1, node1, pool1, driverA)),
			node:          node(node1, region1),
			expectResults: nil,
		},
		"all-devices-no-slices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					DeviceClassName: classA,
				},
			})),
			classes:       objects(class(classA, driverA)),
			slices:        nil,
			node:          node(node1, region1),
			expectResults: nil,
		},
		"all-devices-some-allocated": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					DeviceClassName: classA,
				},
			})),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
			},
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1, nil, nil), device(device2, nil, nil)),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		"all-devices-some-allocated-admin-access": {
			features: Features{
				AdminAccess: true,
			},
			claimsToAllocate: func() []wrapResourceClaim {
				c := claimWithRequest(claim0, req0, classA)
				c.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				c.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationModeAll
				return []wrapResourceClaim{c}
			}(),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
			},
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1, nil, nil), device(device2, nil, nil)),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, true),
				deviceAllocationResult(req0, driverA, pool1, device2, true),
			)},
		},
		"all-devices-some-allocating-admin-access": {
			features: Features{
				AdminAccess: true,
			},
			claimsToAllocate: func() []wrapResourceClaim {
				c := claimWithRequests(claim0, nil, request(req0, classA, 1), request(req1, classA, 1))
				c.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				// Second request for `All` cannot be fulfilled because
				// the first request claims a device.
				c.Spec.Devices.Requests[1].Exactly.AdminAccess = ptr.To(true)
				c.Spec.Devices.Requests[1].Exactly.AllocationMode = resourceapi.DeviceAllocationModeAll
				return []wrapResourceClaim{c}
			}(),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1, nil, nil), device(device2, nil, nil)),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		"count-devices-some-allocated-admin-access": {
			features: Features{
				AdminAccess: true,
			},
			claimsToAllocate: func() []wrapResourceClaim {
				c := claimWithRequest(claim0, req0, classA)
				c.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				c.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationModeExactCount
				c.Spec.Devices.Requests[0].Exactly.Count = 2
				return []wrapResourceClaim{c}
			}(),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
			},
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1, nil, nil), device(device2, nil, nil)),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, true),
				deviceAllocationResult(req0, driverA, pool1, device2, true),
			)},
		},
		"separate-claims-share-device-admin-access": {
			features: Features{
				AdminAccess: true,
			},
			claimsToAllocate: func() []wrapResourceClaim {
				c1 := claimWithRequest(claim0, req0, classA)
				c1.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)

				c2 := claimWithRequest(claim1, req0, classA)
				c2.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				return []wrapResourceClaim{c1, c2}
			}(),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1, nil, nil), device(device2, nil, nil)),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, true),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, true),
				),
			},
		},
		"admin-access-claims-device-allocating-for-non-admin-access": {
			features: Features{
				AdminAccess: true,
			},
			claimsToAllocate: func() []wrapResourceClaim {
				admin := claimWithRequest(claim1, req0, classA)
				admin.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				return []wrapResourceClaim{claimWithRequest(claim0, req0, classA), admin}
			}(),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1, nil, nil), device(device2, nil, nil)),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, true),
				),
			},
		},
		"all-devices-slice-without-devices-prioritized-list": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claim(claim0).withRequests(
						requestWithPrioritizedList(req0,
							subRequest(subReq0, classA, 1),
							subRequest(subReq1, classB, 1),
						),
					)
					claim.Spec.Devices.Requests[0].FirstAvailable[0].AllocationMode = resourceapi.DeviceAllocationModeAll
					claim.Spec.Devices.Requests[0].FirstAvailable[0].Count = 0
					return claim
				}(),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(
				sliceWithNoDevices(slice1, node1, pool1, driverA),
				sliceWithOneDevice(slice2, node1, pool2, driverB),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverB, pool2, device1, false),
			)},
		},
		"all-devices-no-slices-prioritized-list": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claim(claim0).withRequests(
						requestWithPrioritizedList(req0,
							subRequest(subReq0, classA, 1),
							subRequest(subReq1, classB, 1),
						),
					)
					claim.Spec.Devices.Requests[0].FirstAvailable[0].AllocationMode = resourceapi.DeviceAllocationModeAll
					claim.Spec.Devices.Requests[0].FirstAvailable[0].Count = 0
					return claim
				}(),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice2, node1, pool2, driverB),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverB, pool2, device1, false),
			)},
		},
		"all-devices-some-allocated-prioritized-list": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claim(claim0).withRequests(
						requestWithPrioritizedList(req0,
							subRequest(subReq0, classA, 1),
							subRequest(subReq1, classB, 1),
						),
					)
					claim.Spec.Devices.Requests[0].FirstAvailable[0].AllocationMode = resourceapi.DeviceAllocationModeAll
					claim.Spec.Devices.Requests[0].FirstAvailable[0].Count = 0
					return claim
				}(),
			),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
			},
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1, nil, nil), device(device2, nil, nil)),
				sliceWithOneDevice(slice2, node1, pool2, driverB),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverB, pool2, device1, false),
			)},
		},
		"network-attached-device": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, nodeLabelSelector(regionKey, region1), pool1, driverA)),
			node:             node(node1, region1),

			expectResults: []any{allocationResult(
				nodeLabelSelector(regionKey, region1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"unsuccessful-allocation-network-attached-device": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, nodeLabelSelector(regionKey, region1), pool1, driverA)),
			// Wrong region, no devices available.
			node: node(node2, region2),

			expectResults: nil,
		},
		"many-network-attached-devices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 4))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, nodeLabelSelector(regionKey, region1), pool1, driverA),
				sliceWithOneDevice(slice1, nodeSelectionAll, pool2, driverA),
				sliceWithOneDevice(slice1, nodeLabelSelector(planetKey, planetValueEarth), pool3, driverA),
				sliceWithOneDevice(slice1, localNodeSelector(node1), pool4, driverA),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				// A union of the individual selectors.
				&v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{Key: regionKey, Operator: v1.NodeSelectorOpIn, Values: []string{region1}},
							{Key: planetKey, Operator: v1.NodeSelectorOpIn, Values: []string{planetValueEarth}},
						},
						MatchFields: []v1.NodeSelectorRequirement{
							{Key: fieldNameKey, Operator: v1.NodeSelectorOpIn, Values: []string{node1}},
						},
					}},
				},
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool2, device1, false),
				deviceAllocationResult(req0, driverA, pool3, device1, false),
				deviceAllocationResult(req0, driverA, pool4, device1, false),
			)},
		},
		"local-and-network-attached-devices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 2))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, nodeLabelSelector(regionKey, region1), pool1, driverA),
				sliceWithOneDevice(slice1, node1, pool2, driverA),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				// Once there is any node-local device, the selector is for that node.
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool2, device1, false),
			)},
		},
		"several-different-drivers": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA), claimWithRequest(claim0, req0, classB)),
			classes:          objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil),
					device(device2, nil, nil),
				),
				sliceWithOneDevice(slice1, node1, pool1, driverB),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverA, pool1, device1, false)),
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverB, pool1, device1, false)),
			},
		},
		"already-allocated-devices": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
				MakeDeviceID(driverA, pool1, device2),
			},
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: nil,
		},
		"admin-access-disabled": {
			features: Features{
				AdminAccess: false,
			},
			claimsToAllocate: func() []wrapResourceClaim {
				c := claimWithRequest(claim0, req0, classA)
				c.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				return []wrapResourceClaim{c}
			}(),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: admin access is requested, but the feature is disabled")),
		},
		"admin-access-enabled": {
			features: Features{
				AdminAccess: true,
			},
			claimsToAllocate: func() []wrapResourceClaim {
				c := claimWithRequest(claim0, req0, classA)
				c.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				return []wrapResourceClaim{c}
			}(),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
				MakeDeviceID(driverA, pool1, device2),
			},
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: []any{
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverA, pool1, device1, true)),
			},
		},
		"with-constraint": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{
					{MatchAttribute: &intAttribute},
					{MatchAttribute: &versionAttribute},
					{MatchAttribute: &stringAttribute},
					{MatchAttribute: &boolAttribute},
				},
				request(req0, classA, 1),
				request(req1, classA, 1),
			),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion":   {VersionValue: ptr.To("1.0.0")},
					"numa":            {IntValue: ptr.To(int64(1))},
					"stringAttribute": {StringValue: ptr.To("stringAttributeValue")},
					"boolAttribute":   {BoolValue: ptr.To(true)},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion":   {VersionValue: ptr.To("1.0.0")},
					"numa":            {IntValue: ptr.To(int64(1))},
					"stringAttribute": {StringValue: ptr.To("stringAttributeValue")},
					"boolAttribute":   {BoolValue: ptr.To(true)},
				}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1, driverA, pool1, device2, false),
			)},
		},
		"with-constraint-non-existent-attribute": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA, resourceapi.DeviceConstraint{
				MatchAttribute: &nonExistentAttribute,
			})),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: nil,
		},
		"with-constraint-not-matching-int-attribute": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &intAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"numa": {IntValue: ptr.To(int64(1))},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"numa": {IntValue: ptr.To(int64(2))},
				}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"with-constraint-not-matching-int-attribute-all-devices": {
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claimWithRequests(
						claim0,
						[]resourceapi.DeviceConstraint{{MatchAttribute: &intAttribute}},
						request(req0, classA, 0),
					)
					claim.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationModeAll
					return claim
				}(),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"numa": {IntValue: ptr.To(int64(1))},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"numa": {IntValue: ptr.To(int64(2))},
				}),
			)),
			node: node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: cannot add device driver-a/pool-1/device-2 because a claim constraint would not be satisfied")),
		},
		"with-constraint-not-matching-string-attribute": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"stringAttribute": {StringValue: ptr.To("stringAttributeValue")},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"stringAttribute": {StringValue: ptr.To("stringAttributeValue2")},
				}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"with-constraint-not-matching-bool-attribute": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &boolAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"boolAttribute": {BoolValue: ptr.To(true)},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"boolAttribute": {BoolValue: ptr.To(false)},
				}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"with-constraint-not-matching-version-attribute": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &versionAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion": {VersionValue: ptr.To("1.0.0")},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion": {VersionValue: ptr.To("2.0.0")},
				}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"with-constraint-for-request": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{
					{
						Requests:       []string{req0},
						MatchAttribute: &versionAttribute,
					},
				},
				request(req0, classA, 1),
				request(req1, classA, 1),
			)),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion": {VersionValue: ptr.To("1.0.0")},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion": {VersionValue: ptr.To("2.0.0")},
				}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1, driverA, pool1, device2, false),
			)},
		},
		"with-constraint-for-request-retry": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{
					{
						Requests:       []string{req0},
						MatchAttribute: &versionAttribute,
					},
					{
						MatchAttribute: &stringAttribute,
					},
				},
				request(req0, classA, 1),
				request(req1, classA, 1),
			)),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				// This device does not satisfy the second
				// match attribute, so the allocator must
				// backtrack.
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion":   {VersionValue: ptr.To("1.0.0")},
					"stringAttribute": {StringValue: ptr.To("a")},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion":   {VersionValue: ptr.To("2.0.0")},
					"stringAttribute": {StringValue: ptr.To("b")},
				}),
				device(device3, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion":   {VersionValue: ptr.To("3.0.0")},
					"stringAttribute": {StringValue: ptr.To("b")},
				}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req1, driverA, pool1, device3, false),
			)},
		},
		"with-class-device-config": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(classWithConfig(classA, driverA, "classAttribute")),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: []any{
				allocationResultWithConfigs(
					localNodeSelector(node1),
					objects(deviceAllocationResult(req0, driverA, pool1, device1, false)),
					[]resourceapi.DeviceAllocationConfiguration{
						{
							Source:              resourceapi.AllocationConfigSourceClass,
							Requests:            []string{req0},
							DeviceConfiguration: deviceConfiguration(driverA, "classAttribute"),
						},
					},
				),
			},
		},
		"with-class-device-config-with-multiple-request-and-configs": {
			claimsToAllocate: objects(claimWithRequests(claim0, []resourceapi.DeviceConstraint{}, request(req0, classA, 1), request(req1, classA, 1), request(req2, classB, 1))),
			classes: objects(
				classWithConfig(classA, driverA, "classAttribute-A"),
				classWithConfig(classB, driverB, "classAttribute-B"),
			),
			slices: unwrap(
				sliceWithMultipleDevices(slice1, node1, pool1, driverA, 2),
				sliceWithOneDevice(slice2, node1, pool1, driverB),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResultWithConfigs(
					localNodeSelector(node1),
					objects(
						deviceAllocationResult(req0, driverA, pool1, device0, false),
						deviceAllocationResult(req1, driverA, pool1, device1, false),
						deviceAllocationResult(req2, driverB, pool1, device1, false),
					),
					[]resourceapi.DeviceAllocationConfiguration{
						{
							Source:              resourceapi.AllocationConfigSourceClass,
							Requests:            []string{req0, req1},
							DeviceConfiguration: deviceConfiguration(driverA, "classAttribute-A"),
						},
						{
							Source:              resourceapi.AllocationConfigSourceClass,
							Requests:            []string{req2},
							DeviceConfiguration: deviceConfiguration(driverB, "classAttribute-B"),
						},
					},
				),
			},
		},
		"claim-with-device-config": {
			claimsToAllocate: objects(claimWithDeviceConfig(claim0, req0, classA, driverA, "deviceAttribute")),
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: []any{
				allocationResultWithConfig(
					localNodeSelector(node1),
					driverA,
					resourceapi.AllocationConfigSourceClaim,
					"deviceAttribute",
					deviceAllocationResult(req0, driverA, pool1, device1, false),
				),
			},
		},
		"unknown-selector": {
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claimWithRequest(claim0, req0, classA)
					claim.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{
						{ /* empty = unknown future selector */ },
					}
					return claim
				}(),
			),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("CEL expression empty (unsupported selector type?)")),
		},
		"unknown-allocation-mode": {
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claimWithRequest(claim0, req0, classA)
					claim.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationMode("future-mode")
					return claim
				}(),
			),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("unsupported count mode future-mode")),
		},
		"unknown-constraint": {
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claimWithRequest(claim0, req0, classA)
					claim.Spec.Devices.Constraints = []resourceapi.DeviceConstraint{
						{ /* empty = unknown */ },
					}
					return claim
				}(),
			),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("empty constraint (unsupported constraint type?)")),
		},
		"invalid-CEL-one-device": {
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claimWithRequest(claim0, req0, classA)
					claim.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{
						{CEL: &resourceapi.CELDeviceSelector{Expression: "noSuchVar"}},
					}
					return claim
				}(),
			),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("undeclared reference")),
		},
		"invalid-CEL-one-device-class": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes: objects(
				func() *resourceapi.DeviceClass {
					c := class(classA, driverA)
					c.Spec.Selectors[0].CEL.Expression = "noSuchVar"
					return c
				}(),
			),
			slices: unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:   node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("undeclared reference")),
		},
		"invalid-CEL-all-devices": {
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claimWithRequest(claim0, req0, classA)
					claim.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{
						{CEL: &resourceapi.CELDeviceSelector{Expression: "noSuchVar"}},
					}
					claim.Spec.Devices.Requests[0].Exactly.AllocationMode = resourceapi.DeviceAllocationModeAll
					return claim
				}(),
			),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("undeclared reference")),
		},
		"too-many-devices-single-request": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 500))),
			classes:          objects(class(classA, driverA)),

			expectError: gomega.MatchError(gomega.ContainSubstring("exceeds the claim limit")),
		},
		"many-devices-okay": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, resourceapi.AllocationResultsMaxSize))),
			classes:          objects(class(classA, driverA)),
		},
		"too-many-devices-total": {
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, resourceapi.AllocationResultsMaxSize),
					request(req1, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithMultipleDevices(slice1, node1, pool1, driverA, resourceapi.AllocationResultsMaxSize+1)),
			node:    node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("exceeds the claim limit")),
		},
		"all-devices-invalid-CEL": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 500))),
			classes:          objects(class(classA, driverA)),

			expectError: gomega.MatchError(gomega.ContainSubstring("exceeds the claim limit")),
		},
		"prioritized-list-first-unavailable": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classB, 1),
						subRequest(subReq1, classA, 1),
					))),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device1, false),
			)},
		},
		"prioritized-list-non-available": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classB, 2),
						subRequest(subReq1, classA, 2),
					))),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(
				sliceWithOneDevice(slice1, node1, pool1, driverA),
				sliceWithOneDevice(slice2, node1, pool2, driverB),
			),
			node: node(node1, region1),

			expectResults: nil,
		},
		"prioritized-list-device-config": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 1),
						subRequest(subReq1, classB, 2),
					),
				).withConfigs(
					objects(
						deviceClaimConfig([]string{req0SubReq0}, deviceConfiguration(driverA, "foo")),
						deviceClaimConfig([]string{req0SubReq1}, deviceConfiguration(driverB, "bar")),
					),
				),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverB,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResultWithConfigs(
				localNodeSelector(node1),
				objects(
					deviceAllocationResult(req0SubReq1, driverB, pool1, device1, false),
					deviceAllocationResult(req0SubReq1, driverB, pool1, device2, false),
				),
				[]resourceapi.DeviceAllocationConfiguration{
					{
						Source: resourceapi.AllocationConfigSourceClaim,
						Requests: []string{
							req0SubReq1,
						},
						DeviceConfiguration: deviceConfiguration(driverB, "bar"),
					},
				},
			)},
		},
		"prioritized-list-class-config": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 2),
						subRequest(subReq1, classB, 2),
					))),
			classes: objects(
				classWithConfig(classA, driverA, "foo"),
				classWithConfig(classB, driverB, "bar"),
			),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverB,
					device(device1, nil, nil),
					device(device2, nil, nil),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device3, nil, nil),
				),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResultWithConfigs(
				localNodeSelector(node1),
				objects(
					deviceAllocationResult(req0SubReq1, driverB, pool1, device1, false),
					deviceAllocationResult(req0SubReq1, driverB, pool1, device2, false),
				),
				[]resourceapi.DeviceAllocationConfiguration{
					{
						Source:              resourceapi.AllocationConfigSourceClass,
						Requests:            []string{req0SubReq1},
						DeviceConfiguration: deviceConfiguration(driverB, "bar"),
					},
				},
			)},
		},
		"prioritized-list-subrequests-with-expressions": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1).withSelectors(
						resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("1Gi")) >= 0`, driverA),
							},
						},
					),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("4Gi")) >= 0`, driverA),
							}}),
						subRequest(subReq1, classA, 2, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
							}}),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("2Gi"),
				}, nil),
				device(device2, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("2Gi"),
				}, nil),
				device(device3, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("1Gi"),
				}, nil),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
				deviceAllocationResult(req1SubReq1, driverA, pool1, device1, false),
				deviceAllocationResult(req1SubReq1, driverA, pool1, device2, false),
			)},
		},
		"prioritized-list-subrequests-with-constraints-ref-parent-request": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(
					resourceapi.DeviceConstraint{
						Requests:       []string{req0, req1},
						MatchAttribute: &versionAttribute,
					},
				).withRequests(
					deviceRequest(req0, classA, 1).withSelectors(resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("8Gi")) >= 0`, driverA),
						}},
					),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
							}},
						),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("1Gi")) >= 0`, driverA),
							}},
						),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1,
						map[resourceapi.QualifiedName]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						},
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"driverVersion": {VersionValue: ptr.To("1.0.0")},
						},
					),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device2,
						map[resourceapi.QualifiedName]resource.Quantity{
							"memory": resource.MustParse("2Gi"),
						},
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"driverVersion": {VersionValue: ptr.To("2.0.0")},
						},
					),
					device(device3,
						map[resourceapi.QualifiedName]resource.Quantity{
							"memory": resource.MustParse("1Gi"),
						},
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"driverVersion": {VersionValue: ptr.To("1.0.0")},
						},
					),
				),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1SubReq1, driverA, pool2, device3, false),
			)},
		},
		"prioritized-list-subrequests-with-constraints-ref-sub-request": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(
					resourceapi.DeviceConstraint{
						Requests:       []string{req0, req1SubReq0},
						MatchAttribute: &versionAttribute,
					},
				).withRequests(
					deviceRequest(req0, classA, 1).withSelectors(resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("8Gi")) >= 0`, driverA),
						}},
					),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
							}},
						),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("1Gi")) >= 0`, driverA),
							}},
						),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1,
						map[resourceapi.QualifiedName]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						},
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"driverVersion": {VersionValue: ptr.To("1.0.0")},
						},
					),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device2,
						map[resourceapi.QualifiedName]resource.Quantity{
							"memory": resource.MustParse("2Gi"),
						},
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"driverVersion": {VersionValue: ptr.To("2.0.0")},
						},
					),
				),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1SubReq1, driverA, pool2, device2, false),
			)},
		},
		"prioritized-list-subrequests-with-allocation-mode-all": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claim(claim0).withRequests(
						requestWithPrioritizedList(req0,
							subRequest(subReq0, classA, 1),
							subRequest(subReq1, classA, 1),
						),
					)
					claim.Spec.Devices.Requests[0].FirstAvailable[0].AllocationMode = resourceapi.DeviceAllocationModeAll
					claim.Spec.Devices.Requests[0].FirstAvailable[0].Count = 0
					return claim
				}(),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil),
					device(device2, nil, nil),
				),
			),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
			},
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device2, false),
			)},
		},
		"prioritized-list-allocation-mode-all-multiple-requests": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, 0).withAllocationMode(resourceapi.DeviceAllocationModeAll),
						subRequest(subReq1, classA, 1),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil),
					device(device2, nil, nil),
				),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1SubReq1, driverA, pool1, device2, false),
			)},
		},
		"prioritized-list-disabled": {
			features: Features{
				PrioritizedList: false,
			},
			claimsToAllocate: objects(
				func() wrapResourceClaim {
					claim := claim(claim0).withRequests(
						requestWithPrioritizedList(req0,
							subRequest(subReq0, classA, 1),
						),
					)
					return claim
				}(),
			),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: has subrequests, but the DRAPrioritizedList feature is disabled")),
		},
		"prioritized-list-multi-request": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req1, classA, 1).withSelectors(
						resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("8Gi")) >= 0`, driverA),
							}},
					),
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("8Gi")) >= 0`, driverA),
							}},
						),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("4Gi")) >= 0`, driverA),
							}},
						),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1,
						map[resourceapi.QualifiedName]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						},
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
					),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device2,
						map[resourceapi.QualifiedName]resource.Quantity{
							"memory": resource.MustParse("4Gi"),
						},
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req1, driverA, pool1, device1, false),
				deviceAllocationResult(req0SubReq1, driverA, pool2, device2, false),
			)},
		},
		"prioritized-list-with-backtracking": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("8Gi")) >= 0`, driverA),
							}},
						),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("4Gi")) >= 0`, driverA),
							}},
						),
					),
					deviceRequest(req1, classA, 1).withSelectors(
						resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("8Gi")) >= 0`, driverA),
							}},
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1,
						map[resourceapi.QualifiedName]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						},
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
					),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device2,
						map[resourceapi.QualifiedName]resource.Quantity{
							"memory": resource.MustParse("4Gi"),
						},
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool2, device2, false),
				deviceAllocationResult(req1, driverA, pool1, device1, false),
			)},
		},
		"prioritized-list-too-many-in-first-subrequest": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(requestWithPrioritizedList(req0,
					subRequest(subReq0, classB, 500),
					subRequest(subReq1, classA, 1),
				))),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices:  unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device1, false),
			)},
		},
		"partitionable-devices-single-device": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"partitionable-devices-prioritized-list": {
			features: Features{
				PrioritizedList:      true,
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("6Gi")) >= 0`, driverA),
							}},
						),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("4Gi")) >= 0`, driverA),
							}},
						),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("6Gi"),
							},
						),
					),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1SubReq1, driverA, pool1, device3, false),
			)},
		},
		"partitionable-devices-multiple-devices": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
					request(req1, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("6Gi"),
							},
						),
					),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1, driverA, pool1, device3, false),
			)},
		},
		"partitionable-devices-multiple-capacity-pools": {
			features: Features{
				PrioritizedList:      true,
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("6Gi")) >= 0`, driverA),
							}},
						),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("4Gi")) >= 0`, driverA),
							}},
						),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"cpus": resource.MustParse("4"),
							},
						),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("6Gi"),
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"cpus": resource.MustParse("6"),
							},
						),
					),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"cpus": resource.MustParse("4"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("18Gi"),
						},
					),
					counterSet(counterSet2,
						map[string]resource.Quantity{
							"cpus": resource.MustParse("8"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1SubReq1, driverA, pool1, device3, false),
			)},
		},
		"partitionable-devices-multiple-counters": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
					request(req1, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
								"cpus":   resource.MustParse("6"),
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("4"),
								"memory": resource.MustParse("2Gi"),
							},
						),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("6Gi"),
								"cpus":   resource.MustParse("4"),
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("6"),
								"memory": resource.MustParse("6Gi"),
							},
						),
					),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
								"cpus":   resource.MustParse("4"),
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("4"),
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"cpus":   resource.MustParse("8"),
							"memory": resource.MustParse("18Gi"),
						},
					),
					counterSet(counterSet2,
						map[string]resource.Quantity{
							"cpus":   resource.MustParse("12"),
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req1, driverA, pool1, device3, false),
			)},
		},
		"partitionable-devices-multiple-pools": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
					request(req1, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("6"),
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("4"),
								"memory": resource.MustParse("6Gi"),
							},
						),
					),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("4"),
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"cpus":   resource.MustParse("8"),
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool2, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("6"),
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("1"),
								"memory": resource.MustParse("7Gi"),
							},
						),
					),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("1"),
								"memory": resource.MustParse("7Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice4, node1, resourcePool(pool2, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"cpus":   resource.MustParse("8"),
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
			),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device2),
				MakeDeviceID(driverA, pool1, device3),
				MakeDeviceID(driverA, pool2, device1),
			},
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool2, device2, false),
				deviceAllocationResult(req1, driverA, pool2, device3, false),
			)},
		},
		"partitionable-devices-unused-counters-in-counter-set": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("5Gi"),
							},
						),
					),
					device(device2, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device3, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
							"unused": resource.MustParse("1Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"partitionable-devices-no-capacity-available": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("16Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, pool1, driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
			),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device2),
			},
			node:          node(node1, region1),
			expectResults: nil,
		},
		"partitionable-devices-overallocated-capacity-pool": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("20Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, pool1, driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
			),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device2),
			},
			node:          node(node1, region1),
			expectResults: nil,
		},
		"partitionable-devices-disabled-device-consumes-counters": {
			features: Features{
				PartitionableDevices: false,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, pool1, driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		"partitionable-devices-disabled-other-device-do-not-consume-counters": {
			features: Features{
				PartitionableDevices: false,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2, nil, nil),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"partitionable-devices-disabled-per-device-node-selection": {
			features: Features{
				PartitionableDevices: false,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, nodeSelectionPerDevice, pool1, driverA,
					device(device2, nil, nil).withNodeSelection(node1),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		"partitionable-devices-per-device-node-selection-nodename": {
			features: Features{
				PartitionableDevices: true,
				PrioritizedList:      true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("6Gi")) >= 0`, driverA),
							}},
						),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("4Gi")) >= 0`, driverA),
							}},
						),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, nodeSelectionPerDevice, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withNodeSelection(node1),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("6Gi"),
							},
						),
					).withNodeSelection(node2),
				),
				sliceWithCounterSets(slice2, nodeSelectionPerDevice, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device1, false),
			)},
		},
		"partitionable-devices-per-device-node-selection-node-selector": {
			features: Features{
				PartitionableDevices: true,
				PrioritizedList:      true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, nodeSelectionPerDevice, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withNodeSelection(nodeLabelSelector(regionKey, region1)),
				),
				sliceWithCounterSets(slice2, nodeSelectionPerDevice, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				&v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{Key: regionKey, Operator: v1.NodeSelectorOpIn, Values: []string{region1}},
						},
					}},
				},
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"partitionable-devices-per-device-node-selection-node-selector-multiple-devices": {
			features: Features{
				PartitionableDevices: true,
				PrioritizedList:      true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 3),
					request(req1, classB, 3),
				),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, nodeSelectionPerDevice, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withNodeSelection(nodeLabelSelector(regionKey, region1)),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withNodeSelection(node1),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withNodeSelection(nodeSelectionAll),
				),
				sliceWithCounterSets(slice2, nodeSelectionPerDevice, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("18Gi"),
						},
					),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool2, 2), driverB,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice4, node1, resourcePool(pool2, 2), driverB,
					counterSet(counterSet2,
						map[string]resource.Quantity{
							"memory": resource.MustParse("12Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				&v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{{
						MatchFields: []v1.NodeSelectorRequirement{
							{Key: fieldNameKey, Operator: v1.NodeSelectorOpIn, Values: []string{node1}},
						},
					}},
				},
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
				deviceAllocationResult(req1, driverB, pool2, device1, false),
				deviceAllocationResult(req1, driverB, pool2, device2, false),
				deviceAllocationResult(req1, driverB, pool2, device3, false),
			)},
		},
		"tainted-two-devices": {
			features: Features{
				DeviceTaints: true,
			},
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule),
				device(device2, nil, nil).withTaints(taintNoExecute),
			)),
			node: node(node1, region1),
		},
		"tainted-one-device-two-taints": {
			features: Features{
				DeviceTaints: true,
			},
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule, taintNoExecute),
			)),
			node: node(node1, region1),
		},
		"tainted-two-devices-tolerated": {
			features: Features{
				DeviceTaints: true,
			},
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA).withTolerations(tolerationNoExecute)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule),
				device(device2, nil, nil).withTaints(taintNoExecute),
			)),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false, tolerationNoExecute), // Only second device's taints are tolerated.
			)},
		},
		"tainted-two-devices-prioritized-list-tolerated": {
			features: Features{
				DeviceTaints:    true,
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(requestWithPrioritizedList(req0,
					subRequest(subReq0, classA, 1),
					subRequest(subReq1, classA, 1, tolerationNoExecute),
				))),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule),
				device(device2, nil, nil).withTaints(taintNoExecute),
			)),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device2, false, tolerationNoExecute), // Only second device's taints are tolerated.
			)},
		},
		"tainted-no-effect": {
			features: Features{
				DeviceTaints: true,
			},
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNone),
			)),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"tainted-one-device-two-taints-both-tolerated": {
			features: Features{
				DeviceTaints: true,
			},
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA).withTolerations(tolerationNoSchedule, tolerationNoExecute)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule, taintNoExecute),
			)),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false, tolerationNoSchedule, tolerationNoExecute),
			)},
		},
		"tainted-disabled": {
			features: Features{
				DeviceTaints: false,
			},
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule, taintNoExecute),
			)),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"tainted-prioritized-list": {
			features: Features{
				DeviceTaints:    true,
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classB, 1),
						subRequest(subReq1, classA, 1),
					))),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule),
			)),
			node: node(node1, region1),
		},
		"tainted-prioritized-list-disabled": {
			features: Features{
				DeviceTaints:    false,
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classB, 1),
						subRequest(subReq1, classA, 1),
					))),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device1, false),
			)},
		},
		"tainted-admin-access": {
			features: Features{
				DeviceTaints: true,
				AdminAccess:  true,
			},
			claimsToAllocate: func() []wrapResourceClaim {
				c := claimWithRequest(claim0, req0, classA)
				c.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				return []wrapResourceClaim{c}
			}(),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
				MakeDeviceID(driverA, pool1, device2),
			},
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule),
			)),
			node: node(node1, region1),
		},
		"tainted-admin-access-disabled": {
			features: Features{
				DeviceTaints: false,
				AdminAccess:  true,
			},
			claimsToAllocate: func() []wrapResourceClaim {
				c := claimWithRequest(claim0, req0, classA)
				c.Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
				return []wrapResourceClaim{c}
			}(),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
				MakeDeviceID(driverA, pool1, device2),
			},
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, true),
			)},
		},
		"tainted-all-devices-single": {
			features: Features{
				DeviceTaints: true,
			},
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					DeviceClassName: classA,
				},
			})),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule),
			)),
			node: node(node1, region1),
		},
		"tainted-all-devices-single-disabled": {
			features: Features{
				DeviceTaints: false,
			},
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					DeviceClassName: classA,
				},
			})),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withTaints(taintNoSchedule),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"prioritized-list-allocation-mode-all": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 0).withAllocationMode(resourceapi.DeviceAllocationModeAll),
						subRequest(subReq1, classA, 1),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil),
				device(device2, nil, nil),
			)),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device2),
			},
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device1, false),
			)},
		},
		"max-number-devices": {
			claimsToAllocate: objects(
				claimWithRequests(
					claim0, nil, request(req0, classA, resourceapi.AllocationResultsMaxSize),
				),
			),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithMultipleDevices(slice1, node1, pool1, driverA, resourceapi.AllocationResultsMaxSize)),
			node:    node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				multipleDeviceAllocationResults(req0, driverA, pool1, resourceapi.AllocationResultsMaxSize, 0)...,
			)},
		},
		"partitionable-devices-with-attribute-selector": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 2),
					request(req1, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.attributes["%s"].special`, driverA),
						},
					}),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil,
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"special": {
								BoolValue: ptr.To(true),
							},
						},
					).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpu1": resource.MustParse("1"),
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"mem": resource.MustParse("10Gi"),
							},
						),
					),
					device(device2, nil,
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"special": {
								BoolValue: ptr.To(false),
							},
						},
					).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpu2": resource.MustParse("1"),
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"mem": resource.MustParse("10Gi"),
							},
						),
					),
					device(device3, nil,
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"special": {
								BoolValue: ptr.To(false),
							},
						},
					).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpu3": resource.MustParse("1"),
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"mem": resource.MustParse("10Gi"),
							},
						),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"cpu1": resource.MustParse("1"),
							"cpu2": resource.MustParse("1"),
							"cpu3": resource.MustParse("1"),
						},
					),
					counterSet(counterSet2,
						map[string]resource.Quantity{
							"mem": resource.MustParse("30Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
				deviceAllocationResult(req1, driverA, pool1, device1, false),
			)},
		},
		"prioritized-list-max-allocation-limit-request": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, resourceapi.AllocationResultsMaxSize),
						subRequest(subReq1, classA, resourceapi.AllocationResultsMaxSize/2),
					),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, resourceapi.AllocationResultsMaxSize),
						subRequest(subReq1, classA, resourceapi.AllocationResultsMaxSize/2),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithMultipleDevices(slice1, node1, pool1, driverA, resourceapi.AllocationResultsMaxSize*2)),
			node:    node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				slices.Concat(
					multipleDeviceAllocationResults(req0SubReq1, driverA, pool1, resourceapi.AllocationResultsMaxSize/2, 0),
					multipleDeviceAllocationResults(req1SubReq1, driverA, pool1, resourceapi.AllocationResultsMaxSize/2, resourceapi.AllocationResultsMaxSize/2),
				)...,
			)},
			expectNumAllocateOneInvocations: 75,
		},
		"prioritized-list-max-allocation-allocation-mode-all": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 0).withAllocationMode(resourceapi.DeviceAllocationModeAll),
						subRequest(subReq1, classA, 2),
					),
					deviceRequest(req1, classB, 2),
				),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(
				sliceWithMultipleDevices(slice1, node1, pool1, driverA, resourceapi.AllocationResultsMaxSize-1),
				sliceWithMultipleDevices(slice2, node1, pool2, driverB, 2),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, "device-0", false),
				deviceAllocationResult(req0SubReq1, driverA, pool1, "device-1", false),
				deviceAllocationResult(req1, driverB, pool2, "device-0", false),
				deviceAllocationResult(req1, driverB, pool2, "device-1", false),
			)},
			expectNumAllocateOneInvocations: 42,
		},
		"device-binding-conditions": {
			features: Features{
				DeviceBindingAndStatus: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1))),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}))),
			node: node(node1, region1),

			expectResults: []any{
				resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device1, []string{"IsPrepare"}, []string{"BindingFailed"}),
						},
					},
					NodeSelector: localNodeSelector(node1),
				},
			},
		},
		"binding-conditions-multiple-devices": {
			features: Features{
				DeviceBindingAndStatus: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 2),
					request(req1, classA, 1))),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
				device(device2, nil, nil).withBindingConditions([]string{"IsPrepare2"}, []string{"BindingFailed2"}),
				device(device3, nil, nil).withBindingConditions([]string{"IsPrepare3"}, []string{"BindingFailed3"}),
			)),
			node: node(node1, region1),
			expectResults: []any{
				resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device1, []string{"IsPrepare"}, []string{"BindingFailed"}),
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device2, []string{"IsPrepare2"}, []string{"BindingFailed2"}),
							deviceRequestAllocationResultWithBindingConditions(req1, driverA, pool1, device3, []string{"IsPrepare3"}, []string{"BindingFailed3"}),
						},
					},
					NodeSelector: localNodeSelector(node1),
				},
			},
		},
		"binding-conditions-with-and-without": {
			features: Features{
				DeviceBindingAndStatus: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
					request(req1, classA, 1))),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil),
				device(device2, nil, nil).withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
			)),
			node: node(node1, region1),
			expectResults: []any{
				resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							deviceAllocationResult(req0, driverA, pool1, device1, false),
							deviceRequestAllocationResultWithBindingConditions(req1, driverA, pool1, device2, []string{"IsPrepare"}, []string{"BindingFailed"}),
						},
					},
					NodeSelector: localNodeSelector(node1),
				},
			},
		},
		"binding-conditions-without-feature-gate": {
			features: Features{
				DeviceBindingAndStatus: false,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1))),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1, nil, nil).withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}))),
			node:          node(node1, region1),
			expectResults: nil,
		},
		"node-restriction": {
			features: Features{
				DeviceBindingAndStatus: true,
			},
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, nodeLabelSelector(planetKey, planetValueEarth), pool1, driverA,
				device(device1, nil, nil).withBindsToNode(true))),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"partitionable-devices-with-binding-conditions": {
			features: Features{
				PartitionableDevices:   true,
				DeviceBindingAndStatus: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device1, []string{"IsPrepare"}, []string{"BindingFailed"}),
						},
					},
					NodeSelector: localNodeSelector(node1),
				},
			},
		},
		"partitionable-devices-with-binding-conditions-multiple": {
			features: Features{
				PartitionableDevices:   true,
				DeviceBindingAndStatus: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 2),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
					device(device2, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare2"}, []string{"BindingFailed2"}),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device1, []string{"IsPrepare"}, []string{"BindingFailed"}),
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device2, []string{"IsPrepare2"}, []string{"BindingFailed2"}),
						},
					},
					NodeSelector: localNodeSelector(node1),
				},
			},
		},
		"partitionable-devices-with-binding-conditions-some-devices-no-conditions": {
			features: Features{
				PartitionableDevices:   true,
				DeviceBindingAndStatus: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 3),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("2Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
					device(device2, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("2Gi"),
							}),
						),
					device(device3, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						).
						withBindingConditions([]string{"IsReady"}, []string{"BindingTimeout"}),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device1, []string{"IsPrepare"}, []string{"BindingFailed"}),
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device2, nil, nil),
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device3, []string{"IsReady"}, []string{"BindingTimeout"}),
						},
					},
					NodeSelector: localNodeSelector(node1),
				},
			},
		},
		"partitionable-devices-with-binding-conditions-multi-slices": {
			features: Features{
				PartitionableDevices:   true,
				DeviceBindingAndStatus: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 4), driverA,
					device(device1, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 4), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool1, 4), driverA,
					device(device2, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet2, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						),
				),
				sliceWithCounterSets(slice4, node1, resourcePool(pool1, 4), driverA,
					counterSet(counterSet2, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device2, nil, nil),
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device1, []string{"IsPrepare"}, []string{"BindingFailed"}),
						},
					},
					NodeSelector: localNodeSelector(node1),
				},
			},
		},
		"partitionable-devices-with-binding-conditions-and-taints": {
			features: Features{
				PartitionableDevices:   true,
				DeviceBindingAndStatus: true,
				DeviceTaints:           true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}).
						withTaints(resourceapi.DeviceTaint{
							Key:    "key1",
							Value:  "value1",
							Effect: resourceapi.DeviceTaintEffectNoSchedule,
						}),
					device(device2, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						),
				),
				sliceWithCounterSets(slice2, node1, pool1, driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		"partitionable-devices-with-binding-conditions-and-taints-tolerated": {
			features: Features{
				PartitionableDevices:   true,
				DeviceBindingAndStatus: true,
				DeviceTaints:           true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 2)).withTolerations(tolerationNoScheduleKey1),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}).
						withTaints(resourceapi.DeviceTaint{
							Key:    "key1",
							Value:  "value1",
							Effect: resourceapi.DeviceTaintEffectNoSchedule,
						}),
					device(device2, fromCounters, nil).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				resourceapi.AllocationResult{
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device1, []string{"IsPrepare"}, []string{"BindingFailed"}, tolerationNoScheduleKey1),
							deviceRequestAllocationResultWithBindingConditions(req0, driverA, pool1, device2, nil, nil, tolerationNoScheduleKey1),
						},
					},
					NodeSelector: localNodeSelector(node1),
				},
			},
		},
		"consumable-capacity-disabled-feature": {
			features: Features{
				DeviceBindingAndStatus: true, // add to forcefully use experimenting allocator
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			classes:       objects(class(classA, driverA)),
			slices:        unwrap(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:          node(node1, region1),
			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: has capacity requests, but the DRAConsumableCapacity feature is disabled")),
		},
		"consumable-capacity-disabled-feature-with-prioritized-list": {
			features: Features{
				PrioritizedList:        true,
				DeviceBindingAndStatus: true, // add to forcefully use experimenting allocator
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(
						req0,
						subRequest(subReq0, classA, 1).withCapacityRequest(ptr.To(one)),
						subRequest(subReq1, classA, 1).withCapacityRequest(ptr.To(one)),
					),
				),
			),
			classes:       objects(class(classA, driverA)),
			slices:        unwrap(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:          node(node1, region1),
			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("claim claim-0, subrequest subReq-0: has capacity requests, but the DRAConsumableCapacity feature is disabled")),
		},
		"consumable-capacity-multi-allocatable-device-with-missing-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity1: one}, nil).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"consumable-capacity-multi-allocatable-device-allocation-mode-all-with-missing-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(ptr.To(one))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity1: one}, nil).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"consumable-capacity-multi-allocatable-device-without-capacity-without-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1)),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{ // both share the same device; no capacity is consumed since none is defined.
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, nil),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, nil),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-without-policy-without-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1)),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
					device(device2, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{ // both should get full capacity for different devices
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device2).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-without-policy-without-capacity-request-block": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1)),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}, nil).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{}, // should not satisfy both
		},
		"consumable-capacity-multi-allocatable-device-without-policy-with-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-with-policy-without-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
				claimWithRequests(claim1, nil, request(req0, classA, 1)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			),
			node: node(node1, region1),

			expectResults: []any{ // both requests applied the default request (two)
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-with-zero-request-policy": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
				claimWithRequests(claim1, nil, request(req0, classA, 1)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyValidValues(zero, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil),
				),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: zero}),
				),
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: zero}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-violates-zero-request-policy": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyValidValues(zero, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil),
				),
			),
			node: node(node1, region1),

			expectResults: []any{},
		},
		"consumable-capacity-multi-allocatable-device-with-range-policy-with-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{ // both requests allocated to the same device
				allocationResult(
					localNodeSelector(node1),
					// capacity must be rounded up to default minimum (two)
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
				allocationResult(
					localNodeSelector(node1),
					// capacity must be rounded up to default minimum (two)
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-with-range-policy-step": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(resource.NewQuantity(3, resource.BinarySI))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					// capacity must be rounded up with Step (3 to 2+2 = 4)
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-with-valid-values-policy": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(resource.NewQuantity(2, resource.BinarySI))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyValidValues(one, map[resourceapi.QualifiedName]resource.Quantity{capacity0: four},
						[]resource.Quantity{two}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-with-valid-values-policy-round-up": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(resource.NewQuantity(2, resource.BinarySI))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyValidValues(one, map[resourceapi.QualifiedName]resource.Quantity{capacity0: three},
						[]resource.Quantity{three}), // capacity value must be explicitly added
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: three}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-with-exceeded-consumable-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(two))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			),
			node: node(node1, region1),

			expectResults: []any{},
		},
		"consumable-capacity-multi-allocatable-device-with-request-over-max-sharing-policy": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(resource.NewQuantity(6, resource.BinarySI))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: *resource.NewQuantity(10, resource.BinarySI)}),
				),
			),
			node: node(node1, region1),

			expectResults: []any{}, // default max requestPolicy is 4, should not allocate even though 6 < 10
		},
		"consumable-capacity-multi-allocatable-device-with-allocated-capacity-use-remaining": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(two))),
			),
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					capacity0: ptr.To(two),
				},
			},
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-with-allocated-capacity-next-device": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(two))),
			),
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					capacity0: ptr.To(two),
				},
			},
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
					device(device2, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device2).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			},
		},
		"consumable-capacity-with-multi-allocatable-device-backtrack": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				// 2 requests with two per each
				claim(claim0).withConstraints(resourceapi.DeviceConstraint{MatchAttribute: &stringAttribute}).withRequests(
					deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(two)),
					deviceRequest(req1, classA, 1).withCapacityRequest(ptr.To(two)),
				),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(two))),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device2, nil,
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"stringAttribute": {StringValue: ptr.To("stringAttributeValue1")},
						}).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
					device(device1, nil,
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
							"stringAttribute": {StringValue: ptr.To("stringAttributeValue2")},
						}).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
					deviceRequestAllocationResult(req1, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
				allocationResult(localNodeSelector(node1), deviceRequestAllocationResult(req0, driverA, pool1, device2).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two})),
			},
		},
		"consumable-capacity-multi-allocatable-device-with-subrequest": {
			features: Features{
				ConsumableCapacity: true,
				PrioritizedList:    true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 1).withCapacityRequest(ptr.To(four)),
						subRequest(subReq1, classA, 1).withCapacityRequest(ptr.To(two)),
						subRequest(subReq1, classA, 1),
					),
				),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0SubReq1, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-with-no-available-consumable-capacity": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(two))),
			),
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					capacity0: ptr.To(two),
				},
			},
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			),
			node: node(node1, region1),

			expectResults: []any{},
		},
		"consumable-capacity-multi-allocatable-device-allocation-mode-all-filter-out-no-capacity": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(ptr.To(two))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}, nil).withAllowMultipleAllocations(),
					device(device2, nil, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{ // only match device is allocated
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-allocation-mode-all-filter-out-insufficient-capacity": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(ptr.To(two))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}, nil).withAllowMultipleAllocations(),
					device(device2, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{ // only match device is allocated
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-allocation-mode-all-allocating-sufficient": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one)),
					allDeviceRequest(req1, classA).withCapacityRequest(ptr.To(one)),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}, nil).withAllowMultipleAllocations(),
					device(device2, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
					deviceRequestAllocationResult(req1, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
					deviceRequestAllocationResult(req1, driverA, pool1, device2).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-allocation-mode-all-in-use-sufficient": {
			features: Features{
				ConsumableCapacity: true,
			},
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					capacity0: ptr.To(one),
				},
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					allDeviceRequest(req0, classA).withCapacityRequest(ptr.To(one)),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}, nil).withAllowMultipleAllocations(),
					device(device2, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
					deviceRequestAllocationResult(req0, driverA, pool1, device2).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
				),
			},
		},
		"consumable-capacity-multi-allocatable-device-allocation-mode-all-allocating-insufficient": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one)),
					allDeviceRequest(req1, classA).withCapacityRequest(ptr.To(one)),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
					device(device2, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"consumable-capacity-multi-allocatable-device-allocation-mode-all-in-use-insufficient": {
			features: Features{
				ConsumableCapacity: true,
			},
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					capacity0: ptr.To(one),
				},
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					allDeviceRequest(req0, classA).withCapacityRequest(ptr.To(one)),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
					device(device2, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device2).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
				),
			},
		},
		"consumable-capacity-dedicated-device-with-consumable-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, false)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil),
				),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
				),
			},
		},
		"consumable-capacity-dedicated-device-with-multiple-consumable-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, false)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"allow-multiple-allocations-exclude-multi-allocatable-device-by-class-selector": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, false)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),

			expectResults: []any{},
		},
		"allow-multiple-allocations-not-allocate-multi-allocatable-device-which-already-dedicated": {
			features: Features{
				ConsumableCapacity: true,
			},
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			),
			node: node(node1, region1),

			expectResults: []any{},
		},
		"allow-multiple-allocations-not-allocate-dedicated-device-which-change-from-multi-allocatable": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one))),
			),
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					capacity0: ptr.To(one),
				},
			},
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			),
			node: node(node1, region1),

			expectResults: []any{},
		},
		"allow-multiple-allocations-with-partitionable-device": {
			features: Features{
				PrioritizedList:      true,
				PartitionableDevices: true,
				ConsumableCapacity:   true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("6Gi")) >= 0`, driverA),
							}},
						),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("4Gi")) >= 0`, driverA),
							}},
						),
					),
				),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withAllowMultipleAllocations(),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("6Gi"),
							},
						),
					).withAllowMultipleAllocations(),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withAllowMultipleAllocations(),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{"memory": resource.MustParse("4Gi")}),
				deviceRequestAllocationResult(req1SubReq1, driverA, pool1, device3).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{"memory": resource.MustParse("4Gi")}),
			)},
		},
		"consumable-capacity-with-partitionable-device-multiple-capacity-pools": {
			// This test case combines integration of PrioritizedList, PartitionableDevices, and ConsumableCapacity features.
			features: Features{
				PrioritizedList:      true,
				PartitionableDevices: true,
				ConsumableCapacity:   true,
			},
			// There are two basic requests (req0 and req2), requesting 1 and 2 units of capacity0, respectively.
			// In addition, there is one request (req1) with a prioritized list.
			// The prioritized request prefers devices with capacity0 >= 4 over those with capacity0 >= 2.
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one)),
					deviceRequest(req2, classA, 1).withCapacityRequest(ptr.To(two)),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"]["%s"].compareTo(quantity("4")) >= 0`, driverA, capacity0),
							}}).withCapacityRequest(ptr.To(one)),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"]["%s"].compareTo(quantity("2")) >= 0`, driverA, capacity0),
							}}).withCapacityRequest(ptr.To(one)),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			// There three device option consuming counters from the PartitionableDevices's CounterSet.
			// All devices can be allocated multiple times with `allowMultipleAllocations=true`.
			// CounterSet (4, 8)
			//   device1, device2, and device3 consume (2,4), (4,4), and (2,4) for (capacity0, capacity1) respectively.
			//   i.e., only two options are valid those are [device1, device3] or [device2].
			// Capacity.RequestPolicy of ConsumableCapacity forces the capacity1 consuming with range policy (min,step,max)=(2,2,4).
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								capacity0: two,
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								capacity1: four,
							},
						),
					).withAllowMultipleAllocations().withCapacityRequestPolicyRange((map[resourceapi.QualifiedName]resource.Quantity{capacity1: four})),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								capacity0: four,
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								capacity1: four,
							},
						),
					).withAllowMultipleAllocations().withCapacityRequestPolicyRange((map[resourceapi.QualifiedName]resource.Quantity{capacity1: four})),
					device(device3, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								capacity0: two,
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								capacity1: four,
							},
						),
					).withAllowMultipleAllocations().withCapacityRequestPolicyRange((map[resourceapi.QualifiedName]resource.Quantity{capacity1: four})),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							capacity0: four,
						},
					),
					counterSet(counterSet2,
						map[string]resource.Quantity{
							capacity1: resource.MustParse("8"),
						},
					),
				),
			),
			node: node(node1, region1),
			// Expected results:
			//   - [req0] The scheduler should be able to allocate device1 to req0.
			//     1 unit of capacity0 is consumed according to the requested capacity, and 2 units of capacity1 are consumed according to the RequestPolicy.
			//   - [req2] device1 does not have enough capacity0 remaining for req2, which requests 2 units (only 1 unit remains).
			//     The scheduler must also skip device2 due to the CounterSet constraint.
			//     Therefore, device3 is allocated to req2, providing 2 units of capacity0 and 2 units of capacity1.
			//   - [req1] The first prioritized subrequest of req1 cannot find a device because device2 cannot be selected.
			//     The second prioritized subrequest is then chosen and consumes the remaining 1 unit of capacity0 on device1.
			//     Consequently, device1 is allocated to subreq1 of req1 with 1 unit of capacity0 and 2 units of capacity1.
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one, capacity1: two}),
				deviceRequestAllocationResult(req2, driverA, pool1, device3).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two, capacity1: two}),
				deviceRequestAllocationResult(req1SubReq1, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one, capacity1: two}),
			)},
		},
		"with-distinct-constraints": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(
					resourceapi.DeviceConstraint{DistinctAttribute: &boolAttribute, Requests: []string{req0, req1}},
					resourceapi.DeviceConstraint{DistinctAttribute: &versionAttribute, Requests: []string{req0, req1, req2}},
					resourceapi.DeviceConstraint{DistinctAttribute: &intAttribute, Requests: []string{req0, req1, req2, req3}},
				).withRequests(
					deviceRequest(req0, classA, 1),
					deviceRequest(req1, classA, 1),
					deviceRequest(req2, classA, 1),
					deviceRequest(req3, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
						"driverVersion": {VersionValue: ptr.To("1.0.0")},
						"numa":          {IntValue: ptr.To(int64(0))},
						"boolAttribute": {BoolValue: ptr.To(true)},
					}).withAllowMultipleAllocations(),
					device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
						"driverVersion": {VersionValue: ptr.To("1.0.1")},
						"numa":          {IntValue: ptr.To(int64(1))},
						"boolAttribute": {BoolValue: ptr.To(false)},
					}).withAllowMultipleAllocations(),
					device(device3, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
						"driverVersion": {VersionValue: ptr.To("1.0.2")},
						"numa":          {IntValue: ptr.To(int64(2))},
						"boolAttribute": {BoolValue: ptr.To(true)},
					}).withAllowMultipleAllocations(),
					device(device4, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
						"driverVersion": {VersionValue: ptr.To("1.0.0")},
						"numa":          {IntValue: ptr.To(int64(3))},
						"boolAttribute": {BoolValue: ptr.To(true)},
					}).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, nil),
				deviceRequestAllocationResult(req1, driverA, pool1, device2).withConsumedCapacity(&fixedShareID, nil),
				deviceRequestAllocationResult(req2, driverA, pool1, device3).withConsumedCapacity(&fixedShareID, nil),
				deviceRequestAllocationResult(req3, driverA, pool1, device4).withConsumedCapacity(&fixedShareID, nil),
			)},
		},
		"with-distinct-constraints-attribute-not-set": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(
					resourceapi.DeviceConstraint{DistinctAttribute: &boolAttribute, Requests: []string{req0}},
				).withRequests(
					deviceRequest(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),

			expectResults: []any{},
		},
		"with-distinct-constraints-unknown-type-attribute": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(
					resourceapi.DeviceConstraint{DistinctAttribute: &boolAttribute, Requests: []string{req0}},
				).withRequests(
					deviceRequest(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"boolAttribute": {}}).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
			expectError:   gomega.MatchError(gomega.ContainSubstring("unsupported attribute value")),
		},
		"with-distinct-constraints-with-subrequests": {
			features: Features{
				ConsumableCapacity: true,
				PrioritizedList:    true,
			},
			// There are three requests, all preferring the boolAttribute.
			// However, the DeviceConstraint applies only to the first two requests.
			claimsToAllocate: objects(
				claim(claim0).withConstraints(
					resourceapi.DeviceConstraint{DistinctAttribute: &boolAttribute, Requests: []string{req0, req1}}).
					withRequests(
						requestWithPrioritizedList(req0,
							subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
								CEL: &resourceapi.CELDeviceSelector{
									Expression: fmt.Sprintf(`device.attributes["%s"].boolAttribute`, driverA),
								}}),
							subRequest(subReq1, classA, 1),
						),
						requestWithPrioritizedList(req1,
							subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
								CEL: &resourceapi.CELDeviceSelector{
									Expression: fmt.Sprintf(`device.attributes["%s"].boolAttribute`, driverA),
								}}),
							subRequest(subReq1, classA, 1),
						),
						requestWithPrioritizedList(req2,
							subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
								CEL: &resourceapi.CELDeviceSelector{
									Expression: fmt.Sprintf(`device.attributes["%s"].boolAttribute`, driverA),
								}}),
							subRequest(subReq1, classA, 1),
						),
					),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
						"boolAttribute": {BoolValue: ptr.To(true)},
					}).withAllowMultipleAllocations(),
					device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
						"boolAttribute": {BoolValue: ptr.To(false)},
					}).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			// Because the constraint applies only to req0 and req1,
			// subreq0 cannot satisfy req1, so device2 (subreq1) is allocated instead.
			// However, req2 has no such constraint and can use device1 (subreq0).
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0SubReq0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, nil),
				deviceRequestAllocationResult(req1SubReq1, driverA, pool1, device2).withConsumedCapacity(&fixedShareID, nil),
				deviceRequestAllocationResult(req2SubReq0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, nil),
			)},
		},
		"distinct-constraint-one-multi-allocatable-device-with-distinct-constraint": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(resourceapi.DeviceConstraint{DistinctAttribute: &stringAttribute}).withRequests(
					deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(one)),
					deviceRequest(req1, classA, 1).withCapacityRequest(ptr.To(one)),
				)),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil,
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"stringAttribute": {StringValue: ptr.To("stringAttributeValue")}},
					).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"distinct-constraint-two-multi-allocatable-devices-with-distinct-constraint": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(resourceapi.DeviceConstraint{DistinctAttribute: &stringAttribute}).withRequests(
					deviceRequest(req0, classA, 1).withCapacityRequest(ptr.To(two)),
					deviceRequest(req1, classA, 1).withCapacityRequest(ptr.To(two)),
				),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil,
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"stringAttribute": {StringValue: ptr.To("stringAttributeValue1")}},
					).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device2, nil,
						map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"stringAttribute": {StringValue: ptr.To("stringAttributeValue2")}},
					).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
					deviceRequestAllocationResult(req1, driverA, pool1, device2).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			},
		},

		"allocation-mode-all-with-multi-host-resource-pool": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					DeviceClassName: classA,
				},
			})),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil),
				),
				sliceWithDevices(slice2, node2, resourcePool(pool1, 2), driverA,
					device(device2, nil, nil),
				),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
				),
			},
		},
		"multi-host-resource-pool-with-stale-slice-for-current-node": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node1, pool1, driverA,
						device(device1, nil, nil),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node2, pool1, driverA,
						device(device2, nil, nil),
					)
					s.Spec.Pool.Generation = 2
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
			),
			node: node(node1, region1),
		},

		// update-from-node-1-to-node-2 and vice-versa is dangerous,
		// DRA drivers should not do this: there is no guarantee that
		// the scheduler sees the change and then might allocate a
		// device for a node where it is no longer available.
		//
		// We do not have to support such a change exactly as expected
		// by these test cases if it makes the implementation too
		// expensive because of additional checks.
		//
		// The support for allocation from incomplete pools is also under
		// debate. It's currently done because without scoring it makes
		// no difference how many slices are available, as long as one
		// device can be found.

		"multi-host-resource-pool-complete-update-from-node-1-to-node-2": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				// Node 1, generation 1.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node1, pool1, driverA,
						device(device1, nil, nil),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node1, pool1, driverA,
						device(device2, nil, nil),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				// Node 2, generation 2.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node2, pool1, driverA,
						device(device1, nil, nil),
					)
					s.Spec.Pool.Generation = 2
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node2, pool1, driverA,
						device(device2, nil, nil),
					)
					s.Spec.Pool.Generation = 2
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
			),
			node: node(node1, region1),

			// We'd like this to *not* be allocated, but the current implementation ignores
			// the updated slices. Allocating from the old pool is also what would happen
			// if the scheduler hadn't received the update yet, so DRA drivers shouldn't
			// depend on better handling of this situation.
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
				),
			},
		},

		"multi-host-resource-pool-complete-update-from-node-2-to-node-1": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				// Node 2, generation 1.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node2, pool1, driverA,
						device(device1, nil, nil),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node2, pool1, driverA,
						device(device2, nil, nil),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				// Node 1, generation 2.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node1, pool1, driverA,
						device(device1, nil, nil),
					)
					s.Spec.Pool.Generation = 2
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node1, pool1, driverA,
						device(device2, nil, nil),
					)
					s.Spec.Pool.Generation = 2
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
				),
			},
		},

		"multi-host-resource-pool-incomplete-update-from-node-1-to-node-2": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				// Node 1, generation 1.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node1, pool1, driverA,
						device(device1, nil, nil),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node1, pool1, driverA,
						device(device2, nil, nil),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				// Node 2, generation 2.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node2, pool1, driverA,
						device(device1, nil, nil),
					)
					s.Spec.Pool.Generation = 2
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
			),
			node: node(node1, region1),

			// We'd like this to *not* be allocated, but the current implementation ignores
			// the updated slices. Allocating from the old pool is also what would happen
			// if the scheduler hadn't received the update yet, so DRA drivers shouldn't
			// depend on better handling of this situation.
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0, driverA, pool1, device1, false),
				),
			},
		},

		"multi-host-resource-pool-incomplete-update-from-node-2-to-node-1": {
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				// Node 2, generation 1.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node2, pool1, driverA,
						device(device1, nil, nil),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node2, pool1, driverA,
						device(device2, nil, nil),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				// Node 1, generation 2.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node1, pool1, driverA,
						device(device1, nil, nil),
					)
					s.Spec.Pool.Generation = 2
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
			),
			node: node(node1, region1),
			// This does not get allocated since slice2 is incomplete.
		},
		"partitionable-devices-same-counter-set-name-in-different-resource-slices": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 4), driverA,
					device(device1, nil, nil),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 4), driverA,
					counterSet(counterSet1, nil),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool1, 4), driverA,
					device(device2, nil, nil),
				),
				sliceWithCounterSets(slice4, node1, resourcePool(pool1, 4), driverA,
					counterSet(counterSet1, nil),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("invalid resource pools were encountered")),
		},
		"partitionable-devices-device-counter-consumption-references-unknown-counter-set": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet2, nil),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, nil),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("invalid resource pools were encountered")),
		},
		"partitionable-devices-device-counter-consumption-references-unknown-counter-in-counter-set": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						}),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"other-memory": resource.MustParse("8Gi"),
					}),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("invalid resource pools were encountered")),
		},
		"partitionable-devices-device-counter-consumption-references-counter-set-in-other-pool": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("8Gi"),
						}),
					),
				),
				sliceWithCounterSets(slice2, node1, pool2, driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("invalid resource pools were encountered")),
		},
		"partitionable-devices-devices-on-different-nodes-consume-same-counter-set": {
			features: Features{
				PartitionableDevices: true,
				PrioritizedList:      true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("6Gi")) >= 0`, driverA),
							},
						}),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
							},
						}),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithCounterSets(slice1, node1, resourcePool(pool1, 3), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 3), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("6Gi"),
						}),
					),
					device(device2, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("2Gi"),
						}),
					),
				),
				sliceWithDevices(slice3, node2, resourcePool(pool1, 3), driverA,
					device(device3, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("6Gi"),
						}),
					),
				),
			),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device3),
			},
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device2, false),
			)},
		},
		"partitionable-devices-validation-error-on-non-targeting-slice": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithCounterSets(slice1, node1, resourcePool(pool1, 3), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 3), driverA,
					device(device1, fromCounters, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("6Gi"),
						}),
					),
				),
				sliceWithDevices(slice3, node2, resourcePool(pool1, 3), driverA,
					device(device3, nil, nil).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet2, map[string]resource.Quantity{
							"memory": resource.MustParse("6Gi"),
						}),
					),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("invalid resource pools were encountered")),
		},
		"different-resourceslices-in-pool-can-target-different-nodes": {
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node2, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 2), driverA,
					device(device2, nil, nil),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"invalid-pool-is-ok-if-devices-can-be-allocated-from-other": {
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool2, 1), driverA,
					device(device2, nil, nil),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool2, device2, false),
			)},
		},
		"invalid-pool-is-error-if-no-other-device-can-be-allocated": {
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool2, 1), driverA,
					device(device2, nil, nil),
				),
			),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool2, device2),
			},
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("invalid resource pools were encountered")),
		},
		"no-allocation-from-incomplete-pools": {
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1, nil, nil),
				),
			),
			node: node(node1, region1),
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			g := gomega.NewWithT(t)

			required := tc.features.Set()
			supported := supportedFeatures.Set()
			missing := required.Difference(supported)
			if missing.Len() > 0 {
				// Skip the test, at least one of its required features isn't supported
				// and the test would fail.
				t.Skipf("SKIP: required feature(s) %v not supported by allocator", sets.List(missing))
			}

			// Listing objects is deterministic and returns them in the same
			// order as in the test case. That makes the allocation result
			// also deterministic.
			var classLister informerLister[resourceapi.DeviceClass]
			for _, class := range tc.classes {
				classLister.objs = append(classLister.objs, class.DeepCopy())
			}
			claimsToAllocate := slices.Clone(tc.claimsToAllocate)
			allocatedDevices := slices.Clone(tc.allocatedDevices)
			allocatedShare := tc.allocatedCapacityDevices.Clone()
			slices := slices.Clone(tc.slices)
			allocatedState := AllocatedState{
				AllocatedDevices:         sets.New(allocatedDevices...),
				AllocatedSharedDeviceIDs: tc.allocatedSharedDeviceIDs,
				AggregatedCapacity:       allocatedShare,
			}
			allocator, err := newAllocator(ctx, tc.features, allocatedState, classLister, slices, cel.NewCache(1, cel.Features{EnableConsumableCapacity: tc.features.ConsumableCapacity}))
			g.Expect(err).ToNot(gomega.HaveOccurred())

			if _, ok := allocator.(internal.AllocatorExtended); tc.expectNumAllocateOneInvocations > 0 && !ok {
				t.Skipf("%T does not support the AllocatorStats interface", allocator)
			}

			results, err := allocator.Allocate(ctx, tc.node, unwrap(claimsToAllocate...))
			matchError := tc.expectError
			if matchError == nil {
				matchError = gomega.Not(gomega.HaveOccurred())
			}
			g.Expect(err).To(matchError)

			t.Logf("name: %s", name)
			// replace any share id with fixed value for testing
			for ri, result := range results {
				for ai, allocation := range result.Devices.Results {
					if allocation.ShareID != nil {
						results[ri].Devices.Results[ai].ShareID = &fixedShareID
					}
					t.Logf("allocated capacity: %v", allocation.ConsumedCapacity)
				}
			}
			g.Expect(results).To(gomega.ConsistOf(tc.expectResults...))

			// Objects that the allocator had access to should not have been modified.
			g.Expect(claimsToAllocate).To(gomega.HaveExactElements(tc.claimsToAllocate))
			g.Expect(allocatedDevices).To(gomega.HaveExactElements(tc.allocatedDevices))
			g.Expect(slices).To(gomega.ConsistOf(tc.slices))
			g.Expect(classLister.objs).To(gomega.ConsistOf(tc.classes))

			if tc.expectNumAllocateOneInvocations > 0 {
				stats := allocator.(internal.AllocatorExtended).GetStats()
				g.Expect(stats.NumAllocateOneInvocations).To(gomega.Equal(tc.expectNumAllocateOneInvocations))
			}
		})
	}

	t.Run("interrupt", func(t *testing.T) {
		for _, name := range []string{"off", "timeout", "deadline", "cancel"} {
			t.Run(name, func(t *testing.T) {
				_, ctx := ktesting.NewTestContext(t)
				g := gomega.NewWithT(t)

				// This testcase is a smaller variant of the one in https://github.com/kubernetes/kubernetes/issues/131730#issuecomment-2873598287.
				// That one took over 30 seconds, this one here only 0.07 seconds.
				// But even that is too long when we interrupt in the near future or
				// even before starting...
				classLister := informerLister[resourceapi.DeviceClass]{
					objs: []*resourceapi.DeviceClass{class(classA, driverA)},
				}
				claimsToAllocate := unwrap(claimWithRequests(claim0, nil,
					request(req0, classA, 6),
				))
				slices := unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1, nil, nil),
					device(device2, nil, nil),
					device(device3, nil, nil),
					device(device4, nil, nil),
					device("device-5", nil, nil),
				))
				node := node(node1, region1)

				switch name {
				case "off":
				case "timeout":
					c, cancel := context.WithTimeout(ctx, time.Nanosecond)
					defer cancel()
					ctx = c
				case "deadline":
					c, cancel := context.WithDeadline(ctx, time.Now())
					defer cancel()
					ctx = c
				case "cancel":
					c, cancel := context.WithCancel(ctx)
					cancel()
					ctx = c
				}

				allocator, err := newAllocator(ctx, Features{}, AllocatedState{}, classLister, slices, cel.NewCache(1, cel.Features{}))
				g.Expect(err).ToNot(gomega.HaveOccurred())
				_, err = allocator.Allocate(ctx, node, claimsToAllocate)
				t.Logf("got error %v", err)
				if ctx.Err() != nil {
					if !errors.Is(err, ctx.Err()) {
						t.Fatalf("expected %v, got error: %v", ctx.Err(), err)
					}
				} else {
					if err != nil {
						t.Fatalf("expected no error, got %v", err)
					}
				}
			})
		}
	})
}

type informerLister[T any] struct {
	objs []*T
	err  error
}

func (l informerLister[T]) List() (ret []*T, err error) {
	return l.objs, l.err
}

func (l informerLister[T]) Get(name string) (*T, error) {
	for _, obj := range l.objs {
		accessor, err := meta.Accessor(obj)
		if err != nil {
			return nil, err
		}
		if accessor.GetName() == name {
			return obj, nil
		}
	}
	return nil, apierrors.NewNotFound(schema.GroupResource{}, "not found")
}
