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
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/blang/semver/v4"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"
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
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/component-helpers/nodedeclaredfeatures/features/draoptionalnodeoperations"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured/internal"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

type Allocator = internal.Allocator
type DeviceClassLister = internal.DeviceClassLister
type Features = internal.Features
type DeviceID = internal.DeviceID

type ConsumedCapacityCollection = internal.ConsumedCapacityCollection
type ConsumedCapacity = internal.ConsumedCapacity
type AllocatedState = internal.AllocatedState
type DeviceAffinityState = internal.DeviceAffinityState
type AffinityStateCollection = internal.AffinityStateCollection
type SharingAffinityState = internal.SharingAffinityState
type PoolID = schedulerapi.PoolID

type allocatorFactory func(context.Context, Features, AllocatedState, DeviceClassLister, []*resourceapi.ResourceSlice, *cel.Cache) (Allocator, error)

type affinityAllocator interface {
	AllocateWithAffinity(context.Context, *v1.Node, []*resourceapi.ResourceClaim) ([]resourceapi.AllocationResult, []internal.ClaimAffinity, error)
}

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
	req2SubReq1 = "req-2/subReq-1"
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
	counter0    = "counter0"
	counter1    = "counter1"
	capacity0   = resourceapi.QualifiedName("capacity-0")
	capacity1   = resourceapi.QualifiedName("capacity-1")
)

var (
	fixedShareID = apitypes.UID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

	zero  = *resource.NewQuantity(0, resource.BinarySI)
	one   = *resource.NewQuantity(1, resource.BinarySI)
	two   = *resource.NewQuantity(2, resource.BinarySI)
	three = *resource.NewQuantity(3, resource.BinarySI)
	four  = *resource.NewQuantity(4, resource.BinarySI)

	pointOne     = resource.MustParse("100m")
	pointTwo     = resource.MustParse("200m")
	pointTwoFive = resource.MustParse("250m")
	// pointThree uses NewMilliQuantity so its s field matches allocator output (s:"").
	pointThree = *resource.NewMilliQuantity(300, resource.DecimalSI)
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

// nodeWithDRAOptionalNodeOperations returns a node declaring DRAOptionalNodeOperations feature support
// required by devices that define SkipNodeOperations.
func nodeWithDRAOptionalNodeOperations(name, region string) *v1.Node {
	n := node(name, region)
	n.Status.DeclaredFeatures = append(n.Status.DeclaredFeatures, draoptionalnodeoperations.DRAOptionalNodeOperationsFeatureGate)
	return n
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
		case []resourceapi.DeviceDerivedAttribute:
			subRequest.DerivedAttributes = append(subRequest.DerivedAttributes, typedField...)
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

func (in wrapDeviceRequest) withCapacityRequest(name resourceapi.QualifiedName, quantity resource.Quantity) wrapDeviceRequest {
	out := in.DeepCopy()
	addCapacityRequest(&out.Exactly.Capacity, name, quantity)
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

func (in wrapDeviceSubRequest) withCapacityRequest(name resourceapi.QualifiedName, quantity resource.Quantity) wrapDeviceSubRequest {
	out := in.DeepCopy()
	addCapacityRequest(&out.Capacity, name, quantity)
	return wrapDeviceSubRequest{*out}
}

func addCapacityRequest(capacity **resourceapi.CapacityRequirements, name resourceapi.QualifiedName, quantity resource.Quantity) {
	if *capacity == nil {
		*capacity = &resourceapi.CapacityRequirements{}
	}
	if (*capacity).Requests == nil {
		(*capacity).Requests = make(map[resourceapi.QualifiedName]resource.Quantity)
	}
	(*capacity).Requests[name] = quantity
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

// generate a Device object with the given name.
func device(name string) wrapDevice {
	device := resourceapi.Device{
		Name: name,
	}
	return wrapDevice{Device: device}
}

type wrapDevice struct {
	resourceapi.Device
}

func (in wrapDevice) obj() resourceapi.Device {
	return in.Device
}

// withAttribute adds or overwrites an attribute.
// The value can be an int, int64, string, bool, semver, or a list of those.
// A DeviceAttribute can be passed to create an invalid empty value.
func (in wrapDevice) withAttribute(name resourceapi.QualifiedName, value any) wrapDevice {
	device := in.Device.DeepCopy()
	if device.Attributes == nil {
		device.Attributes = make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
	}
	var attr resourceapi.DeviceAttribute
	switch value := value.(type) {
	case int:
		attr.IntValue = new(int64(value))
	case int64:
		attr.IntValue = new(value)
	case string:
		attr.StringValue = new(value)
	case bool:
		attr.BoolValue = new(value)
	case semver.Version:
		attr.VersionValue = new(value.String())
	case []int:
		for _, value := range value {
			attr.IntValues = append(attr.IntValues, int64(value))
		}
	case []int64:
		attr.IntValues = slices.Clone(value)
	case []string:
		attr.StringValues = slices.Clone(value)
	case []bool:
		attr.BoolValues = slices.Clone(value)
	case []semver.Version:
		for _, value := range value {
			attr.VersionValues = append(attr.VersionValues, value.String())
		}
	case resourceapi.DeviceAttribute:
		attr = value
	default:
		panic(fmt.Sprintf("unexpected attribute %q value %T: %+v", name, value, value))
	}
	device.Attributes[name] = attr
	return wrapDevice{Device: *device}
}

// withCapacity adds or overwrites a capacity.
// The value can be a quantity, string (gets parsed as quantity).
// A DeviceCapacity can be passed to create an invalid empty value.
func (in wrapDevice) withCapacity(name resourceapi.QualifiedName, value any) wrapDevice {
	device := in.Device.DeepCopy()
	if device.Capacity == nil {
		device.Capacity = make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity)
	}
	switch value := value.(type) {
	case resource.Quantity:
		device.Capacity[name] = resourceapi.DeviceCapacity{Value: value}
	case resourceapi.DeviceCapacity:
		device.Capacity[name] = value
	case string:
		device.Capacity[name] = resourceapi.DeviceCapacity{Value: resource.MustParse(value)}
	default:
		panic(fmt.Sprintf("unexpected capacity %q value %T: %+v", name, value, value))
	}
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withTaints(taints ...resourceapi.DeviceTaint) wrapDevice {
	device := in.Device.DeepCopy()
	device.Taints = append(device.Taints, taints...)
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withDeviceCounterConsumption(deviceCounterConsumption ...resourceapi.DeviceCounterConsumption) wrapDevice {
	device := in.Device.DeepCopy()
	device.ConsumesCounters = append(device.ConsumesCounters, deviceCounterConsumption...)
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withNodeSelection(nodeSelection any) wrapDevice {
	device := in.Device.DeepCopy()
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
	device := in.Device.DeepCopy()
	device.BindingConditions = bindingConditions
	device.BindingFailureConditions = bindingFailureConditions
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withBindsToNode(bindsToNode bool) wrapDevice {
	device := in.Device.DeepCopy()
	device.BindsToNode = ptr.To(bindsToNode)
	return wrapDevice{Device: *device}
}

func (in wrapDevice) withAllowMultipleAllocations() wrapDevice {
	device := in.Device.DeepCopy()
	device.AllowMultipleAllocations = ptr.To(true)
	return wrapDevice{Device: *device}
}

// withCapacityRequestPolicyRange adds capacity with default requestPolicy (2,2,4)
func (in wrapDevice) withCapacityRequestPolicyRange(capacity map[resourceapi.QualifiedName]resource.Quantity) wrapDevice {
	device := in.Device.DeepCopy()
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

// withFractionalCapacityRequestPolicyRange adds capacity with a fractional requestPolicy
// (min=200m, step=100m, max=1, default=200m).
func (in wrapDevice) withFractionalCapacityRequestPolicyRange(capacity map[resourceapi.QualifiedName]resource.Quantity) wrapDevice {
	device := in.Device.DeepCopy()
	if device.Capacity == nil {
		device.Capacity = make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, len(capacity))
	}
	for name, quantity := range capacity {
		device.Capacity[name] = resourceapi.DeviceCapacity{
			Value: quantity,
			RequestPolicy: &resourceapi.CapacityRequestPolicy{
				Default: &pointTwo,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{
					Min:  &pointTwo,
					Step: &pointOne,
					Max:  &one,
				},
			},
		}
	}
	return wrapDevice{Device: *device}
}

// withCapacityRequestPolicyValidValues adds capacity with default valid values (1)
func (in wrapDevice) withCapacityRequestPolicyValidValues(defaultValue resource.Quantity, capacity map[resourceapi.QualifiedName]resource.Quantity, additionalValidValues []resource.Quantity) wrapDevice {
	device := in.Device.DeepCopy()
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

func deviceCounterConsumptionWithGroups(counterSet string, counters map[string]resource.Quantity, compatibilityGroups ...string) resourceapi.DeviceCounterConsumption {
	return resourceapi.DeviceCounterConsumption{
		CounterSet:          counterSet,
		Counters:            toCounters(counters),
		CompatibilityGroups: compatibilityGroups,
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

func (in wrapResourceSliceWithDevices) withSkipNodeOperations(skip ...resourceapi.SkipNodeOperation) wrapResourceSliceWithDevices {
	out := in.DeepCopy()
	out.Spec.SkipNodeOperations = skip
	return wrapResourceSliceWithDevices{ResourceSlice: out}
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

func deviceAllocationResultWithSkipNodeOperations(request, driver, pool, device string, skip ...resourceapi.SkipNodeOperation) resourceapi.DeviceRequestAllocationResult {
	r := deviceAllocationResult(request, driver, pool, device, false)
	r.SkipNodeOperations = skip
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
	return sliceWithDevices(name, nodeSelection, pool, driver, device(device1))
}

// generate a ResourceSclie object with the given parameters and the specified number of devices.
func sliceWithMultipleDevices(name string, nodeSelection, pool any, driver string, count int) wrapResourceSliceWithDevices {
	var devices []wrapDevice
	for i := 0; i < count; i++ {
		devices = append(devices, device(fmt.Sprintf("device-%d", i)))
	}
	return sliceWithDevices(name, nodeSelection, pool, driver, devices...)
}

func counterSet(name string, counters map[string]resource.Quantity) resourceapi.CounterSet {
	return resourceapi.CounterSet{
		Name:     name,
		Counters: toCounters(counters),
	}
}

func toCounters(counters map[string]resource.Quantity) map[string]resourceapi.Counter {
	out := make(map[string]resourceapi.Counter, len(counters))
	for name, quantity := range counters {
		out[name] = resourceapi.Counter{Value: quantity}
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

type AllocatorTestCase struct {
	features                 Features
	claimsToAllocate         []wrapResourceClaim
	allocatedDevices         []DeviceID
	allocatedSharedDeviceIDs sets.Set[DeviceID]
	allocatedCapacityDevices ConsumedCapacityCollection
	affinityStates           AffinityStateCollection
	stalePools               sets.Set[PoolID]
	classes                  []*resourceapi.DeviceClass
	slices                   []*resourceapi.ResourceSlice
	node                     *v1.Node

	expectResults          []any
	expectResultsByChannel map[internal.AllocatorChannel][]any
	expectError            types.GomegaMatcher // can be used to check for no error or match specific error
	expectErrorIs          error               // optional error to check with errors.Is

	// Isolation tests may mutate the received state only after all allocator
	// assertions, so their deliberate writes cannot hide allocator mutations.
	inspectAllocatedState func(t *testing.T, received AllocatedState)

	// Expected allocateOne invocations, asserted for every variant. A variant
	// that searches differently overrides this through the map below.
	expectNumAllocateOneInvocations int64

	// expectNumAllocateOneInvocationsByChannel overrides expectNumAllocateOneInvocations with
	// different values for specific implementations (e.g. "experimental").
	//
	// Ignored unless expectNumAllocateOneInvocations is also set.
	// expectNumAllocateOneInvocations should contain the "best" result, so
	// expectNumAllocateOneInvocationsByChannel is only needed as long as we have "worse"
	// implementations.
	expectNumAllocateOneInvocationsByChannel map[internal.AllocatorChannel]int64
}

func requestWithDerivedAttributes(name, class string, count int64, derivedAttrs []resourceapi.DeviceDerivedAttribute, selectors ...resourceapi.DeviceSelector) resourceapi.DeviceRequest {
	req := request(name, class, count, selectors...)
	req.Exactly.DerivedAttributes = derivedAttrs
	return req
}

const sharingAffinityDriver = "sharing.example.com"

func sharingAffinityExtractor(name, expression string) resourceapi.SharingAffinityExtractor {
	return resourceapi.SharingAffinityExtractor{Name: name, APIGroup: "*", Kind: "*", Expression: expression}
}

func sharingAffinityConfig(raw string, requests ...string) resourceapi.DeviceClaimConfiguration {
	return deviceClaimConfig(requests, resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
		Driver: sharingAffinityDriver, Parameters: runtime.RawExtension{Raw: []byte(raw)},
	}})
}

func sharingAffinityAllocConfig(raw string, source resourceapi.AllocationConfigSource, requests ...string) resourceapi.DeviceAllocationConfiguration {
	return resourceapi.DeviceAllocationConfiguration{
		Source: source, Requests: requests, DeviceConfiguration: sharingAffinityConfig(raw).DeviceConfiguration,
	}
}

func sharingAffinityResult(request, device string) resourceapi.DeviceRequestAllocationResult {
	return resourceapi.DeviceRequestAllocationResult{
		Request: request, Driver: sharingAffinityDriver, Pool: "pool", Device: device, ShareID: &fixedShareID,
	}
}

// Each case owns its fixtures and its expected result, including opaque config
// bytes. The common runner checks both full results and input immutability.
func sharingAffinityCase(mutate func(*AllocatorTestCase)) AllocatorTestCase {
	claim := claimWithRequests("claim", nil, request("req", "network", 1)).
		withConfigs([]resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(`{"tenant":"blue"}`)})
	claim.Namespace = "default"
	claim.UID = "claim-uid"
	pool := resourceapi.ResourcePool{Name: "pool", Generation: 1, ResourceSliceCount: 2}
	metadata := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: "metadata"},
		Spec: resourceapi.ResourceSliceSpec{
			Driver: sharingAffinityDriver, Pool: pool,
			SharingAffinity: []resourceapi.SharingAffinityExtractor{sharingAffinityExtractor("tenant", `parameters.?tenant.orValue("")`)},
		},
	}
	tc := AllocatorTestCase{
		features:                 Features{SharingAffinity: true, ConsumableCapacity: true, AdminAccess: true, PrioritizedList: true},
		claimsToAllocate:         objects(claim),
		classes:                  objects(class("network", sharingAffinityDriver)),
		slices:                   []*resourceapi.ResourceSlice{metadata, sliceWithDevices("devices", nodeSelectionAll, pool, sharingAffinityDriver, device("device-a").withAllowMultipleAllocations()).obj()},
		allocatedSharedDeviceIDs: sets.New[DeviceID](),
		allocatedCapacityDevices: internal.NewConsumedCapacityCollection(),
		affinityStates:           AffinityStateCollection{},
		expectResults: []any{allocationResultWithConfigs(nil,
			[]resourceapi.DeviceRequestAllocationResult{sharingAffinityResult("req", "device-a")},
			[]resourceapi.DeviceAllocationConfiguration{sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim)})},
	}
	if mutate != nil {
		mutate(&tc)
	}
	return tc
}

func sharingAffinityUnconstrained(tc *AllocatorTestCase) {
	tc.slices = tc.slices[1:2]
	tc.slices[0].Spec.Pool.ResourceSliceCount = 1
}

func sharingAffinityLockDevice(tc *AllocatorTestCase, device string, values map[string]string) {
	id := MakeDeviceID(sharingAffinityDriver, "pool", device)
	tc.allocatedSharedDeviceIDs.Insert(id)
	tc.affinityStates[id] = DeviceAffinityState{Status: internal.AffinityStatusLocked, LockedAffinity: values}
}

func sharingAffinityExpect(tc *AllocatorTestCase, result resourceapi.DeviceRequestAllocationResult, configs ...resourceapi.DeviceAllocationConfiguration) {
	tc.expectResults = []any{allocationResultWithConfigs(nil, []resourceapi.DeviceRequestAllocationResult{result}, configs)}
}

func sharingAffinityCandidates(testcases map[string]AllocatorTestCase) {
	for _, candidate := range []struct {
		name   string
		mutate func(*AllocatorTestCase)
		reject bool
	}{
		{name: "viable-clean"},
		{name: "viable-equal-lock", mutate: func(tc *AllocatorTestCase) {
			sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "blue"})
		}},
		{name: "reject-mismatch-lock", reject: true, mutate: func(tc *AllocatorTestCase) {
			sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "red"})
		}},
		{name: "reject-lock-superset", reject: true, mutate: func(tc *AllocatorTestCase) {
			sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "blue", "extra": "value"})
		}},
		{name: "reject-lock-subset", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.slices[0].Spec.SharingAffinity = append(tc.slices[0].Spec.SharingAffinity, sharingAffinityExtractor("zone", `"west"`))
			sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "blue"})
		}},
		{name: "reject-partial-satisfaction", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.slices[0].Spec.SharingAffinity = append(tc.slices[0].Spec.SharingAffinity, sharingAffinityExtractor("zone", `""`))
		}},
		{name: "reject-no-config", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.claimsToAllocate[0].Spec.Devices.Config = nil
		}},
		{name: "reject-empty-values", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.claimsToAllocate[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(`{"tenant":""}`)}
		}},
		{name: "reject-unreconstructable", reject: true, mutate: func(tc *AllocatorTestCase) {
			id := MakeDeviceID(sharingAffinityDriver, "pool", "device-a")
			tc.affinityStates[id] = DeviceAffinityState{Status: internal.AffinityStatusUnreconstructable}
			tc.allocatedSharedDeviceIDs.Insert(id)
		}},
		{name: "reject-unreconstructable-unconstrained", reject: true, mutate: func(tc *AllocatorTestCase) {
			sharingAffinityUnconstrained(tc)
			id := MakeDeviceID(sharingAffinityDriver, "pool", "device-a")
			tc.affinityStates[id] = DeviceAffinityState{Status: internal.AffinityStatusUnreconstructable}
			tc.allocatedSharedDeviceIDs.Insert(id)
		}},
		{name: "permissive-pool-no-config", mutate: func(tc *AllocatorTestCase) {
			sharingAffinityUnconstrained(tc)
			tc.claimsToAllocate[0].Spec.Devices.Config = nil
			sharingAffinityExpect(tc, sharingAffinityResult("req", "device-a"))
		}},
		{name: "permissive-pool-no-decoding", mutate: func(tc *AllocatorTestCase) {
			sharingAffinityUnconstrained(tc)
			tc.claimsToAllocate[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(`{`)}
			sharingAffinityExpect(tc, sharingAffinityResult("req", "device-a"), sharingAffinityAllocConfig(`{`, resourceapi.AllocationConfigSourceClaim))
		}},
		{name: "first-fit-clean-before-locked", mutate: func(tc *AllocatorTestCase) {
			tc.slices[1].Spec.Devices = append(tc.slices[1].Spec.Devices, device("device-b").withAllowMultipleAllocations().obj())
			sharingAffinityLockDevice(tc, "device-b", map[string]string{"tenant": "blue"})
		}},
		{name: "first-fit-mismatch-then-clean", mutate: func(tc *AllocatorTestCase) {
			tc.slices[1].Spec.Devices = append(tc.slices[1].Spec.Devices, device("device-b").withAllowMultipleAllocations().obj())
			sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "red"})
			sharingAffinityExpect(tc, sharingAffinityResult("req", "device-b"), sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim))
		}},
		{name: "dedicated-unsatisfied", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.slices[1].Spec.Devices[0].AllowMultipleAllocations = nil
			tc.claimsToAllocate[0].Spec.Devices.Config = nil
		}},
		{name: "dedicated-satisfied", mutate: func(tc *AllocatorTestCase) {
			tc.slices[1].Spec.Devices[0].AllowMultipleAllocations = nil
			result := sharingAffinityResult("req", "device-a")
			result.ShareID = nil
			sharingAffinityExpect(tc, result, sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim))
		}},
		{name: "admin-unsatisfied", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			tc.claimsToAllocate[0].Spec.Devices.Config = nil
		}},
		{name: "admin-mismatch", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "red"})
		}},
		{name: "admin-unreconstructable", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			tc.affinityStates[MakeDeviceID(sharingAffinityDriver, "pool", "device-a")] = DeviceAffinityState{Status: internal.AffinityStatusUnreconstructable}
		}},
		{name: "admin-unconstrained-fenced", reject: true, mutate: func(tc *AllocatorTestCase) {
			sharingAffinityUnconstrained(tc)
			tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			tc.affinityStates[MakeDeviceID(sharingAffinityDriver, "pool", "device-a")] = DeviceAffinityState{Status: internal.AffinityStatusUnreconstructable}
		}},
		{name: "dedicated-unconstrained-fenced", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.slices[1].Spec.Devices[0].AllowMultipleAllocations = nil
			sharingAffinityUnconstrained(tc)
			tc.affinityStates[MakeDeviceID(sharingAffinityDriver, "pool", "device-a")] = DeviceAffinityState{Status: internal.AffinityStatusUnreconstructable}
		}},
		{name: "admin-satisfied", mutate: func(tc *AllocatorTestCase) {
			tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly.AdminAccess = ptr.To(true)
			result := sharingAffinityResult("req", "device-a")
			result.AdminAccess = ptr.To(true)
			sharingAffinityExpect(tc, result, sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim))
		}},
	} {
		tc := sharingAffinityCase(candidate.mutate)
		if candidate.reject {
			tc.expectResults = nil
		}
		testcases["sharing-affinity-candidates-"+candidate.name] = tc
	}
	for name, consumed := range map[string]string{"insufficient": "3", "sufficient": "2"} {
		tc := sharingAffinityCase(func(tc *AllocatorTestCase) {
			sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "blue"})
			tc.slices[1].Spec.Devices[0].Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{"slots": {Value: resource.MustParse("4")}}
			tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{Requests: map[resourceapi.QualifiedName]resource.Quantity{"slots": resource.MustParse("2")}}
			tc.allocatedCapacityDevices.Insert(schedulerapi.NewDeviceConsumedCapacity(MakeDeviceID(sharingAffinityDriver, "pool", "device-a"), map[resourceapi.QualifiedName]resource.Quantity{"slots": resource.MustParse(consumed)}))
		})
		tc.expectResults = nil
		if name == "sufficient" {
			result := sharingAffinityResult("req", "device-a")
			result.ConsumedCapacity = map[resourceapi.QualifiedName]resource.Quantity{"slots": resource.MustParse("2")}
			sharingAffinityExpect(&tc, result, sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim))
		}
		testcases["sharing-affinity-candidates-capacity-"+name] = tc
	}
}

func sharingAffinityAddMetadata(tc *AllocatorTestCase, entries ...resourceapi.SharingAffinityExtractor) {
	metadata := tc.slices[0].DeepCopy()
	metadata.Name = fmt.Sprintf("metadata-%d", len(tc.slices))
	metadata.Spec.SharingAffinity = entries
	tc.slices = append(tc.slices, metadata)
	for _, s := range tc.slices {
		s.Spec.Pool.ResourceSliceCount = int64(len(tc.slices))
	}
}

func sharingAffinityPools(testcases map[string]AllocatorTestCase) {
	for _, pool := range []struct {
		name   string
		mutate func(*AllocatorTestCase)
		reject bool
	}{
		{name: "multiple-metadata-slices-union-partial", reject: true, mutate: func(tc *AllocatorTestCase) {
			sharingAffinityAddMetadata(tc, sharingAffinityExtractor("zone", `parameters.?zone.orValue("")`))
		}},
		{name: "multiple-metadata-slices-union-complete", mutate: func(tc *AllocatorTestCase) {
			sharingAffinityAddMetadata(tc, sharingAffinityExtractor("zone", `parameters.?zone.orValue("")`))
			tc.claimsToAllocate[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(`{"tenant":"blue","zone":"west"}`)}
			sharingAffinityExpect(tc, sharingAffinityResult("req", "device-a"), sharingAffinityAllocConfig(`{"tenant":"blue","zone":"west"}`, resourceapi.AllocationConfigSourceClaim))
		}},
		{name: "generation-isolation-new-complete", mutate: func(tc *AllocatorTestCase) {
			old := tc.slices[0].DeepCopy()
			old.Name = "old"
			old.Spec.SharingAffinity[0].Expression = `parameters.missing`
			for _, s := range tc.slices {
				s.Spec.Pool.Generation = 2
			}
			tc.slices = append(tc.slices, old)
		}},
		{name: "generation-isolation-new-incomplete", reject: true, mutate: func(tc *AllocatorTestCase) {
			newer := tc.slices[0].DeepCopy()
			newer.Name, newer.Spec.Pool.Generation = "new", 2
			tc.slices = append(tc.slices, newer)
		}},
		{name: "driver-isolation", mutate: func(tc *AllocatorTestCase) {
			other := tc.slices[0].DeepCopy()
			other.Name, other.Spec.Driver, other.Spec.Pool.ResourceSliceCount = "other-driver", "other.example.com", 1
			other.Spec.SharingAffinity[0].Expression = `parameters.missing`
			tc.slices = append(tc.slices, other)
		}},
		{name: "incomplete-metadata", reject: true, mutate: func(tc *AllocatorTestCase) { tc.slices = tc.slices[1:] }},
		{name: "inconsistent-slice-count", reject: true, mutate: func(tc *AllocatorTestCase) { tc.slices[0].Spec.Pool.ResourceSliceCount = 3 }},
		{name: "duplicate-slice-name", reject: true, mutate: func(tc *AllocatorTestCase) { tc.slices[0].Name = tc.slices[1].Name }},
		{name: "duplicate-names-across-slices", reject: true, mutate: func(tc *AllocatorTestCase) {
			tc.slices[0].Spec.SharingAffinity[0].Expression = `string(1 / parameters.zero)`
			sharingAffinityAddMetadata(tc, sharingAffinityExtractor("tenant", `parameters.missing`))
			tc.claimsToAllocate[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(`{"zero":0}`)}
			// Duplicate keys invalidate the pool before either runtime error
			// can occur, even though both expressions compile.
			tc.expectErrorIs = internal.ErrFailedAllocationOnNode
			tc.expectError = gomega.And(gomega.HaveOccurred(),
				gomega.Not(gomega.MatchError(sharingaffinity.ErrCELEval)),
				gomega.Not(gomega.MatchError(sharingaffinity.ErrMissingField)))
		}},
		{name: "sixteen-entries-across-two-slices", mutate: func(tc *AllocatorTestCase) {
			tc.slices[0].Spec.SharingAffinity = nil
			var second []resourceapi.SharingAffinityExtractor
			for i := 0; i < 8; i++ {
				tc.slices[0].Spec.SharingAffinity = append(tc.slices[0].Spec.SharingAffinity, sharingAffinityExtractor(fmt.Sprintf("first-%d", i), `"blue"`))
				second = append(second, sharingAffinityExtractor(fmt.Sprintf("second-%d", i), `"west"`))
			}
			sharingAffinityAddMetadata(tc, second...)
		}},
		{name: "invalid-pool-does-not-poison-other-pools", mutate: func(tc *AllocatorTestCase) {
			sharingAffinityAddMetadata(tc, sharingAffinityExtractor("tenant", `parameters.missing`))
			alternative := tc.slices[1].DeepCopy()
			alternative.Name = "alternative-devices"
			alternative.Spec.Pool = resourceapi.ResourcePool{Name: "zz-alternative", Generation: 1, ResourceSliceCount: 1}
			tc.slices = append(tc.slices, alternative)
			result := sharingAffinityResult("req", "device-a")
			result.Pool = "zz-alternative"
			sharingAffinityExpect(tc, result, sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim))
		}},
	} {
		tc := sharingAffinityCase(pool.mutate)
		if pool.reject {
			tc.expectResults = nil
		}
		testcases["sharing-affinity-pool-"+pool.name] = tc
	}
}

func sharingAffinityGateAndState(testcases map[string]AllocatorTestCase) {
	for _, state := range []struct {
		name   string
		status internal.AffinityStatus
	}{
		{"clean", internal.AffinityStatusClean},
		{"conflicting-lock", internal.AffinityStatusLocked},
		{"unreconstructable", internal.AffinityStatusUnreconstructable},
	} {
		for _, enabled := range []bool{false, true} {
			tc := sharingAffinityCase(func(tc *AllocatorTestCase) {
				tc.features = Features{ConsumableCapacity: true, SharingAffinity: enabled}
				tc.slices[0].Spec.AllNodes = ptr.To(true)
				tc.affinityStates = nil
				tc.allocatedSharedDeviceIDs = nil
				if state.status != internal.AffinityStatusClean {
					device := MakeDeviceID(sharingAffinityDriver, "pool", "device-a")
					lock := DeviceAffinityState{Status: state.status}
					if state.status == internal.AffinityStatusLocked {
						lock.LockedAffinity = map[string]string{"tenant": "red"}
					}
					tc.affinityStates = AffinityStateCollection{device: lock}
					tc.allocatedSharedDeviceIDs = sets.New(device)
				}
			})
			if !enabled || state.status != internal.AffinityStatusClean {
				tc.expectResults = nil
			}
			testcases[fmt.Sprintf("sharing-affinity-gate-and-state-%s-enabled=%t", state.name, enabled)] = tc
		}
	}
}

func sharingAffinityGateOff(testcases map[string]AllocatorTestCase) {
	for name, mode := range map[string]resourceapi.DeviceAllocationMode{
		"exactcount": resourceapi.DeviceAllocationModeExactCount,
		"all":        resourceapi.DeviceAllocationModeAll,
	} {
		for _, scope := range []string{"all-nodes", "no-node", "other-node", "per-device-node", "unconstrained", "newest-declares", "old-declares"} {
			tc := sharingAffinityCase(func(tc *AllocatorTestCase) {
				tc.features = Features{}
				tc.slices[1].Spec.Devices[0].AllowMultipleAllocations = nil
				req := tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly
				req.AllocationMode = mode
				if mode == resourceapi.DeviceAllocationModeAll {
					req.Count = 0
				}
				// Disabled affinity must never evaluate even invalid CEL.
				tc.slices[0].Spec.SharingAffinity[0].Expression = `parameters.[`
				switch scope {
				case "all-nodes":
					tc.slices[0].Spec.AllNodes = ptr.To(true)
				case "other-node":
					tc.slices[0].Spec.NodeName = ptr.To("elsewhere")
				case "per-device-node":
					tc.slices[0].Spec.PerDeviceNodeSelection = ptr.To(true)
				case "unconstrained":
					sharingAffinityUnconstrained(tc)
				case "newest-declares":
					old := tc.slices[1].DeepCopy()
					old.Name, old.Spec.Pool.ResourceSliceCount = "old-devices", 1
					for _, s := range tc.slices {
						s.Spec.Pool.Generation = 2
					}
					tc.slices = append(tc.slices, old)
				case "old-declares":
					tc.slices[1].Spec.Pool.Generation = 2
					tc.slices[1].Spec.Pool.ResourceSliceCount = 1
				}
			})
			tc.expectResults = nil
			if scope == "unconstrained" || scope == "old-declares" {
				result := sharingAffinityResult("req", "device-a")
				result.ShareID = nil
				sharingAffinityExpect(&tc, result, sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim))
			}
			testcases["sharing-affinity-gate-off-"+name+"-"+scope] = tc
		}
	}
	testcases["sharing-affinity-gate-off-consumable-capacity-enabled"] = sharingAffinityCase(func(tc *AllocatorTestCase) {
		tc.features = Features{ConsumableCapacity: true}
		tc.expectResults = nil
	})
	all := resourceapi.DeviceSubRequest{Name: "all", DeviceClassName: "network", AllocationMode: resourceapi.DeviceAllocationModeAll}
	exact := resourceapi.DeviceSubRequest{Name: "exact", DeviceClassName: "network", AllocationMode: resourceapi.DeviceAllocationModeExactCount, Count: 1}
	for _, scenario := range []struct {
		name        string
		requests    []resourceapi.DeviceSubRequest
		otherNode   bool
		wantRequest string
	}{
		{name: "fallback", requests: []resourceapi.DeviceSubRequest{all, exact}, wantRequest: "req/exact"},
		{name: "all-only", requests: []resourceapi.DeviceSubRequest{all}},
		{name: "exact-first", requests: []resourceapi.DeviceSubRequest{exact, all}, wantRequest: "req/exact"},
		{name: "unrelated-node", requests: []resourceapi.DeviceSubRequest{all, exact}, otherNode: true, wantRequest: "req/all"},
	} {
		testcases["sharing-affinity-gate-off-first-available-"+scenario.name] = sharingAffinityCase(func(tc *AllocatorTestCase) {
			tc.features = Features{PrioritizedList: true}
			tc.slices[1].Spec.Devices[0].AllowMultipleAllocations = nil
			available := tc.slices[1].DeepCopy()
			available.Name = "available"
			available.Spec.Pool = resourceapi.ResourcePool{Name: "unconstrained", Generation: 1, ResourceSliceCount: 1}
			tc.slices[0].Spec.SharingAffinity[0].Expression = `parameters.[`
			if scenario.otherNode {
				for _, slice := range tc.slices {
					slice.Spec.AllNodes = nil
					slice.Spec.NodeName = ptr.To("elsewhere")
				}
			}
			tc.slices = append(tc.slices, available)
			tc.claimsToAllocate[0].Spec.Devices.Requests = []resourceapi.DeviceRequest{{
				Name: "req", FirstAvailable: scenario.requests,
			}}
			tc.expectResults = nil
			if scenario.wantRequest != "" {
				result := sharingAffinityResult(scenario.wantRequest, "device-a")
				result.Pool, result.ShareID = "unconstrained", nil
				sharingAffinityExpect(tc, result, sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim))
			}
		})
	}
	testcases["sharing-affinity-gate-off-all-invalid-node-selector"] = sharingAffinityCase(func(tc *AllocatorTestCase) {
		tc.features = Features{}
		tc.slices[0].Spec.NodeSelector = &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{
			MatchExpressions: []v1.NodeSelectorRequirement{{Key: "zone", Operator: "invalid"}},
		}}}
		request := tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly
		request.AllocationMode, request.Count = resourceapi.DeviceAllocationModeAll, 0
		tc.expectResults = nil
		tc.expectError = gomega.MatchError(gomega.ContainSubstring("failed to parse node selector"))
	})
	for _, obsolete := range []bool{true, false} {
		testcases[fmt.Sprintf("sharing-affinity-gate-off-obsolete-placement-obsolete-%t", obsolete)] = sharingAffinityCase(func(tc *AllocatorTestCase) {
			tc.features = Features{}
			sharingAffinityUnconstrained(tc)
			tc.slices[0].Spec.AllNodes = nil
			tc.slices[0].Spec.NodeName = ptr.To(node1)
			tc.slices[0].Spec.Devices[0].AllowMultipleAllocations = nil
			req := tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly
			req.AllocationMode, req.Count = resourceapi.DeviceAllocationModeAll, 0
			tc.slices = append(tc.slices, sliceWithDevices("blocked-new", "elsewhere",
				resourceapi.ResourcePool{Name: "blocked", Generation: 2, ResourceSliceCount: 2},
				sharingAffinityDriver, device("device-new")).obj())
			if obsolete {
				old := sliceWithNoDevices("blocked-old", node1,
					resourceapi.ResourcePool{Name: "blocked", Generation: 1, ResourceSliceCount: 1}, sharingAffinityDriver).obj()
				old.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{sharingAffinityExtractor("tenant", `"x"`)}
				tc.slices = append(tc.slices, old)
			}
			result := sharingAffinityResult("req", "device-a")
			result.ShareID = nil
			tc.expectResults = []any{allocationResultWithConfigs(localNodeSelector(node1),
				[]resourceapi.DeviceRequestAllocationResult{result},
				[]resourceapi.DeviceAllocationConfiguration{sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim)})}
		})
	}
}

func sharingAffinityObsoleteGeneration(testcases map[string]AllocatorTestCase) {
	// Only the newest generation's node placement determines whether another
	// pool must be complete before an All request can be allocated.
	for _, newest := range []string{"none", "complete", "incomplete"} {
		testcases["sharing-affinity-obsolete-generation-"+newest] = sharingAffinityCase(func(tc *AllocatorTestCase) {
			sharingAffinityUnconstrained(tc)
			req := tc.claimsToAllocate[0].Spec.Devices.Requests[0].Exactly
			req.AllocationMode, req.Count = resourceapi.DeviceAllocationModeAll, 0
			other := func(name, node string, generation, count int64) *resourceapi.ResourceSlice {
				return sliceWithDevices(name, node,
					resourceapi.ResourcePool{Name: "other", Generation: generation, ResourceSliceCount: count},
					"other.example.com", device(name+"-device")).obj()
			}
			if newest != "none" {
				tc.slices = append(tc.slices, other("old", node1, 1, 1))
			}
			switch newest {
			case "complete":
				tc.slices = append(tc.slices, other("new", "elsewhere", 2, 1))
			case "incomplete":
				tc.slices = append(tc.slices, other("new", "elsewhere", 2, 2))
			}
		})
	}
}

func sharingAffinityMetadata(testcases map[string]AllocatorTestCase) {
	for _, metadata := range []struct {
		name, group, kind, raw string
		want                   bool
	}{
		{"exact", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":"Config","tenant":"blue"}`, true},
		{"version-independent", "example.com", "Config", `{"apiVersion":"example.com/v99alpha1","kind":"Config","tenant":"blue"}`, true},
		{"group-mismatch", "example.com", "Config", `{"apiVersion":"other.example.com/v1","kind":"Config","tenant":"blue"}`, false},
		{"kind-mismatch", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":"Other","tenant":"blue"}`, false},
		{"kind-case-sensitive", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":"config","tenant":"blue"}`, false},
		{"wildcard", "*", "*", `{"tenant":"blue"}`, true},
		{"wildcard-group", "*", "Config", `{"kind":"Config","tenant":"blue"}`, true},
		{"wildcard-kind", "example.com", "*", `{"apiVersion":"example.com/v1","tenant":"blue"}`, true},
		{"core-group", "*", "Config", `{"apiVersion":"v1","kind":"Config","tenant":"blue"}`, true},
		{"core-group-exact-filter", "example.com", "Config", `{"apiVersion":"v1","kind":"Config","tenant":"blue"}`, false},
		{"missing-api-version", "example.com", "Config", `{"kind":"Config","tenant":"blue"}`, false},
		{"missing-kind", "*", "Config", `{"tenant":"blue"}`, false},
		{"extra-fields-ignored", "*", "*", `{"tenant":"blue","private":{"password":"different"},"zone":"ignored"}`, true},
	} {
		tc := sharingAffinityCase(func(tc *AllocatorTestCase) {
			tc.slices[0].Spec.SharingAffinity[0].APIGroup = metadata.group
			tc.slices[0].Spec.SharingAffinity[0].Kind = metadata.kind
			tc.claimsToAllocate[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(metadata.raw)}
			sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "blue"})
		})
		tc.expectResults = nil
		if metadata.want {
			sharingAffinityExpect(&tc, sharingAffinityResult("req", "device-a"), sharingAffinityAllocConfig(metadata.raw, resourceapi.AllocationConfigSourceClaim))
		}
		testcases["sharing-affinity-metadata-"+metadata.name] = tc
	}
}

func sharingAffinityScope(testcases map[string]AllocatorTestCase) {
	for _, scope := range []string{"other-driver-before-decode", "other-request-before-decode", "device-class-config-excluded", "device-class-cannot-satisfy", "other-claim-cannot-satisfy", "subrequest-scope", "parent-scope"} {
		testcases["sharing-affinity-scope-"+scope] = sharingAffinityCase(func(tc *AllocatorTestCase) {
			switch scope {
			case "other-driver-before-decode":
				config := sharingAffinityConfig(`{`)
				config.Opaque.Driver = "other.example.com"
				tc.claimsToAllocate[0].Spec.Devices.Config = append(tc.claimsToAllocate[0].Spec.Devices.Config, config)
				expected := sharingAffinityAllocConfig(`{`, resourceapi.AllocationConfigSourceClaim)
				expected.Opaque.Driver = "other.example.com"
				sharingAffinityExpect(tc, sharingAffinityResult("req", "device-a"),
					sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim), expected)
			case "other-request-before-decode":
				tc.claimsToAllocate[0].Spec.Devices.Config = append(tc.claimsToAllocate[0].Spec.Devices.Config, sharingAffinityConfig(`{`, "unselected"))
			case "device-class-config-excluded":
				tc.classes[0].Spec.Config = []resourceapi.DeviceClassConfiguration{{DeviceConfiguration: sharingAffinityConfig(`{"tenant":"red"}`).DeviceConfiguration}}
				sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "blue"})
				sharingAffinityExpect(tc, sharingAffinityResult("req", "device-a"),
					sharingAffinityAllocConfig(`{"tenant":"red"}`, resourceapi.AllocationConfigSourceClass),
					sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim))
			case "device-class-cannot-satisfy":
				tc.classes[0].Spec.Config = []resourceapi.DeviceClassConfiguration{{DeviceConfiguration: sharingAffinityConfig(`{"tenant":"blue"}`).DeviceConfiguration}}
				tc.claimsToAllocate[0].Spec.Devices.Config = nil
				tc.expectResults = nil
			case "other-claim-cannot-satisfy":
				second := wrapResourceClaim{tc.claimsToAllocate[0].DeepCopy()}
				second.Name, second.UID, second.Spec.Devices.Config = "second", "second-uid", nil
				tc.claimsToAllocate = append(tc.claimsToAllocate, second)
				tc.expectResults = nil
			case "subrequest-scope", "parent-scope":
				tc.claimsToAllocate[0] = tc.claimsToAllocate[0].withRequests(requestWithPrioritizedList("req",
					subRequest("first", "network", 1), subRequest("second", "network", 1)))
				sharingAffinityLockDevice(tc, "device-a", map[string]string{"tenant": "blue"})
				selected := "req/first"
				if scope == "subrequest-scope" {
					tc.claimsToAllocate[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{
						sharingAffinityConfig(`{"tenant":"red"}`, "req/first"),
						sharingAffinityConfig(`{"tenant":"blue"}`, "req/second"),
					}
					selected = "req/second"
				} else {
					tc.claimsToAllocate[0].Spec.Devices.Config[0].Requests = []string{"req"}
				}
				sharingAffinityExpect(tc, sharingAffinityResult(selected, "device-a"), sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim))
			}
		})
	}
	testcases["sharing-affinity-backtracking-rollback-local-lock"] = sharingAffinityCase(func(tc *AllocatorTestCase) {
		tc.slices[1].Spec.Devices = []resourceapi.Device{
			device("device-a").withAllowMultipleAllocations().withAttribute("ordinal", int64(0)).obj(),
			device("device-b").withAllowMultipleAllocations().withAttribute("ordinal", int64(1)).obj(),
		}
		second := request("second", "network", 1, resourceapi.DeviceSelector{
			CEL: &resourceapi.CELDeviceSelector{Expression: `device.attributes["sharing.example.com"].ordinal == 0`},
		})
		tc.claimsToAllocate[0].Spec.Devices.Requests = append(tc.claimsToAllocate[0].Spec.Devices.Requests, second)
		tc.claimsToAllocate[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{
			sharingAffinityConfig(`{"tenant":"blue"}`, "req"), sharingAffinityConfig(`{"tenant":"red"}`, "second"),
		}
		tc.expectResults = []any{allocationResultWithConfigs(nil,
			[]resourceapi.DeviceRequestAllocationResult{sharingAffinityResult("req", "device-b"), sharingAffinityResult("second", "device-a")},
			[]resourceapi.DeviceAllocationConfiguration{
				sharingAffinityAllocConfig(`{"tenant":"blue"}`, resourceapi.AllocationConfigSourceClaim, "req"),
				sharingAffinityAllocConfig(`{"tenant":"red"}`, resourceapi.AllocationConfigSourceClaim, "second"),
			})}
	})
}

type allocationAffinityFixture struct {
	newAllocator allocatorFactory
	features     Features
	state        AllocatedState
	class        *resourceapi.DeviceClass
	slices       []*resourceapi.ResourceSlice
	claims       []*resourceapi.ResourceClaim
	node         *v1.Node
	cache        *cel.Cache
}

func newAllocationAffinityFixture(newAllocator allocatorFactory) *allocationAffinityFixture {
	tc := sharingAffinityCase(nil)
	return &allocationAffinityFixture{
		newAllocator: newAllocator,
		features:     tc.features,
		state: AllocatedState{
			AllocatedDevices: sets.New[DeviceID](), AllocatedSharedDeviceIDs: tc.allocatedSharedDeviceIDs,
			AggregatedCapacity:    tc.allocatedCapacityDevices,
			DeviceSharingAffinity: SharingAffinityState{DeviceStates: tc.affinityStates},
		},
		class:  tc.classes[0],
		slices: tc.slices,
		claims: unwrap(tc.claimsToAllocate...),
		node:   &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}},
		cache:  cel.NewCache(32, cel.Features{}),
	}
}

func allocationAffinitySelector(expression string) resourceapi.DeviceSelector {
	return resourceapi.DeviceSelector{CEL: &resourceapi.CELDeviceSelector{Expression: expression}}
}

func allocationAffinityRequest(name string) resourceapi.DeviceRequest {
	return request(name, "network", 1)
}

func (f *allocationAffinityFixture) addMetadata(entries ...resourceapi.SharingAffinityExtractor) {
	slice := f.slices[0].DeepCopy()
	slice.Name = fmt.Sprintf("metadata-%d", len(f.slices))
	slice.Spec.SharingAffinity = entries
	f.slices = append(f.slices, slice)
	for _, s := range f.slices {
		s.Spec.Pool.ResourceSliceCount = int64(len(f.slices))
	}
}

func (f *allocationAffinityFixture) unconstrained() {
	f.slices = f.slices[1:2]
	f.slices[0].Spec.Pool.ResourceSliceCount = 1
}

func (f *allocationAffinityFixture) lock(device string, values map[string]string) {
	id := MakeDeviceID(sharingAffinityDriver, "pool", device)
	f.state.AllocatedSharedDeviceIDs.Insert(id)
	f.state.DeviceSharingAffinity.DeviceStates[id] = DeviceAffinityState{Status: internal.AffinityStatusLocked, LockedAffinity: values}
}

func (f *allocationAffinityFixture) allocator(t *testing.T, ctx context.Context) Allocator {
	t.Helper()
	a, err := f.newAllocator(ctx, f.features, f.state, informerLister[resourceapi.DeviceClass]{objs: []*resourceapi.DeviceClass{f.class}}, f.slices, f.cache)
	if err != nil {
		t.Fatalf("NewAllocator: %v", err)
	}
	return a
}

func assertAffinityAllocation(t *testing.T, got []resourceapi.AllocationResult, err error, devices ...string) {
	t.Helper()
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	var actual []string
	for _, claim := range got {
		for _, device := range claim.Devices.Results {
			if device.Driver != sharingAffinityDriver || device.Pool != "pool" {
				t.Errorf("allocated unexpected driver/pool: %s/%s", device.Driver, device.Pool)
			}
			actual = append(actual, device.Device)
		}
	}
	if !reflect.DeepEqual(actual, devices) {
		t.Fatalf("allocated devices = %v, want %v", actual, devices)
	}
	if len(devices) == 0 && len(got) != 0 {
		t.Errorf("failed allocation returned %d partial claim results", len(got))
	}
}

func testSharingAffinityAllocation(t *testing.T, newAllocator allocatorFactory) {
	// Ordinary allocation cases run through the common TestAllocator matrix.
	// These cases need provenance, multi-call state, or detailed errors.
	for _, tc := range []struct {
		name string
		run  func(*testing.T, allocatorFactory)
	}{
		{"pool", testSharingAffinityPools},
		{"fatal", testSharingAffinityFatal},
		{"provenance", testSharingAffinityProvenance},
		{"isolation", testSharingAffinityIsolation},
		{"backtracking", testSharingAffinityBacktracking},
		{"concurrent", testSharingAffinityConcurrent},
		{"budget", testSharingAffinityBudget},
		{"freshness", testSharingAffinityFreshness},
	} {
		t.Run(tc.name, func(t *testing.T) { tc.run(t, newAllocator) })
	}
	t.Run("cancellation", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		_, ctx := ktesting.NewTestContext(t)
		a := f.allocator(t, ctx)
		ctx, cancel := context.WithCancel(ctx)
		cancel()
		got, err := a.Allocate(ctx, f.node, f.claims)
		if !errors.Is(err, context.Canceled) || len(got) != 0 {
			t.Fatalf("cancelled allocation = %v, %v; want no results and context.Canceled", got, err)
		}
	})
}

func testSharingAffinityGateOffEntryPoints(t *testing.T, newAllocator allocatorFactory) {
	for _, entry := range []struct {
		name       string
		provenance bool
	}{
		{name: "allocate"},
		{name: "provenance", provenance: true},
	} {
		for _, mode := range []resourceapi.DeviceAllocationMode{resourceapi.DeviceAllocationModeExactCount, resourceapi.DeviceAllocationModeAll} {
			for _, canceled := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/canceled-%t", entry.name, mode, canceled), func(t *testing.T) {
					f := newAllocationAffinityFixture(newAllocator)
					f.features = Features{}
					f.slices[1].Spec.Devices[0].AllowMultipleAllocations = nil
					request := f.claims[0].Spec.Devices.Requests[0].Exactly
					request.AllocationMode = mode
					if mode == resourceapi.DeviceAllocationModeAll {
						request.Count = 0
					}
					_, ctx := ktesting.NewTestContext(t)
					allocator := f.allocator(t, ctx)
					withAffinity, hasProvenance := allocator.(affinityAllocator)
					if entry.provenance && !hasProvenance {
						t.Skipf("%s does not expose the provenance entry point", allocator.Channel())
					}
					var wantErr error
					if canceled {
						var cancel context.CancelFunc
						ctx, cancel = context.WithCancel(ctx)
						cancel()
						wantErr = context.Canceled
					}
					var results []resourceapi.AllocationResult
					var provenance []internal.ClaimAffinity
					var err error
					if entry.provenance {
						results, provenance, err = withAffinity.AllocateWithAffinity(ctx, f.node, f.claims)
					} else {
						results, err = allocator.Allocate(ctx, f.node, f.claims)
					}
					if !errors.Is(err, wantErr) || len(results) != 0 || len(provenance) != 0 {
						t.Fatalf("allocation = %v, provenance = %v, error = %v; want no results or provenance and error %v", results, provenance, err, wantErr)
					}
				})
			}
		}
	}
}

func testSharingAffinityProvenance(t *testing.T, newAllocator allocatorFactory) {
	t.Run("result-index-and-ownership", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.slices[1].Spec.Devices = append(f.slices[1].Spec.Devices, resourceapi.Device{Name: "device-b"})
		f.slices[1].Spec.Devices[0].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"dedicated": {BoolValue: ptr.To(false)}}
		f.slices[1].Spec.Devices[1].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"dedicated": {BoolValue: ptr.To(true)}}
		f.claims[0].Spec.Devices.Requests = append(f.claims[0].Spec.Devices.Requests, allocationAffinityRequest("admin"), allocationAffinityRequest("dedicated"))
		f.claims[0].Spec.Devices.Requests[1].Exactly.AdminAccess = ptr.To(true)
		f.claims[0].Spec.Devices.Requests[2].Exactly.Selectors = []resourceapi.DeviceSelector{allocationAffinitySelector(`device.attributes["sharing.example.com"].dedicated`)}
		f.claims[0].Spec.Devices.Config = append(f.claims[0].Spec.Devices.Config,
			sharingAffinityConfig(`{"tenant":"blue"}`, "admin"),
			sharingAffinityConfig(`{"tenant":"blue"}`, "dedicated"),
		)
		_, ctx := ktesting.NewTestContext(t)
		allocator := f.allocator(t, ctx).(affinityAllocator)
		results, affinityResults, err := allocator.AllocateWithAffinity(ctx, f.node, f.claims)
		assertAffinityAllocation(t, results, err, "device-a", "device-a", "device-b")
		if len(affinityResults) != 1 || len(affinityResults[0].Devices) != len(results[0].Devices.Results) {
			t.Fatalf("affinity results = %#v, allocation results = %#v; want one provenance entry per result", affinityResults, results)
		}
		digest := internal.ExtractorDigest(f.slices[0].Spec.SharingAffinity)
		for i, prov := range affinityResults[0].Devices {
			result := results[0].Devices.Results[i]
			if prov.ResultIndex != i {
				t.Errorf("ResultIndex[%d] = %d, want %d", i, prov.ResultIndex, i)
			}
			if prov.Request != result.Request || prov.Device != MakeDeviceID(result.Driver, result.Pool, result.Device) {
				t.Errorf("provenance[%d] = request %q device %q, want %q/%s", i, prov.Request, prov.Device, result.Request, result.Device)
			}
			if prov.Pool.Driver != result.Driver || prov.Pool.Name != result.Pool || prov.Pool.Generation != f.slices[0].Spec.Pool.Generation || prov.Digest != digest {
				t.Errorf("pool provenance[%d] = %#v digest %q, want %s/%s gen %d digest %q", i, prov.Pool, prov.Digest, result.Driver, result.Pool, f.slices[0].Spec.Pool.Generation, digest)
			}
			if !prov.Constrained || prov.Values["tenant"] != "blue" {
				t.Errorf("constrained values[%d] = %t/%v, want tenant=blue", i, prov.Constrained, prov.Values)
			}
			wantContributes := result.ShareID != nil && !ptr.Deref(result.AdminAccess, false)
			if prov.Contributes != wantContributes {
				t.Errorf("Contributes[%d] = %t, want %t for result %#v", i, prov.Contributes, wantContributes, result)
			}
		}
	})
	t.Run("unconstrained-and-admin", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.unconstrained()
		f.claims[0].Spec.Devices.Requests = append(f.claims[0].Spec.Devices.Requests, allocationAffinityRequest("admin"))
		f.claims[0].Spec.Devices.Requests[1].Exactly.AdminAccess = ptr.To(true)
		_, ctx := ktesting.NewTestContext(t)
		allocator := f.allocator(t, ctx).(affinityAllocator)
		results, affinityResults, err := allocator.AllocateWithAffinity(ctx, f.node, f.claims)
		assertAffinityAllocation(t, results, err, "device-a", "device-a")
		if len(affinityResults) != 1 || len(affinityResults[0].Devices) != len(results[0].Devices.Results) || len(results[0].Devices.Results) != 2 {
			t.Fatalf("affinity results = %#v, allocation results = %#v; want two provenance entries, one per result", affinityResults, results)
		}
		for i, prov := range affinityResults[0].Devices {
			result := results[0].Devices.Results[i]
			if prov.Request != result.Request || prov.Device != MakeDeviceID(result.Driver, result.Pool, result.Device) ||
				prov.Pool.Driver != result.Driver || prov.Pool.Name != result.Pool {
				t.Errorf("unconstrained provenance[%d] = %#v, want request %q device %s/%s/%s", i, prov, result.Request, result.Driver, result.Pool, result.Device)
			}
			if prov.ResultIndex != i || prov.Pool.Generation != f.slices[0].Spec.Pool.Generation || prov.Digest != internal.EmptyExtractorDigest {
				t.Errorf("unconstrained provenance[%d] = %#v, want matching index/generation empty digest", i, prov)
			}
			if prov.Constrained || prov.Values != nil || prov.Contributes {
				t.Errorf("unconstrained provenance[%d] constrained/values/contributes = %t/%v/%t, want false/nil/false", i, prov.Constrained, prov.Values, prov.Contributes)
			}
		}
	})
}

func testSharingAffinityPools(t *testing.T, newAllocator allocatorFactory) {
	t.Run("aggregate-cost-overflow", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		// A constant-input regex has bounded cost and stays within the CEL
		// expression length limit; two individually legal slices exceed the pool budget.
		var expression string
		for size := 1000; size <= 5000; size += 500 {
			candidate := fmt.Sprintf("'%s'.matches('%s') ? 'blue' : 'blue'", strings.Repeat("a", size), strings.Repeat("a", size))
			result := sharingaffinity.ValidateExpression(candidate, environment.StoredExpressions)
			if result.Error != nil {
				t.Fatalf("compile aggregate-cost fixture: %v", result.Error)
			}
			if result.MaxCost > resourceapi.SharingAffinityCELMaxCost/2 && result.MaxCost <= resourceapi.SharingAffinityCELMaxCost {
				expression = candidate
				break
			}
		}
		if expression == "" {
			t.Fatal("could not construct an individually valid expression exceeding half the aggregate budget")
		}
		entry := sharingAffinityExtractor("tenant", expression)
		if err := sharingaffinity.ValidateExtractorSet([]resourceapi.SharingAffinityExtractor{entry}); err != nil {
			t.Fatalf("individual metadata slice must be valid: %v", err)
		}
		second := entry
		second.Name = "zone"
		if err := sharingaffinity.ValidateExtractorSet([]resourceapi.SharingAffinityExtractor{entry, second}); !errors.Is(err, sharingaffinity.ErrCostExceeded) {
			t.Fatalf("combined fixture must exceed estimated cost: %v", err)
		}
		f.slices[0].Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{entry}
		f.addMetadata(second)
		_, ctx := ktesting.NewTestContext(t)
		got, err := f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
		if !errors.Is(err, internal.ErrFailedAllocationOnNode) || len(got) != 0 {
			t.Fatalf("over-budget pool allocation = %v, %v; want invalid-pool error", got, err)
		}
	})
}

func testSharingAffinityFatal(t *testing.T, newAllocator allocatorFactory) {
	const secret = "s3cr3t-affinity-payload"
	for _, tc := range []struct {
		name, expression, raw string
		wantErr               error
	}{
		{"cel-error", `string(int(parameters.secret))`, `{"secret":"s3cr3t-affinity-payload"}`, sharingaffinity.ErrCELEval},
		{"missing-field", `parameters[parameters.secret]`, `{"secret":"s3cr3t-affinity-payload"}`, sharingaffinity.ErrMissingField},
		{"non-string", `parameters.tenant`, `{"tenant":42,"secret":"s3cr3t-affinity-payload"}`, sharingaffinity.ErrNonStringReturn},
		{"parameters", `parameters.tenant`, `{"tenant":"s3cr3t-affinity-payload"`, sharingaffinity.ErrParameters},
		{"claim-inconsistent", `parameters.tenant`, `{"tenant":"s3cr3t-affinity-payload"}`, sharingaffinity.ErrClaimSelfInconsistent},
		{"sibling-unsatisfied-before-error", `parameters.missing`, `{"tenant":"s3cr3t-affinity-payload"}`, sharingaffinity.ErrMissingField},
		{"cost-exhausted-admissible", "", "", sharingaffinity.ErrCostExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, provenance := range []bool{false, true} {
				t.Run(fmt.Sprintf("provenance-%t", provenance), func(t *testing.T) {
					f := newAllocationAffinityFixture(newAllocator)
					f.slices[0].Spec.SharingAffinity[0].Expression = tc.expression
					f.claims[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(tc.raw)}
					if tc.wantErr == sharingaffinity.ErrClaimSelfInconsistent {
						f.claims[0].Spec.Devices.Config = append(f.claims[0].Spec.Devices.Config, sharingAffinityConfig(`{"tenant":"other-value"}`))
					}
					if tc.name == "sibling-unsatisfied-before-error" {
						f.slices[0].Spec.SharingAffinity = append([]resourceapi.SharingAffinityExtractor{sharingAffinityExtractor("absent", `parameters.?absent.orValue("")`)}, f.slices[0].Spec.SharingAffinity...)
					}
					if tc.wantErr == sharingaffinity.ErrCostExceeded {
						f.slices[0].Spec.SharingAffinity, f.claims[0].Spec.Devices.Config = admissibleBudgetFixture(t)
					}
					// A second candidate must not turn a runtime failure into success.
					f.slices[1].Spec.Devices = append(f.slices[1].Spec.Devices, resourceapi.Device{Name: "device-b", AllowMultipleAllocations: ptr.To(true)})
					logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(10)))
					ctx := klog.NewContext(context.Background(), logger)
					if tc.wantErr == sharingaffinity.ErrCostExceeded {
						pool := sharingaffinity.PoolExtractors{
							Key:        sharingaffinity.PoolKey{Driver: sharingAffinityDriver, Name: "pool", Generation: 1},
							Extractors: f.slices[0].Spec.SharingAffinity,
						}
						result, err := sharingaffinity.NewExtractor(1).Extract(ctx, pool, sharingaffinity.Request{Name: "req"}, f.claims[0].Spec.Devices.Config[:1])
						if err != nil || !result.Satisfied {
							t.Fatalf("one admissible evaluation must fit the runtime budget: %+v, %v", result, err)
						}
					}
					a := f.allocator(t, ctx)
					var got []resourceapi.AllocationResult
					var affinity []internal.ClaimAffinity
					var err error
					if provenance {
						got, affinity, err = a.(affinityAllocator).AllocateWithAffinity(ctx, f.node, f.claims)
					} else {
						got, err = a.Allocate(ctx, f.node, f.claims)
					}
					if len(got) != 0 || len(affinity) != 0 {
						t.Errorf("fatal extraction returned results=%v, provenance=%v", got, affinity)
					}
					if !errors.Is(err, tc.wantErr) {
						t.Fatalf("Allocate error = %v, want wrapped %v", err, tc.wantErr)
					}
					wantOutcome := sharingaffinity.OutcomeRuntime
					if tc.wantErr == sharingaffinity.ErrClaimSelfInconsistent {
						wantOutcome = sharingaffinity.OutcomeClaimInconsistent
					}
					if sharingaffinity.Classify(err) != wantOutcome || errors.Is(err, internal.ErrFailedAllocationOnNode) {
						t.Errorf("fatal extraction has wrong scheduler classification: %v", err)
					}
					var attributed *sharingaffinity.Error
					if !errors.As(err, &attributed) || attributed.Pool.Driver != sharingAffinityDriver || attributed.Pool.Name != "pool" || attributed.Request != "req" {
						t.Fatalf("missing typed pool/request error attribution: %v", err)
					}
					if tc.wantErr == sharingaffinity.ErrCostExceeded && (attributed.ConfigIndex <= 0 || attributed.ConfigIndex >= len(f.claims[0].Spec.Devices.Config)) {
						t.Errorf("budget failed at config %d, want shared budget exhausted after at least one evaluation", attributed.ConfigIndex)
					}
					secrets := []string{secret, strings.Repeat("a", 64)}
					for _, extractor := range f.slices[0].Spec.SharingAffinity {
						secrets = append(secrets, extractor.Expression)
					}
					for _, config := range f.claims[0].Spec.Devices.Config {
						secrets = append(secrets, string(config.Opaque.Parameters.Raw))
					}
					logs := logger.GetSink().(ktesting.Underlier).GetBuffer().String()
					for _, secret := range secrets {
						if secret == "" {
							t.Fatal("redaction check requires non-empty expression and payload")
						}
						if strings.Contains(err.Error(), secret) || strings.Contains(logs, secret) {
							t.Errorf("error or logs disclosed input %.40q", secret)
						}
					}
				})
			}
		})
	}
}

func testSharingAffinityIsolation(t *testing.T, newAllocator allocatorFactory) {
	for _, tc := range []struct {
		name, second string
		separate     bool
		want         []string
	}{
		{"two-requests-same-device-equal", "blue", false, []string{"device-a", "device-a"}},
		{"two-requests-same-device-mismatch", "red", false, []string{"device-a", "device-b"}},
		{"two-claims-same-device-equal", "blue", true, []string{"device-a", "device-a"}},
		{"two-claims-same-device-mismatch", "red", true, []string{"device-a", "device-b"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newAllocationAffinityFixture(newAllocator)
			f.slices[1].Spec.Devices = append(f.slices[1].Spec.Devices, resourceapi.Device{Name: "device-b", AllowMultipleAllocations: ptr.To(true)})
			if tc.separate {
				second := f.claims[0].DeepCopy()
				second.Name, second.UID = "second", "second-uid"
				second.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(fmt.Sprintf(`{"tenant":%q}`, tc.second))}
				f.claims = append(f.claims, second)
			} else {
				f.claims[0].Spec.Devices.Requests = append(f.claims[0].Spec.Devices.Requests, allocationAffinityRequest("second"))
				f.claims[0].Spec.Devices.Config[0].Requests = []string{"req"}
				f.claims[0].Spec.Devices.Config = append(f.claims[0].Spec.Devices.Config, sharingAffinityConfig(fmt.Sprintf(`{"tenant":%q}`, tc.second), "second"))
			}
			_, ctx := ktesting.NewTestContext(t)
			a := f.allocator(t, ctx)
			got, err := a.Allocate(ctx, f.node, f.claims)
			assertAffinityAllocation(t, got, err, tc.want...)
			if len(got) != len(f.claims) {
				t.Fatalf("results = %d, want one per claim (%d)", len(got), len(f.claims))
			}
			for i, result := range got {
				for j, device := range result.Devices.Results {
					if device.Request != f.claims[i].Spec.Devices.Requests[j].Name || device.ShareID == nil || *device.ShareID == "" {
						t.Errorf("lost request or sharing identity: %#v", device)
					}
				}
			}
			// A later call may choose an incompatible value on the same clean
			// device: Filter has not acquired a durable owner.
			next := f.claims[0].DeepCopy()
			next.Spec.Devices.Requests = []resourceapi.DeviceRequest{allocationAffinityRequest("req")}
			next.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(`{"tenant":"green"}`)}
			again, err := a.Allocate(ctx, f.node, []*resourceapi.ResourceClaim{next})
			assertAffinityAllocation(t, again, err, "device-a")
		})
	}
}

func testSharingAffinityBacktracking(t *testing.T, newAllocator allocatorFactory) {
	t.Run("failed-attempt-does-not-poison-next-call", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.claims[0].Spec.Devices.Requests = append(f.claims[0].Spec.Devices.Requests, allocationAffinityRequest("second"))
		f.claims[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{
			sharingAffinityConfig(`{"tenant":"blue"}`, "req"), sharingAffinityConfig(`{"tenant":"red"}`, "second"),
		}
		_, ctx := ktesting.NewTestContext(t)
		a := f.allocator(t, ctx)
		got, err := a.Allocate(ctx, f.node, f.claims)
		assertAffinityAllocation(t, got, err)
		next := f.claims[0].DeepCopy()
		next.Spec.Devices.Requests = next.Spec.Devices.Requests[1:]
		got, err = a.Allocate(ctx, f.node, []*resourceapi.ResourceClaim{next})
		assertAffinityAllocation(t, got, err, "device-a")
	})
}

func testSharingAffinityConcurrent(t *testing.T, newAllocator allocatorFactory) {
	t.Run("nodes", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.lock("unrelated-device", map[string]string{"tenant": "unchanged"})
		before := f.state.DeviceSharingAffinity.DeviceStates.Clone()
		_, ctx := ktesting.NewTestContext(t)
		ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		a := f.allocator(t, ctx)
		start := make(chan struct{})
		var wg sync.WaitGroup
		results := make([][]resourceapi.AllocationResult, 8)
		provenance := make([][]internal.ClaimAffinity, 8)
		errs := make([]error, 8)
		for i := range results {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("node-%d", i)}}
				claim := f.claims[0].DeepCopy()
				claim.UID = apitypes.UID(fmt.Sprintf("claim-%d", i))
				claim.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{sharingAffinityConfig(fmt.Sprintf(`{"tenant":"value-%d"}`, i))}
				results[i], provenance[i], errs[i] = a.(affinityAllocator).AllocateWithAffinity(ctx, node, []*resourceapi.ResourceClaim{claim})
			}()
		}
		close(start)
		wg.Wait()
		if err := ctx.Err(); err != nil {
			t.Fatalf("concurrent allocations exceeded their bounded context: %v", err)
		}
		shares := sets.New[apitypes.UID]()
		for i := range results {
			assertAffinityAllocation(t, results[i], errs[i], "device-a")
			share := results[i][0].Devices.Results[0].ShareID
			if share == nil || *share == "" || shares.Has(*share) {
				t.Errorf("node %d has missing or reused share identity: %v", i, share)
			} else {
				shares.Insert(*share)
			}
			if len(provenance[i]) != 1 || len(provenance[i][0].Devices) != 1 || provenance[i][0].Devices[0].Values["tenant"] != fmt.Sprintf("value-%d", i) {
				t.Fatalf("node %d provenance = %#v, want own tenant value", i, provenance[i])
			}
		}
		provenance[0][0].Devices[0].Values["tenant"] = "mutated"
		if provenance[1][0].Devices[0].Values["tenant"] == "mutated" {
			t.Error("mutating one returned affinity map changed another concurrent call")
		}
		if f.state.DeviceSharingAffinity.DeviceStates[MakeDeviceID(sharingAffinityDriver, "pool", "unrelated-device")].LockedAffinity["tenant"] != "unchanged" {
			t.Error("mutating returned affinity map changed the allocation snapshot")
		}
		if !reflect.DeepEqual(before, f.state.DeviceSharingAffinity.DeviceStates) {
			t.Error("concurrent node allocation mutated snapshot lock maps")
		}
	})
}

func testSharingAffinityBudget(t *testing.T, newAllocator allocatorFactory) {
	// Measure an admission-valid expression's real runtime budget through the
	// existing extractor before using it to test allocator budget ownership.
	entries, configs := admissibleBudgetFixture(t)
	entry, config := entries[0], configs[0]
	_, ctx := ktesting.NewTestContext(t)
	pool := sharingaffinity.PoolExtractors{Key: sharingaffinity.PoolKey{Driver: sharingAffinityDriver, Name: "pool", Generation: 1}, Extractors: []resourceapi.SharingAffinityExtractor{entry}}
	extractor := sharingaffinity.NewExtractor(1)
	// Discover the boundary instead of depending on a particular CEL version's
	// exact cost formula. Each extraction consumes a fresh request budget.
	var below []resourceapi.DeviceClaimConfiguration
	for n := 1; n <= resourceapi.DeviceConfigMaxSize; n++ {
		configs := append(append([]resourceapi.DeviceClaimConfiguration(nil), below...), config)
		_, err := extractor.Extract(ctx, pool, sharingaffinity.Request{Name: "req"}, configs)
		if errors.Is(err, sharingaffinity.ErrCostExceeded) {
			break
		}
		if err != nil {
			t.Fatalf("measure runtime budget: %v", err)
		}
		below = configs
	}
	if len(below) == 0 || len(below) == resourceapi.DeviceConfigMaxSize {
		t.Fatalf("could not bracket runtime budget within the config admission limit, successful configs = %d", len(below))
	}
	for _, tc := range []struct {
		name string
		over bool
	}{
		{"per-request", false}, {"cost-exhausted", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newAllocationAffinityFixture(newAllocator)
			f.slices[0].Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{entry}
			f.claims[0].Spec.Devices.Config = append([]resourceapi.DeviceClaimConfiguration(nil), below...)
			f.claims[0].Spec.Devices.Requests = append(f.claims[0].Spec.Devices.Requests, allocationAffinityRequest("second"))
			if tc.over {
				f.claims[0].Spec.Devices.Config = append(f.claims[0].Spec.Devices.Config, config)
			}
			a := f.allocator(t, ctx)
			got, err := a.Allocate(ctx, f.node, f.claims)
			if tc.over {
				if !errors.Is(err, sharingaffinity.ErrCostExceeded) || sharingaffinity.Classify(err) != sharingaffinity.OutcomeRuntime || len(got) != 0 {
					t.Errorf("over-budget allocation = %d results, %v; want runtime ErrCostExceeded", len(got), err)
				}
			} else {
				assertAffinityAllocation(t, got, err, "device-a", "device-a")
			}
			next := f.claims[0].DeepCopy()
			next.Spec.Devices.Config = append([]resourceapi.DeviceClaimConfiguration(nil), below...)
			got, err = a.Allocate(ctx, f.node, []*resourceapi.ResourceClaim{next})
			assertAffinityAllocation(t, got, err, "device-a", "device-a")
		})
	}
	t.Run("across-candidate-rejection", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.slices[0].Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{entry}
		f.claims[0].Spec.Devices.Config = append([]resourceapi.DeviceClaimConfiguration(nil), below...)
		f.slices[1].Spec.Devices = nil
		for i := 0; i < 8; i++ {
			name := fmt.Sprintf("device-%d", i)
			f.slices[1].Spec.Devices = append(f.slices[1].Spec.Devices, resourceapi.Device{Name: name, AllowMultipleAllocations: ptr.To(true)})
			if i < 7 {
				f.lock(name, map[string]string{entry.Name: "red"})
			}
		}
		got, err := f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
		assertAffinityAllocation(t, got, err, "device-7")
	})
	t.Run("memo-once-per-pool-request", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.slices[1].Spec.Devices = nil
		for i := 0; i < 8; i++ {
			name := fmt.Sprintf("device-%d", i)
			f.slices[1].Spec.Devices = append(f.slices[1].Spec.Devices, resourceapi.Device{Name: name, AllowMultipleAllocations: ptr.To(true)})
			if i < 7 {
				f.lock(name, map[string]string{"tenant": "red"})
			}
		}
		logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(6)))
		ctx := klog.NewContext(context.Background(), logger)
		buffer := logger.GetSink().(ktesting.Underlier).GetBuffer()
		extractions := func() int {
			count := 0
			for _, entry := range buffer.Data() {
				if entry.Message == "Extracting sharing affinity" {
					count++
				}
			}
			return count
		}
		a := f.allocator(t, ctx)
		for _, requests := range []int{1, 2} {
			t.Run(fmt.Sprintf("requests-%d", requests), func(t *testing.T) {
				claim := f.claims[0].DeepCopy()
				want := []string{"device-7"}
				if requests == 2 {
					claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, allocationAffinityRequest("second"))
					want = append(want, "device-7")
				}
				before := extractions()
				got, err := a.Allocate(ctx, f.node, []*resourceapi.ResourceClaim{claim})
				// This existing extractor entry log observes real calls without
				// changing the production extractor or introducing a fake.
				if calls := extractions() - before; calls != requests {
					t.Errorf("extraction calls = %d across eight candidates, want %d (once per pool/request per Allocate call)", calls, requests)
				}
				assertAffinityAllocation(t, got, err, want...)
			})
		}
	})
}

func testSharingAffinityFreshness(t *testing.T, newAllocator allocatorFactory) {
	for _, change := range []string{"expression", "name", "group", "kind", "generation", "deletion", "metadata-added", "fenced-after-removal"} {
		t.Run(change, func(t *testing.T) {
			f := newAllocationAffinityFixture(newAllocator)
			f.lock("device-a", map[string]string{"tenant": "blue"})
			_, ctx := ktesting.NewTestContext(t)
			if change == "metadata-added" {
				f.slices[0].Spec.SharingAffinity = nil
			}
			first, err := f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
			assertAffinityAllocation(t, first, err, "device-a")
			// A fresh cycle reuses the program cache, not metadata predicates,
			// extracted values, or the previous cycle's successful decision.
			for i := range f.slices {
				f.slices[i] = f.slices[i].DeepCopy()
			}
			switch change {
			case "expression":
				f.slices[0].Spec.SharingAffinity[0].Expression = `"red"`
			case "name":
				f.slices[0].Spec.SharingAffinity[0].Name = "renamed"
			case "group":
				f.slices[0].Spec.SharingAffinity[0].APIGroup = "new.example.com"
			case "kind":
				f.slices[0].Spec.SharingAffinity[0].Kind = "NewConfig"
			case "generation":
				for _, s := range f.slices {
					s.Spec.Pool.Generation++
				}
				f.slices[0].Spec.SharingAffinity[0].Expression = `"red"`
			case "deletion":
				f.slices = f.slices[1:]
			case "metadata-added":
				f.slices[0].Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{sharingAffinityExtractor("zone", `parameters.?zone.orValue("")`)}
			case "fenced-after-removal":
				f.unconstrained()
				f.state.DeviceSharingAffinity.DeviceStates[MakeDeviceID(sharingAffinityDriver, "pool", "device-a")] = DeviceAffinityState{Status: internal.AffinityStatusUnreconstructable}
			}
			got, err := f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
			assertAffinityAllocation(t, got, err)
		})
	}
	t.Run("delete-newer-incomplete-generation", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		newer := f.slices[0].DeepCopy()
		newer.Name, newer.Spec.Pool.Generation = "newer-metadata", 2
		newer.Spec.SharingAffinity[0].Expression = `""`
		f.slices = append(f.slices, newer)
		_, ctx := ktesting.NewTestContext(t)
		got, err := f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
		assertAffinityAllocation(t, got, err)
		f.slices = f.slices[:2]
		got, err = f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
		assertAffinityAllocation(t, got, err, "device-a")
	})
	t.Run("same-generation-correction", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.slices[0].Spec.SharingAffinity[0].Expression = `""`
		_, ctx := ktesting.NewTestContext(t)
		got, err := f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
		assertAffinityAllocation(t, got, err)
		f.slices[0] = f.slices[0].DeepCopy()
		f.slices[0].Spec.SharingAffinity[0].Expression = `parameters.tenant`
		got, err = f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
		assertAffinityAllocation(t, got, err, "device-a")
	})
}

const (
	diagnosticMismatch    = "incompatible with the device's sharing affinity lock"
	diagnosticUnsatisfied = "claim configuration does not produce a value for every sharing affinity extractor of the pool"
	diagnosticUnknown     = "device sharing affinity is unknown because of legacy or inconsistent allocations; it can only be re-evaluated after all active allocations on the device are removed"
)

// admissibleBudgetFixture exhausts the request-wide budget, not admission limits.
// Keeping the work in a constant literal bounds the admission cost estimate.
func admissibleBudgetFixture(t *testing.T) ([]resourceapi.SharingAffinityExtractor, []resourceapi.DeviceClaimConfiguration) {
	t.Helper()
	const list = "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"
	expr := `has(parameters.kind) && ` + list + `.all(i, ` + list + `.all(j, "` + strings.Repeat("a", 4000) + `".matches("a{1,3}"))) ? "x" : "x"`
	entries := []resourceapi.SharingAffinityExtractor{{Name: "b0", APIGroup: "example.com", Kind: "Config", Expression: expr}}
	if len(expr) > resourceapi.SharingAffinityCELExpressionMaxLength {
		t.Fatalf("expression length %d exceeds admission limit", len(expr))
	}
	if err := sharingaffinity.ValidateExtractorSet(entries); err != nil {
		t.Fatalf("runtime fixture must pass estimated-cost admission: %v", err)
	}
	configs := make([]resourceapi.DeviceClaimConfiguration, 8)
	for i := range configs {
		configs[i] = sharingAffinityConfig(`{"apiVersion":"example.com/v1","kind":"Config"}`)
		if len(configs[i].Opaque.Parameters.Raw) > resourceapi.OpaqueParametersMaxLength {
			t.Fatal("runtime fixture exceeds opaque parameter admission limit")
		}
	}
	if len(configs) > resourceapi.DeviceConfigMaxSize {
		t.Fatal("runtime fixture exceeds claim config admission limit")
	}
	return entries, configs
}

func assertAffinityRejection(t *testing.T, err error, message string) {
	t.Helper()
	if !errors.Is(err, internal.ErrFailedAllocationOnNode) {
		t.Fatalf("error %v must preserve node-local retry semantics", err)
	}
	if err.Error() != message {
		t.Errorf("message = %q, want %q", err.Error(), message)
	}
}

func testSharingAffinityDiagnostics(t *testing.T, newAllocator allocatorFactory) {
	for _, tc := range []struct {
		name    string
		mutate  func(*allocationAffinityFixture)
		message string
	}{
		{"mismatch/locked", func(f *allocationAffinityFixture) {
			f.lock("device-a", map[string]string{"tenant": "private-lock-value"})
		}, diagnosticMismatch},
		{"mismatch/local-in-call", func(f *allocationAffinityFixture) {
			f.claims[0].Spec.Devices.Requests = append(f.claims[0].Spec.Devices.Requests, allocationAffinityRequest("second"))
			f.claims[0].Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{
				sharingAffinityConfig(`{"tenant":"private-first-value"}`, "req"),
				sharingAffinityConfig(`{"tenant":"private-second-value"}`, "second"),
			}
		}, diagnosticMismatch},
		{"unsatisfied/no-keys", func(f *allocationAffinityFixture) {
			f.claims[0].Spec.Devices.Config = nil
		}, diagnosticUnsatisfied},
		{"unsatisfied/sibling-masked", func(f *allocationAffinityFixture) {
			f.slices[0].Spec.SharingAffinity = append(f.slices[0].Spec.SharingAffinity, sharingAffinityExtractor("zone", `""`))
		}, diagnosticUnsatisfied},
		{"unknown/unreconstructable", func(f *allocationAffinityFixture) {
			id := MakeDeviceID(sharingAffinityDriver, "pool", "device-a")
			f.state.DeviceSharingAffinity.DeviceStates[id] = DeviceAffinityState{Status: internal.AffinityStatusUnreconstructable}
			f.state.AllocatedSharedDeviceIDs.Insert(id)
		}, diagnosticUnknown},
		{"invalid-pool-plus-rejection", func(f *allocationAffinityFixture) {
			f.lock("device-a", map[string]string{"tenant": "private-lock-value"})
			invalid := f.slices[0].DeepCopy()
			invalid.Name, invalid.Spec.Pool.Name = "invalid", "invalid"
			invalid.Spec.Pool.ResourceSliceCount = 1
			invalid.Spec.SharingAffinity = append(invalid.Spec.SharingAffinity, invalid.Spec.SharingAffinity[0])
			invalid.Spec.AllNodes = ptr.To(true)
			invalid.Spec.Devices = f.slices[1].Spec.Devices
			f.slices = append(f.slices, invalid)
		}, diagnosticMismatch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newAllocationAffinityFixture(newAllocator)
			tc.mutate(f)
			before := f.state.DeviceSharingAffinity.DeviceStates.Clone()
			_, ctx := ktesting.NewTestContext(t)
			got, provenance, err := f.allocator(t, ctx).(affinityAllocator).AllocateWithAffinity(ctx, f.node, f.claims)
			if len(got) != 0 || len(provenance) != 0 {
				t.Errorf("failed allocation returned results=%v, provenance=%v", got, provenance)
			}
			if !reflect.DeepEqual(before, f.state.DeviceSharingAffinity.DeviceStates) {
				t.Error("diagnostics changed snapshot locks")
			}
			message := "sharing affinity: " + tc.message
			if tc.name == "invalid-pool-plus-rejection" {
				message = "invalid resource pools were encountered; " + message
			}
			assertAffinityRejection(t, err, message)
			for _, secret := range []string{"private-lock-value", "private-first-value", "private-second-value", "blue", f.slices[0].Spec.SharingAffinity[0].Expression} {
				if strings.Contains(err.Error(), secret) {
					t.Errorf("message discloses %q", secret)
				}
			}
		})
	}

	t.Run("combined/real-allocation", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.lock("device-a", map[string]string{"tenant": "private-lock-value"})
		for _, name := range []string{"unsatisfied", "unknown"} {
			slice := f.slices[1].DeepCopy()
			slice.Name, slice.Spec.Pool.Name = name, name
			slice.Spec.Pool.ResourceSliceCount = 1
			slice.Spec.SharingAffinity = []resourceapi.SharingAffinityExtractor{sharingAffinityExtractor("tenant", `""`)}
			if name == "unknown" {
				id := MakeDeviceID(sharingAffinityDriver, name, "device-a")
				f.state.DeviceSharingAffinity.DeviceStates[id] = DeviceAffinityState{Status: internal.AffinityStatusUnreconstructable}
				f.state.AllocatedSharedDeviceIDs.Insert(id)
			}
			f.slices = append(f.slices, slice)
		}
		_, ctx := ktesting.NewTestContext(t)
		got, provenance, err := f.allocator(t, ctx).(affinityAllocator).AllocateWithAffinity(ctx, f.node, f.claims)
		if len(got) != 0 || len(provenance) != 0 {
			t.Error("combined failure returned partial allocation")
		}
		assertAffinityRejection(t, err,
			"sharing affinity: "+strings.Join([]string{diagnosticMismatch, diagnosticUnsatisfied, diagnosticUnknown}, "; "))
	})

	t.Run("success-with-rejected-alternative", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.lock("device-a", map[string]string{"tenant": "red"})
		f.slices[1].Spec.Devices = append(f.slices[1].Spec.Devices, resourceapi.Device{Name: "device-b", AllowMultipleAllocations: ptr.To(true)})
		_, ctx := ktesting.NewTestContext(t)
		a := f.allocator(t, ctx).(affinityAllocator)
		for range 2 {
			got, provenance, err := a.AllocateWithAffinity(ctx, f.node, f.claims)
			assertAffinityAllocation(t, got, err, "device-b")
			if len(provenance) != 1 {
				t.Errorf("provenance count = %d, want 1", len(provenance))
			}
		}
	})
	for _, name := range []string{"allocate-without-provenance", "no-affinity-pool"} {
		t.Run(name, func(t *testing.T) {
			f := newAllocationAffinityFixture(newAllocator)
			_, ctx := ktesting.NewTestContext(t)
			switch name {
			case "allocate-without-provenance":
				f.lock("device-a", map[string]string{"tenant": "red"})
				got, err := f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
				assertAffinityAllocation(t, got, err)
			case "no-affinity-pool":
				f.unconstrained()
				got, _, err := f.allocator(t, ctx).(affinityAllocator).AllocateWithAffinity(ctx, f.node, f.claims)
				assertAffinityAllocation(t, got, err, "device-a")
			}
		})
	}
	t.Run("pool-dump-redaction", func(t *testing.T) { testAffinityDiagnosticPoolDump(t, newAllocator) })
	t.Run("unsatisfied/log-extractor-names", func(t *testing.T) {
		f := newAllocationAffinityFixture(newAllocator)
		f.slices[0].Spec.SharingAffinity = append(f.slices[0].Spec.SharingAffinity, sharingAffinityExtractor("missing-zone", `""`))
		logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.Verbosity(7), ktesting.BufferLogs(true)))
		ctx := klog.NewContext(t.Context(), logger)
		_, _, err := f.allocator(t, ctx).(affinityAllocator).AllocateWithAffinity(ctx, f.node, f.claims)
		found := false
		for _, entry := range logger.GetSink().(ktesting.Underlier).GetBuffer().Data() {
			fields := map[string]any{}
			for _, kv := range [][]any{entry.WithKVList, entry.ParameterKVList} {
				for i := 0; i+1 < len(kv); i += 2 {
					if key, ok := kv[i].(string); ok {
						fields[key] = kv[i+1]
					}
				}
			}
			if fields["reason"] == "unsatisfied" {
				found = true
				if !reflect.DeepEqual(fields["unsatisfied"], []string{"missing-zone"}) {
					t.Errorf("unsatisfied rejection log names = %v, want [missing-zone]", fields["unsatisfied"])
				}
				for _, key := range []string{"device", "request"} {
					if _, ok := fields[key]; !ok {
						t.Errorf("rejection log omits %s", key)
					}
				}
			}
		}
		if !found {
			t.Error("missing V(7) device rejection log")
		}
		assertAffinityRejection(t, err, "sharing affinity: "+diagnosticUnsatisfied)
	})
}

func testAffinityDiagnosticPoolDump(t *testing.T, newAllocator allocatorFactory) {
	const canary = "kep5981-cel-canary-7f3a"
	const annotation = "kubectl.kubernetes.io/last-applied-configuration"
	f := newAllocationAffinityFixture(newAllocator)
	f.slices[0].Spec.AllNodes = ptr.To(true)
	f.slices[0].Spec.SharingAffinity[0].Expression = `"` + canary + `" == "x" ? "a" : "b"`
	for _, slice := range f.slices {
		slice.Annotations = map[string]string{annotation: canary}
		slice.ManagedFields = []metav1.ManagedFieldsEntry{{Manager: "driver", FieldsV1: &metav1.FieldsV1{Raw: []byte(`{"f:metadata":{"f:annotations":{"f:` + canary + `":{}}}}`)}}}
	}
	before := make([]*resourceapi.ResourceSlice, len(f.slices))
	for i, slice := range f.slices {
		before[i] = slice.DeepCopy()
	}
	for _, verbosity := range []int{0, 7} {
		logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.Verbosity(verbosity), ktesting.BufferLogs(true)))
		ctx := klog.NewContext(t.Context(), logger)
		got, err := f.allocator(t, ctx).Allocate(ctx, f.node, f.claims)
		assertAffinityAllocation(t, got, err, "device-a")
		logs := logger.GetSink().(ktesting.Underlier).GetBuffer().String()
		if strings.Contains(logs, canary) {
			t.Errorf("V(%d) pool log discloses CEL or metadata payload", verbosity)
		}
		if verbosity == 7 {
			for _, token := range []string{"content", "tenant", "apiGroup", "kind", annotation} {
				if !strings.Contains(logs, token) {
					t.Errorf("V(7) pool dump missing %q", token)
				}
			}
		}
		if !reflect.DeepEqual(f.slices, before) {
			t.Error("logging mutated informer slices, expressions, annotations or managed fields")
		}
	}
}

// TestAllocatorSelection takes the public constructor without converting its result,
// so a non-nil allocator returned with an error is not hidden by an adapter.
func TestAllocatorSelection[A any](
	t *testing.T,
	enableAllocators func(...string),
	newAllocator func(context.Context, Features, AllocatedState, DeviceClassLister, []*resourceapi.ResourceSlice, *cel.Cache) (A, error),
) {
	t.Cleanup(func() { enableAllocators() })
	t.Run("sharing-affinity", func(t *testing.T) {
		t.Run("feature sets", func(t *testing.T) {
			if !internal.FeaturesAll.SharingAffinity || !internal.FeaturesAll.Set().Has("DRASharingAffinity") {
				t.Error("FeaturesAll must enable and advertise DRASharingAffinity")
			}
			for _, enabled := range []bool{false, true} {
				if got := (Features{SharingAffinity: enabled}).Set().Has("DRASharingAffinity"); got != enabled {
					t.Errorf("Features{SharingAffinity:%t}.Set contains gate = %t", enabled, got)
				}
			}
		})
		for _, tc := range []struct {
			name     string
			channels []string
			wantErr  bool
		}{
			{"stable", []string{"stable"}, true},
			{"incubating", []string{"incubating"}, true},
			{"older implementations", []string{"stable", "incubating"}, true},
			{"experimental", []string{"experimental"}, false},
			{"default selection", nil, false},
		} {
			t.Run(tc.name, func(t *testing.T) {
				_, ctx := ktesting.NewTestContext(t)
				enableAllocators(tc.channels...)
				allocator, err := newAllocator(ctx, Features{ConsumableCapacity: true, SharingAffinity: true}, AllocatedState{}, nil, nil, nil)
				if tc.wantErr {
					if err == nil || any(allocator) != nil {
						t.Errorf("unsupported constructor = (%T, %v), want nil allocator and error", allocator, err)
					}
					return
				}
				if err != nil {
					t.Fatalf("constructor: %v", err)
				}
				actual, ok := any(allocator).(Allocator)
				if !ok {
					t.Fatalf("allocator %T does not expose its implementation channel", allocator)
				}
				if got := actual.Channel(); got != internal.Experimental {
					t.Errorf("channel = %q, want experimental", got)
				}
			})
		}
	})
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
	numaNodeAttribute := resourceapi.FullyQualifiedName("resource.kubernetes.io/numaNode")
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

	testcases := map[string]AllocatorTestCase{
		"empty": {},
		// Two requests in one claim, both landing on the same counter set with the
		// same compatibility group: they can be co-allocated.
		"compatibility-groups-compatible": {
			features: Features{PartitionableDevices: true, CompatibilityGroups: true},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 1),
				deviceRequest(req1, classA, 1),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: two}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: two}, "mig"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1, driverA, pool1, device2, false),
			)},
		},
		// Two requests in one claim, the only candidates are an incompatible pair on
		// the same counter set (mig vs vgpu): no combination satisfies the claim.
		"compatibility-groups-incompatible": {
			features: Features{PartitionableDevices: true, CompatibilityGroups: true},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 1),
				deviceRequest(req1, classA, 1),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: two}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: two}, "vgpu"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		// A single-device claim whose first candidate (device1) declares group "mig"
		// and carries a NoSchedule taint, while the second (device2) declares the
		// disjoint group "vgpu". device1 is rejected by its taint; the compatibility
		// narrowing it consumed on counterSet1 must be rolled back so device2, which
		// shares no group with it, can still be allocated.
		"compatibility-groups-taint-rollback": {
			features: Features{PartitionableDevices: true, CompatibilityGroups: true, DeviceTaints: true},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 1),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withTaints(taintNoSchedule).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: two}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: two}, "vgpu"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		// An incompatible device is already allocated on the counter set; its groups
		// are read live from the slice. The only remaining candidate shares no
		// group, so the claim is unschedulable.
		"compatibility-groups-incompatible-with-allocated-peer": {
			features:         Features{PartitionableDevices: true, CompatibilityGroups: true},
			claimsToAllocate: objects(claim(claim0).withRequests(deviceRequest(req0, classA, 1))),
			allocatedDevices: []DeviceID{MakeDeviceID(driverA, pool1, device1)},
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "vgpu"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		// A compatible device is already allocated on the counter set; the candidate
		// shares the group the peer declares in the slice and is admitted.
		"compatibility-groups-compatible-with-allocated-peer": {
			features:         Features{PartitionableDevices: true, CompatibilityGroups: true},
			claimsToAllocate: objects(claim(claim0).withRequests(deviceRequest(req0, classA, 1))),
			allocatedDevices: []DeviceID{MakeDeviceID(driverA, pool1, device1)},
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		// With the feature disabled, a slice with a device declaring
		// compatibility groups is ignored during pool gathering (the scheduler
		// cannot validate it), which leaves its pool incomplete and blocks the
		// pool as a whole. A separate pool without groups is still allocatable.
		// This is the feature-off safety behavior and is verified across all
		// allocator variants.
		"compatibility-groups-disabled-skips-grouped-pool": {
			// CompatibilityGroups is set explicitly to false so this case keeps
			// exercising the feature-disabled path unchanged after the feature
			// graduates to on-by-default. The scope is the pool: device1's pool1
			// is off-limits, but device2 in the group-free pool2 is still
			// allocated.
			features: Features{PartitionableDevices: true, CompatibilityGroups: false},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 1),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: two}, "mig"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool2, 2), driverA,
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet2, map[string]resource.Quantity{counter0: two}),
					),
				),
				sliceWithCounterSets(slice4, node1, resourcePool(pool2, 2), driverA,
					counterSet(counterSet2, map[string]resource.Quantity{counter0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool2, device2, false),
			)},
		},
		// With the feature disabled, a candidate that declares no groups is not
		// allocatable either when another device of the same pool declares them:
		// only the slice with the grouped device1 gets ignored during pool
		// gathering, but that leaves the pool incomplete, which also blocks
		// device2 in the remaining device slice - whether or not device1 is
		// allocated, so that the outcome does not depend on allocation order.
		// Verified across all allocator variants, including stable.
		"compatibility-groups-disabled-skips-ungrouped-device-in-grouped-pool": {
			features:         Features{PartitionableDevices: true, CompatibilityGroups: false},
			claimsToAllocate: objects(claim(claim0).withRequests(deviceRequest(req0, classA, 1))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 3), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 3), driverA,
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{counter0: one}),
					),
				),
				sliceWithCounterSets(slice3, node1, resourcePool(pool1, 3), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		// Groups are tracked per counter set: two devices whose groups are
		// disjoint can still be co-allocated when they draw from different
		// counter sets of the same pool.
		"compatibility-groups-multiple-counter-sets": {
			features: Features{PartitionableDevices: true, CompatibilityGroups: true},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 1),
				deviceRequest(req1, classA, 1),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: two}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet2, map[string]resource.Quantity{counter0: two}, "vgpu"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
					counterSet(counterSet2, map[string]resource.Quantity{counter0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1, driverA, pool1, device2, false),
			)},
		},
		// A device drawing from several counter sets must be admitted by every
		// one of them. device2 is unopposed on counterSet2 (nothing else draws
		// from that set) but shares no group with device1 on counterSet1, so
		// no combination satisfies the claim - including the backtracked order
		// where device2 goes first and its narrowing of both counter sets has
		// to be rolled back.
		"compatibility-groups-multiple-counter-sets-spanning-device": {
			features: Features{PartitionableDevices: true, CompatibilityGroups: true},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 1),
				deviceRequest(req1, classA, 1),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "vgpu"),
						deviceCounterConsumptionWithGroups(counterSet2, map[string]resource.Quantity{counter0: one}, "vgpu"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
					counterSet(counterSet2, map[string]resource.Quantity{counter0: four}),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		// The first subrequest asks for two devices, but only one candidate is
		// compatible with the "mig" device taken by req0 (device3 declares the
		// disjoint "vgpu"), so it fails and its tentative narrowing is rolled
		// back before the allocator falls back to the second subrequest.
		"compatibility-groups-with-prioritized-list": {
			features: Features{PartitionableDevices: true, CompatibilityGroups: true, PrioritizedList: true},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 1),
				requestWithPrioritizedList(req1,
					subRequest(subReq0, classA, 2),
					subRequest(subReq1, classA, 1),
				),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
					device(device3).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "vgpu"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1SubReq1, driverA, pool1, device2, false),
			)},
		},
		// Compatibility groups steer allocation among shareable devices: req1
		// does not fit on device1 (req0's share exhausted its capacity) and is
		// rejected on device2, which declares the disjoint "vgpu" group on the
		// same counter set, before landing on device3.
		"compatibility-groups-with-consumable-capacity": {
			features: Features{PartitionableDevices: true, CompatibilityGroups: true, ConsumableCapacity: true},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one),
				deviceRequest(req1, classA, 1).withCapacityRequest(capacity0, one),
			)),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withCapacity(capacity0, "1").withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					).withAllowMultipleAllocations(),
					device(device2).withCapacity(capacity0, "1").withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "vgpu"),
					).withAllowMultipleAllocations(),
					device(device3).withCapacity(capacity0, "1").withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					).withAllowMultipleAllocations(),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
				deviceRequestAllocationResult(req1, driverA, pool1, device3).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
			)},
		},
		// Two claims allocated in a single call must satisfy the constraint
		// jointly: the second claim's device is checked against the rolling
		// intersection left by the first claim's device.
		"compatibility-groups-multiple-claims": {
			features: Features{PartitionableDevices: true, CompatibilityGroups: true},
			claimsToAllocate: objects(
				claimWithRequest(claim0, req0, classA),
				claimWithRequest(claim1, req0, classA),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverA, pool1, device1, false)),
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverA, pool1, device2, false)),
			},
		},
		// Each claim could be satisfied on its own, but not both in one call:
		// whichever device goes to the first claim, the other claim's only
		// remaining candidate declares a disjoint group on the same counter
		// set. The rolling intersection spans claims, so the joint allocation
		// is correctly refused.
		"compatibility-groups-multiple-claims-incompatible": {
			features: Features{PartitionableDevices: true, CompatibilityGroups: true},
			claimsToAllocate: objects(
				claimWithRequest(claim0, req0, classA),
				claimWithRequest(claim1, req0, classA),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "mig"),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumptionWithGroups(counterSet1, map[string]resource.Quantity{counter0: one}, "vgpu"),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{counter0: four}),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
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
		"optional-node-operations": {
			features:         Features{OptionalNodeOperations: true},
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA).withSkipNodeOperations(resourceapi.SkipNodeOperationAll)),
			node:             nodeWithDRAOptionalNodeOperations(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResultWithSkipNodeOperations(req0, driverA, pool1, device1, resourceapi.SkipNodeOperationAll),
			)},
		},
		"optional-node-operations-missing-feature": {
			features:         Features{OptionalNodeOperations: true},
			claimsToAllocate: objects(claimWithRequest(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           unwrapResourceSlices(sliceWithOneDevice(slice1, node1, pool1, driverA).withSkipNodeOperations(resourceapi.SkipNodeOperationAll)),
			node:             node(node1, region1),

			expectResults: []any{},
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
				device(device1).withCapacity("memory", "1Gi"),
				device(device2).withCapacity("memory", "2Gi"),
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
				device(device2).withCapacity("memory", "2Gi"),
				device(device1).withCapacity("memory", "1Gi"),
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
				device(device2).withCapacity("memory", "2Gi"),
				device(device1).withCapacity("memory", "1Gi"),
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
					device(device1),
				),
				func() wrapResourceSliceWithDevices {
					slice := sliceWithDevices("slice-1-obsolete", node1, resourcePool(pool1, 2), driverA,
						device(device2),
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
		"all-devices-of-invalid-pool": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name: req0,
				Exactly: &resourceapi.ExactDeviceRequest{
					AllocationMode:  resourceapi.DeviceAllocationModeAll,
					DeviceClassName: classA,
				},
			})),
			classes: objects(class(classA, driverA)),
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

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: asks for all devices, but resource pool driver-a/pool-1 is currently invalid")),
			expectErrorIs: internal.ErrFailedAllocationOnNode,
		},
		"all-devices-with-consumed-counters": {
			features: Features{
				PartitionableDevices: true,
			},
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
					device(device1).withDeviceCounterConsumption(
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
			expectErrorIs: internal.ErrFailedAllocationOnNode,
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
					device(device1),
					device(device2),
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
					device(device1),
					device(device2),
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
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1), device(device2)),
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
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1), device(device2)),
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
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1), device(device2)),
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
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1), device(device2)),
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
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1), device(device2)),
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
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1), device(device2)),
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
				sliceWithDevices(slice1, node1, pool1, driverA, device(device1), device(device2)),
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
					device(device1),
					device(device2),
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
				device(device1).
					withAttribute("driverVersion", semver.MustParse("1.0.0")).
					withAttribute("numa", 1).
					withAttribute("stringAttribute", "stringAttributeValue").
					withAttribute("boolAttribute", true),
				device(device2).
					withAttribute("driverVersion", semver.MustParse("1.0.0")).
					withAttribute("numa", 1).
					withAttribute("stringAttribute", "stringAttributeValue").
					withAttribute("boolAttribute", true),
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
				device(device1).withAttribute("numa", 1),
				device(device2).withAttribute("numa", 2),
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
				device(device1).withAttribute("numa", 1),
				device(device2).withAttribute("numa", 2),
			)),
			node: node(node1, region1),

			expectError: gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: cannot add device driver-a/pool-1/device-2 because a claim constraint would not be satisfied")),
		},
		"partitionable-all-mode-constraint-must-error-with-reserved-counter": {
			// Covers allocateDevice's must=true error path when the rejected device
			// has already reserved a shared counter. An all-mode request over two
			// devices with mismatched constraint attributes fails on the second
			// device, which consumes the counter, so rollbackDevice runs with a
			// reserved counter before the must-error is returned. The error aborts the
			// whole allocation, so this asserts the error contract and exercises the
			// counter-release branch; the released counter is not separately
			// observable through the allocation result.
			features: Features{
				PartitionableDevices: true,
			},
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
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).
						withAttribute("numa", 1),
					// device2 mismatches the constraint and consumes the single
					// counter, so it reserves the counter first and then fails the
					// constraint, reaching the must-error path with state to undo.
					device(device2).
						withAttribute("numa", 2).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
						),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
				),
			),
			node:        node(node1, region1),
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
				device(device1).withAttribute("stringAttribute", "stringAttributeValue"),
				device(device2).withAttribute("stringAttribute", "stringAttributeValue2"),
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
				device(device1).withAttribute("boolAttribute", true),
				device(device2).withAttribute("boolAttribute", false),
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
				device(device1).withAttribute("driverVersion", semver.MustParse("1.0.0")),
				device(device2).withAttribute("driverVersion", semver.MustParse("2.0.0")),
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
				device(device1).withAttribute("driverVersion", semver.MustParse("1.0.0")),
				device(device2).withAttribute("driverVersion", semver.MustParse("2.0.0")),
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
				device(device1).
					withAttribute("driverVersion", semver.MustParse("1.0.0")).
					withAttribute("stringAttribute", "a"),
				device(device2).
					withAttribute("driverVersion", semver.MustParse("2.0.0")).
					withAttribute("stringAttribute", "b"),
				device(device3).
					withAttribute("driverVersion", semver.MustParse("3.0.0")).
					withAttribute("stringAttribute", "b"),
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
							Requests:            nil,
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
		"with-claim-config-for-two-requests": {
			claimsToAllocate: func() []wrapResourceClaim {
				c := claimWithRequests(claim0, nil, request(req0, classA, 1), request(req1, classA, 1), request(req2, classA, 1))
				c.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{
					{
						Requests:            []string{req0, req1},
						DeviceConfiguration: deviceConfiguration(driverA, "deviceAttribute"),
					},
				}
				return []wrapResourceClaim{c}
			}(),
			classes: objects(
				class(classA, driverA),
			),
			slices: unwrap(
				sliceWithMultipleDevices(slice1, node1, pool1, driverA, 3),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResultWithConfigs(
					localNodeSelector(node1),
					objects(
						deviceAllocationResult(req0, driverA, pool1, device0, false),
						deviceAllocationResult(req1, driverA, pool1, device1, false),
						deviceAllocationResult(req2, driverA, pool1, device2, false),
					),
					[]resourceapi.DeviceAllocationConfiguration{
						{
							Source:              resourceapi.AllocationConfigSourceClaim,
							Requests:            []string{req0, req1},
							DeviceConfiguration: deviceConfiguration(driverA, "deviceAttribute"),
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
		// Two counts that are each representable but together overflow the native
		// int accumulator to a negative value, rejected before the sum is formed.
		// The value comes from math.MaxInt so it also triggers on 32-bit builds.
		"exact-count-total-overflows-claim-limit": {
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, int64(math.MaxInt/2+1)),
				deviceRequest(req1, classA, int64(math.MaxInt/2+1)),
			)),
			classes:     objects(class(classA, driverA)),
			expectError: gomega.MatchError(gomega.ContainSubstring("exceeds the claim limit")),
		},
		// Four counts that would wrap a naive int accumulator back to zero rather
		// than negative; each is rejected against remaining capacity before summing.
		"exact-count-total-would-wrap-to-zero": {
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, int64(math.MaxInt/2+1)),
				deviceRequest(req1, classA, int64(math.MaxInt/2+1)),
				deviceRequest(req2, classA, int64(math.MaxInt/2+1)),
				deviceRequest(req3, classA, int64(math.MaxInt/2+1)),
			)),
			classes:     objects(class(classA, driverA)),
			expectError: gomega.MatchError(gomega.ContainSubstring("exceeds the claim limit")),
		},
		// req0 fits on its own, so the rejection has to name req1 as the one
		// that pushes the claim over.
		"exact-count-sum-exceeds-claim-limit": {
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 20),
				deviceRequest(req1, classA, 20),
			)),
			classes: objects(class(classA, driverA)),
			expectError: gomega.MatchError(gomega.And(
				gomega.ContainSubstring("request "+req1),
				gomega.ContainSubstring("exceeds the claim limit"),
			)),
		},
		// Two counts whose sum exactly fills the per-claim limit both allocate;
		// this pins the pre-flight comparison as > and not >=.
		"exact-count-sum-exactly-fills-claim-limit": {
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, resourceapi.AllocationResultsMaxSize/2),
				deviceRequest(req1, classA, resourceapi.AllocationResultsMaxSize/2),
			)),
			classes: objects(class(classA, driverA)),
			slices:  unwrapResourceSlices(sliceWithMultipleDevices(slice1, node1, pool1, driverA, resourceapi.AllocationResultsMaxSize)),
			node:    node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				slices.Concat(
					multipleDeviceAllocationResults(req0, driverA, pool1, resourceapi.AllocationResultsMaxSize/2, 0),
					multipleDeviceAllocationResults(req1, driverA, pool1, resourceapi.AllocationResultsMaxSize/2, resourceapi.AllocationResultsMaxSize/2),
				)...,
			)},
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
				device(device1),
				device(device2),
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
						Source:              resourceapi.AllocationConfigSourceClaim,
						Requests:            nil,
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
					device(device1),
					device(device2),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device3),
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
						Requests:            nil,
						DeviceConfiguration: deviceConfiguration(driverB, "bar"),
					},
				},
			)},
		},
		"claim-config-scoped-to-subrequests-of-one-request": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 1),
						subRequest(subReq1, classB, 1),
					),
					deviceRequest(req1, classA, 1),
				).withConfigs(
					objects(
						deviceClaimConfig([]string{req0SubReq0, req0SubReq1}, deviceConfiguration(driverA, "scoped")),
					),
				),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrap(
				sliceWithMultipleDevices(slice1, node1, pool1, driverA, 2),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResultWithConfigs(
				localNodeSelector(node1),
				objects(
					deviceAllocationResult(req0SubReq0, driverA, pool1, device0, false),
					deviceAllocationResult(req1, driverA, pool1, device1, false),
				),
				[]resourceapi.DeviceAllocationConfiguration{
					{
						Source:              resourceapi.AllocationConfigSourceClaim,
						Requests:            []string{req0SubReq0, req0SubReq1},
						DeviceConfiguration: deviceConfiguration(driverA, "scoped"),
					},
				},
			)},
		},
		"claim-config-covering-all-requests-is-cleared": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 1),
						subRequest(subReq1, classB, 1),
					),
					deviceRequest(req1, classA, 1),
				).withConfigs(
					objects(
						// Covers req0 (via its selected subrequest) and req1 (by name),
						// i.e. every top-level request, so Requests must collapse to nil.
						deviceClaimConfig([]string{req0SubReq0, req1}, deviceConfiguration(driverA, "all")),
					),
				),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrap(
				sliceWithMultipleDevices(slice1, node1, pool1, driverA, 2),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResultWithConfigs(
				localNodeSelector(node1),
				objects(
					deviceAllocationResult(req0SubReq0, driverA, pool1, device0, false),
					deviceAllocationResult(req1, driverA, pool1, device1, false),
				),
				[]resourceapi.DeviceAllocationConfiguration{
					{
						Source:              resourceapi.AllocationConfigSourceClaim,
						Requests:            nil,
						DeviceConfiguration: deviceConfiguration(driverA, "all"),
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
				device(device1).withCapacity("memory", "2Gi"),
				device(device2).withCapacity("memory", "2Gi"),
				device(device3).withCapacity("memory", "1Gi"),
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
					device(device1).
						withCapacity("memory", "8Gi").
						withAttribute("driverVersion", semver.MustParse("1.0.0")),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device2).
						withCapacity("memory", "2Gi").
						withAttribute("driverVersion", semver.MustParse("2.0.0")),
					device(device3).
						withCapacity("memory", "1Gi").
						withAttribute("driverVersion", semver.MustParse("1.0.0")),
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
					device(device1).
						withCapacity("memory", "8Gi").
						withAttribute("driverVersion", semver.MustParse("1.0.0")),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device2).
						withCapacity("memory", "2Gi").
						withAttribute("driverVersion", "2.0.0"),
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
					device(device1),
					device(device2),
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
					device(device1),
					device(device2),
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
					device(device1).withCapacity("memory", "8Gi"),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device2).withCapacity("memory", "4Gi"),
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
					device(device1).withCapacity("memory", "8Gi"),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device2).withCapacity("memory", "4Gi"),
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
		// A prioritized list is worth at least its smallest alternative, so the
		// pre-flight check reads that. min(20, 24) leaves the claim over the
		// limit before any device is looked at.
		"prioritized-list-minimum-sum-exceeds-claim-limit": {
			features: Features{PrioritizedList: true},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 20),
				requestWithPrioritizedList(req1,
					subRequest(subReq0, classA, 20),
					subRequest(subReq1, classA, 24),
				),
			)),
			classes: objects(class(classA, driverA)),
			expectError: gomega.MatchError(gomega.And(
				gomega.ContainSubstring("request "+req1),
				gomega.ContainSubstring("exceeds the claim limit"),
			)),
		},
		// The oversized alternative sits behind a device already allocated for the
		// claim, which is what overflowed the per-request check. Only the pinned
		// invocation count separates pruning it from descending into it.
		"prioritized-list-overflowing-first-subrequest-is-pruned": {
			features: Features{PrioritizedList: true},
			claimsToAllocate: objects(claim(claim0).withRequests(
				deviceRequest(req0, classA, 1),
				requestWithPrioritizedList(req1,
					subRequest(subReq0, classA, int64(math.MaxInt)),
					subRequest(subReq1, classA, 1),
				),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1),
				device(device2),
			)),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1SubReq1, driverA, pool1, device2, false),
			)},
			expectNumAllocateOneInvocations: 8,
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
					device(device1).withDeviceCounterConsumption(
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
		"partitionable-devices-counter-quantities-are-not-modified": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
			),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
			},
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("9000000000000000000")}),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("10000000000000000000")}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"partitionable-devices-counter-cache-scoped-by-pool-ID": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
				claimWithRequests(claim1, nil, request(req1, classB, 1)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(
				// driverA / pool "pool-1": counterSet1 c:1, device1 consumes c:1.
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
				),
				// driverB / pool "pool-1": same pool name, different driver. Its capacity and
				// consumption differ (c:2) on purpose, so that leaving *either* cache keyed by
				// pool name alone fails this case: device2 would be checked against driverA's
				// cached c:1 and wrongly rejected.
				sliceWithDevices(slice3, node1, resourcePool(pool1, 2), driverB,
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("2")}),
					),
				),
				sliceWithCounterSets(slice4, node1, resourcePool(pool1, 2), driverB,
					counterSet(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("2")}),
				),
			),
			node: node(node1, region1),
			// driver-a/pool-1 and driver-b/pool-1 are distinct pools with independent counter
			// sets, so both devices allocate. The differing capacities make this guard both
			// caches: keying availableCounters or consumedCounters by pool name alone lets
			// driver A's cache leak into driver B's pool and fails the case.
			expectResults: []any{
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverA, pool1, device1, false)),
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req1, driverB, pool1, device2, false)),
			},
		},
		"partitionable-devices-counter-cache-not-crossed-between-driver-schemas": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
				claimWithRequests(claim1, nil, request(req1, classB, 1)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: unwrapResourceSlices(
				// driverA / pool "pool-1": counterSet1 with c:1, device1 consumes c:1 (valid).
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
				),
				// driverB / pool "pool-1": same pool name but a different counter set
				// (counterSet2 with d:1). device2 over-consumes it (d:2) and must be rejected.
				sliceWithDevices(slice3, node1, resourcePool(pool1, 2), driverB,
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet2, map[string]resource.Quantity{"d": resource.MustParse("2")}),
					),
				),
				sliceWithCounterSets(slice4, node1, resourcePool(pool1, 2), driverB,
					counterSet(counterSet2, map[string]resource.Quantity{"d": resource.MustParse("1")}),
				),
			),
			node: node(node1, region1),
			// device2 over-consumes its own counter (d:2 > d:1), so it must be rejected and
			// the allocation cannot be satisfied. If the caches are keyed by pool name,
			// device2 reuses driver A's cached counterSet1 and its own counterSet2 is never
			// compared, so it is wrongly accepted (over-allocation).
			expectResults: []any{},
		},
		"partitionable-devices-taint-releases-counter": {
			features: Features{
				PartitionableDevices: true,
				DeviceTaints:         true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withTaints(taintNoSchedule).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
					),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
				),
			),
			node: node(node1, region1),
			// device1 is rejected by its NoSchedule taint after its counter is
			// reserved. Releasing that reservation lets the clean device2 allocate.
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"partitionable-devices-constraint-releases-counter": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0,
					[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
					request(req0, classA, 2),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).
						withAttribute("stringAttribute", "red").
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
						),
					device(device2).
						withAttribute("stringAttribute", "blue").
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
						),
					device(device3).
						withAttribute("stringAttribute", "blue").
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
						),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("2")}),
				),
			),
			node: node(node1, region1),
			// device1 (red) is accepted first; the blue candidates are rejected by
			// the match-attribute constraint after their counters are reserved.
			// Releasing those reservations lets the two blue devices allocate.
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"partitionable-consumable-capacity-backtrack-releases-counter": {
			features: Features{
				PartitionableDevices: true,
				ConsumableCapacity:   true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil,
					request(req0, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.attributes["%s"].kind == "shared"`, driverA),
						}}),
					request(req1, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.attributes["%s"].kind == "counteronly"`, driverA),
						}}),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					// device1 allows multiple allocations and consumes the single
					// shared counter. It is listed first, so req0 reserves its counter
					// on the first try.
					device(device1).
						withAttribute("kind", "shared").
						withAllowMultipleAllocations().
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
						),
					// device2 consumes no counter and is the correct choice for req0
					// once device1 has been backtracked.
					device(device2).
						withAttribute("kind", "shared"),
					// device3 is the only device that satisfies req1 and needs the same
					// counter device1 reserved.
					device(device3).
						withAttribute("kind", "counteronly").
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
						),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
				),
			),
			node: node(node1, region1),
			// device1 allows multiple allocations and reserves the single shared
			// counter when req0 tries it first. req1 can only use device3, which needs
			// that same counter, so the search must backtrack device1. The backtracking
			// undo has to release device1's counter reservation; otherwise req0 cannot
			// fall back to device2 and req1 cannot take device3.
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req1, driverA, pool1, device3, false),
			)},
		},
		"consumable-capacity-rejection-rolls-back-constraints": {
			features: Features{
				ConsumableCapacity: true,
				PrioritizedList:    true,
			},
			claimsToAllocate: objects(
				// claim0 consumes all of device1's capacity first. Its request is not
				// subject to claim1's constraint, so it does not set the reference value.
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1).
						withCapacityRequest(capacity0, two).
						withSelectors(resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].stringAttribute == "cap"`, driverA),
							}}),
				),
				// claim1 has a match-attribute constraint. Its first (all-mode)
				// subrequest includes device1, which was capacity-sufficient when the
				// candidate list was built but is now full, so it is rejected on the
				// in-function capacity check after being added to the constraint. The
				// fallback subrequest then needs device2, whose attribute differs.
				claim(claim1).
					withConstraints(resourceapi.DeviceConstraint{MatchAttribute: &stringAttribute}).
					withRequests(requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].stringAttribute == "cap"`, driverA),
							}}).
							withAllocationMode(resourceapi.DeviceAllocationModeAll).
							withCapacityRequest(capacity0, two),
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].stringAttribute == "fb"`, driverA),
							}}),
					)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).
						withCapacity(capacity0, two).
						withAttribute("stringAttribute", "cap").
						withAllowMultipleAllocations(),
					device(device2).
						withAttribute("stringAttribute", "fb"),
				),
			),
			node: node(node1, region1),
			// claim1's all-mode subReq0 adds device1 to the match constraint and is then
			// rejected because claim0 already consumed device1's capacity. Rolling back
			// that rejected candidate's constraint entry is what lets the fallback
			// subReq1 take device2; otherwise device1's leaked attribute value blocks it.
			expectResults: []any{
				allocationResult(localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).
						withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two})),
				allocationResult(localNodeSelector(node1),
					deviceAllocationResult(req0SubReq1, driverA, pool1, device2, false)),
			},
		},
		"partitionable-empty-capacity-multiple-share-backtrack-keeps-counter-ownership": {
			features: Features{
				PartitionableDevices: true,
				ConsumableCapacity:   true,
			},
			claimsToAllocate: objects(
				// claim0 pins device1 as the first share and reserves its single
				// shared counter.
				claimWithRequests(claim0, nil,
					request(req0, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.attributes["%s"].kind == "shared"`, driverA),
						}}),
				),
				// claim1's req1 first tries device1 as a second share (skipping the
				// counter, since claim0 already reserved it), but the match-attribute
				// constraint with req2 forces it to backtrack onto device2. req3 then
				// shares device1 again, after req1 has already left it.
				claimWithRequests(claim1,
					[]resourceapi.DeviceConstraint{
						{MatchAttribute: &stringAttribute, Requests: []string{req1, req2}},
					},
					request(req1, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.attributes["%s"].kind == "shared" || device.attributes["%s"].kind == "fallback"`, driverA, driverA),
						}}),
					request(req2, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.attributes["%s"].kind == "gate"`, driverA),
						}}),
					request(req3, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.attributes["%s"].kind == "shared"`, driverA),
						}}),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					// device1: allow-multiple, no capacity, consumes the single counter.
					// kind steers the selectors; stringAttribute steers the constraint.
					device(device1).
						withAttribute("kind", "shared").
						withAttribute("stringAttribute", "red").
						withAllowMultipleAllocations().
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
						),
					// device2: the fallback for req1, matching req2's stringAttribute.
					device(device2).
						withAttribute("kind", "fallback").
						withAttribute("stringAttribute", "blue"),
					// device3: the only device req2 can take.
					device(device3).
						withAttribute("kind", "gate").
						withAttribute("stringAttribute", "blue"),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{"c": resource.MustParse("1")}),
				),
			),
			node: node(node1, region1),
			// claim0 reserves device1's counter as the first share. claim1's req1 tries
			// device1 as a second share, skips the counter, then backtracks onto device2
			// because of the constraint with req2. Rolling back that empty-capacity share
			// must not drop device1's shared marker, because claim0's share still holds
			// the counter. req3 then shares device1 again without recharging the counter.
			expectResults: []any{
				allocationResult(localNodeSelector(node1),
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, nil)),
				allocationResult(localNodeSelector(node1),
					deviceAllocationResult(req1, driverA, pool1, device2, false),
					deviceAllocationResult(req2, driverA, pool1, device3, false),
					deviceRequestAllocationResult(req3, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, nil)),
			},
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
					device(device1).withCapacity("memory", "4Gi").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2).withCapacity("memory", "6Gi").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("6Gi"),
							},
						),
					),
					device(device3).withCapacity("memory", "4Gi").withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("6Gi"),
							},
						),
					),
					device(device3).withDeviceCounterConsumption(
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
					device(device1).withCapacity("memory", "4Gi").withDeviceCounterConsumption(
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
					device(device2).withCapacity("memory", "6Gi").withDeviceCounterConsumption(
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
					device(device3).withCapacity("memory", "4Gi").withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
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
					device(device2).withDeviceCounterConsumption(
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
					device(device3).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("6"),
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("4"),
								"memory": resource.MustParse("6Gi"),
							},
						),
					),
					device(device3).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("6"),
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"cpus":   resource.MustParse("1"),
								"memory": resource.MustParse("7Gi"),
							},
						),
					),
					device(device3).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("5Gi"),
							},
						),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device3).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2),
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
					device(device2).withNodeSelection(node1),
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
					device(device1).withCapacity("memory", "4Gi").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withNodeSelection(node1),
					device(device2).withCapacity("memory", "6Gi").withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withNodeSelection(nodeLabelSelector(regionKey, region1)),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withNodeSelection(node1),
					device(device3).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					),
					device(device3).withDeviceCounterConsumption(
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
				device(device1).withTaints(taintNoSchedule),
				device(device2).withTaints(taintNoExecute),
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
				device(device1).withTaints(taintNoSchedule, taintNoExecute),
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
				device(device1).withTaints(taintNoSchedule),
				device(device2).withTaints(taintNoExecute),
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
				device(device1).withTaints(taintNoSchedule),
				device(device2).withTaints(taintNoExecute),
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
				device(device1).withTaints(taintNone),
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
				device(device1).withTaints(taintNoSchedule, taintNoExecute),
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
				device(device1).withTaints(taintNoSchedule, taintNoExecute),
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
				device(device1).withTaints(taintNoSchedule),
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
				device(device1).withTaints(taintNoSchedule),
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
				device(device1).withTaints(taintNoSchedule),
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
				device(device1).withTaints(taintNoSchedule),
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
				device(device1).withTaints(taintNoSchedule),
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
				device(device1).withTaints(taintNoSchedule),
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
				device(device1),
				device(device2),
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
		"prioritized-list-allocation-mode-all-with-selectors": {
			// When using allocationMode:All, at least one device must be allocated, otherwise
			// DRA will attempt the next subRequest.
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 0, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].generation == "v6"`, driverA),
							},
						}).withAllocationMode(resourceapi.DeviceAllocationModeAll),
						subRequest(subReq1, classA, 0, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].generation == "v5"`, driverA),
							},
						}).withAllocationMode(resourceapi.DeviceAllocationModeAll),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAttribute("generation", "v5"),
				),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device1, false),
			)},
		},
		"prioritized-list-allocation-mode-all-with-selectors-multiple-pools": {
			// All devices selected by the DeviceClass will be allocated, even if they
			// are spread across different resource pools.
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 0, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].generation == "v6"`, driverA),
							},
						}).withAllocationMode(resourceapi.DeviceAllocationModeAll),
						subRequest(subReq1, classA, 0, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].generation == "v5"`, driverA),
							},
						}).withAllocationMode(resourceapi.DeviceAllocationModeAll),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAttribute("generation", "v5"),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device1).withAttribute("generation", "v5"),
				),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0SubReq1, driverA, pool1, device1, false),
					deviceAllocationResult(req0SubReq1, driverA, pool2, device1, false),
				),
			},
		},
		"prioritized-list-allocation-mode-all-with-selectors-multiple-pools-across-nodes": {
			// All devices selected by the DeviceClass will be allocated, even if they
			// are spread across different resource pools, as long as they are available
			// on the node.
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 0, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].generation == "v6"`, driverA),
							},
						}).withAllocationMode(resourceapi.DeviceAllocationModeAll),
						subRequest(subReq1, classA, 0, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].generation == "v5"`, driverA),
							},
						}).withAllocationMode(resourceapi.DeviceAllocationModeAll),
					),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAttribute("generation", "v5"),
				),
				sliceWithDevices(slice2, node2, pool2, driverA,
					device(device1).withAttribute("generation", "v5"),
					device(device2).withAttribute("generation", "v6"),
				),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0SubReq1, driverA, pool1, device1, false),
				),
			},
		},
		"prioritized-list-allocation-mode-all-with-selectors-deviceclass-subset": {
			// Only devices selected by the deviceclass will be considered
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					requestWithPrioritizedList(req0,
						subRequest(subReq0, classA, 0, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].generation == "v6"`, driverA),
							},
						}).withAllocationMode(resourceapi.DeviceAllocationModeAll),
						subRequest(subReq1, classA, 0, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].generation == "v5"`, driverA),
							},
						}).withAllocationMode(resourceapi.DeviceAllocationModeAll),
					),
				),
			),
			classes: objects(
				&resourceapi.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{
						Name: classA,
					},
					Spec: resourceapi.DeviceClassSpec{
						Selectors: []resourceapi.DeviceSelector{
							{
								CEL: &resourceapi.CELDeviceSelector{
									Expression: fmt.Sprintf(`device.driver == "%s" && device.attributes["%s"].selected`, driverA, driverA),
								},
							},
						},
					},
				},
			),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).
						withAttribute("generation", "v6").
						withAttribute("selected", false),
					device(device2).
						withAttribute("generation", "v5").
						withAttribute("selected", false),
					device(device3).
						withAttribute("generation", "v5").
						withAttribute("selected", true),
				),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					deviceAllocationResult(req0SubReq1, driverA, pool1, device3, false),
				),
			},
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
					device(device1).
						withAttribute("special", true).
						withDeviceCounterConsumption(
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
					device(device2).
						withAttribute("special", false).
						withDeviceCounterConsumption(
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
					device(device3).
						withAttribute("special", false).
						withDeviceCounterConsumption(
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
				device(device1).withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}))),
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
				device(device1).withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
				device(device2).withBindingConditions([]string{"IsPrepare2"}, []string{"BindingFailed2"}),
				device(device3).withBindingConditions([]string{"IsPrepare3"}, []string{"BindingFailed3"}),
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
				device(device1),
				device(device2).withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
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
				device(device1).withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}))),
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
				device(device1).withBindsToNode(true))),
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
					device(device1).
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
					device(device1).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
					device(device2).
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
					device(device1).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("2Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
					device(device2).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("2Gi"),
							}),
						),
					device(device3).
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
				sliceWithDevices(slice4, node1, resourcePool(pool1, 4), driverA,
					device(device1).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						).
						withBindingConditions([]string{"IsPrepare"}, []string{"BindingFailed"}),
				),
				sliceWithCounterSets(slice3, node1, resourcePool(pool1, 4), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 4), driverA,
					device(device2).
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet2, map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							}),
						),
				),
				sliceWithCounterSets(slice1, node1, resourcePool(pool1, 4), driverA,
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
					device(device1).
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
					device(device2).
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
					device(device1).
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
					device(device2).
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
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
						subRequest(subReq0, classA, 1).withCapacityRequest(capacity0, one),
						subRequest(subReq1, classA, 1).withCapacityRequest(capacity0, one),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity1, one).withAllowMultipleAllocations(),
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
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity1, one).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"consumable-capacity-request-unqualified-and-unqualified-device-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"consumable-capacity-request-unqualified-and-qualified-device-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(driverA+"/"+capacity0, two),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"consumable-capacity-request-qualified-and-unqualified-device-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(driverA+"/"+capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"consumable-capacity-request-unqualified-and-foreign-domain-only-device-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity("foo/"+capacity0, two).withCapacity("bar/"+capacity0, two),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		"consumable-capacity-request-foreign-domain-and-foreign-domain-device-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest("foo/"+capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity("foo/"+capacity0, two),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		"consumable-capacity-request-foreign-domain-and-driver-device-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest("foo/"+capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		// A device can, in principle, publish the same capacity both without a domain and
		// explicitly qualified with its own driver as domain (nothing rejects that on
		// admission). For consistency with attribute and capacity lookup the entry
		// with domain wins.
		"consumable-capacity-request-unqualified-and-device-with-unqualified-and-qualified-capacity-names-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, two)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, one).withCapacity(driverA+"/"+capacity0, four),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
			)},
		},
		// Same as above, but with the values swapped: the explicitly qualified entry is now
		// the too-small one, and takes precedence over the implicit one that would otherwise
		// have satisfied the request.
		"consumable-capacity-request-unqualified-and-device-with-qualified-and-unqualified-capacity-names-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, two)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, four).withCapacity(driverA+"/"+capacity0, one),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		// A claim's request can, in principle, name the same capacity both without a domain
		// and explicitly qualified with the driver as domain (nothing rejects that on
		// admission). Unlike a device's published capacity above, this is ambiguous on the
		// requester's side and is treated as a scheduling failure rather than resolved by
		// preferring one of the two conflicting values.
		"consumable-capacity-request-with-unqualified-and-qualified-capacity-request-names-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one).withCapacityRequest(resourceapi.QualifiedName(driverA+"/"+string(capacity0)), four)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("requested both as")),
		},
		// Same as above, but with the values swapped. Still ambiguous, so still an error.
		"consumable-capacity-request-with-qualified-and-unqualified-capacity-request-names-exclusive": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, four).withCapacityRequest(resourceapi.QualifiedName(driverA+"/"+string(capacity0)), one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("requested both as")),
		},
		"consumable-capacity-request-unqualified-and-unqualified-device-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
			)},
		},
		"consumable-capacity-request-unqualified-and-qualified-device-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(driverA+"/"+capacity0, two).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{driverA + "/" + capacity0: one}),
			)},
		},
		"consumable-capacity-request-qualified-and-unqualified-device-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(driverA+"/"+capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}),
			)},
		},
		"consumable-capacity-request-unqualified-and-foreign-domain-only-device-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity("foo/"+capacity0, two).withCapacity("bar/"+capacity0, two).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		"consumable-capacity-request-foreign-domain-and-foreign-domain-device-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest("foo/"+capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity("foo/"+capacity0, two).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{"foo/" + capacity0: one}),
			)},
		},
		"consumable-capacity-request-foreign-domain-and-driver-device-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest("foo/"+capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		// See the exclusive-device variant above for why this is legal input.
		"consumable-capacity-request-unqualified-and-device-with-unqualified-and-qualified-capacity-names-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, two)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, one).withCapacity(driverA+"/"+capacity0, four).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{driverA + "/" + capacity0: two}),
			)},
		},
		"consumable-capacity-request-unqualified-and-device-with-qualified-and-unqualified-capacity-names-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, two)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, four).withCapacity(driverA+"/"+capacity0, one).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: nil,
		},
		// A capacity's result name only gains the driver domain when the device
		// published it with that domain (see the two test cases above). A
		// differently named capacity that happens to be qualified must not affect
		// that: the unqualified one here must stay unqualified in the result.
		"consumable-capacity-request-unqualified-and-device-with-unrelated-qualified-capacity-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withCapacity(driverA+"/"+capacity1, four).withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{
					capacity0:                 one,
					driverA + "/" + capacity1: four,
				}),
			)},
		},
		"consumable-capacity-request-with-unqualified-and-qualified-capacity-request-names-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one).withCapacityRequest(resourceapi.QualifiedName(driverA+"/"+string(capacity0)), four)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("requested both as")),
		},
		"consumable-capacity-request-with-qualified-and-unqualified-capacity-request-names-shared": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, four).withCapacityRequest(resourceapi.QualifiedName(driverA+"/"+string(capacity0)), one)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("requested both as")),
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
					device(device1).withAllowMultipleAllocations(),
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
					device(device1).withCapacity(capacity0, one).withAllowMultipleAllocations(),
					device(device2).withCapacity(capacity0, one).withAllowMultipleAllocations(),
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
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
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
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
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
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyValidValues(zero, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyValidValues(zero, map[resourceapi.QualifiedName]resource.Quantity{capacity0: one}, nil),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, *resource.NewQuantity(3, resource.BinarySI))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
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
		"consumable-capacity-range-policy-request-above-int64-rejected": {
			features: Features{
				ConsumableCapacity: true,
			},
			// A capacity request above MaxInt64 cannot be rounded on the integer Step path
			// without reading it through Quantity.Value(), which wraps. The request is not
			// representable, so allocation aborts with a specific error instead of skipping.
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, resource.MustParse("18446744073709551616"))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("cannot be represented")),
		},
		"consumable-capacity-range-policy-rounded-value-overflow-rejected": {
			features: Features{
				ConsumableCapacity: true,
			},
			// Rounding a MaxInt64 request up to the next Step multiple passes MaxInt64 and
			// would wrap to a negative value. The rounded value is not representable, so
			// allocation aborts with a specific error instead of acting on the wrapped read.
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, resource.MustParse("9223372036854775807"))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("cannot be represented")),
		},
		"consumable-capacity-range-policy-overflow-aborts-not-skip": {
			features: Features{
				ConsumableCapacity: true,
			},
			// device-1 has a Step range, so a request above MaxInt64 overflows its integer
			// rounding. device-2 has no Step and could satisfy the same request by exact
			// comparison, but a not-representable request is fail-closed: allocation aborts
			// rather than skipping device-1 to allocate device-2. Aborting on the first
			// overflowing device is what stops the backtracking a user-controlled
			// out-of-range request would otherwise force.
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, resource.MustParse("18446744073709551616"))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
					device(device2).withCapacity(capacity0, "36893488147419103232").withAllowMultipleAllocations(),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("cannot be represented")),
		},
		"consumable-capacity-range-policy-allocation-mode-all-overflow-aborts": {
			features: Features{
				ConsumableCapacity: true,
			},
			// The all-devices pre-scan in validateDeviceRequest validates capacity too. An
			// out-of-range request must abort there, not silently drop the overflowing
			// device from the set and allocate whatever remains. The count-mode cases above
			// exercise allocateOne; this one exercises the AllocationModeAll pre-scan.
			claimsToAllocate: objects(
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, resource.MustParse("18446744073709551616"))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("cannot be represented")),
		},
		"consumable-capacity-negative-request-aborts": {
			features: Features{
				ConsumableCapacity: true,
			},
			// A stored request or slice from before stricter validation could carry a
			// negative capacity, which admission does not re-check on existing objects. A
			// negative value must abort rather than record negative consumed capacity and
			// over-allocate the device.
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, resource.MustParse("-5"))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, "10").withAllowMultipleAllocations(),
				),
			),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("negative")),
		},
		"consumable-capacity-multi-allocatable-device-with-valid-values-policy": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, *resource.NewQuantity(2, resource.BinarySI))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyValidValues(one, map[resourceapi.QualifiedName]resource.Quantity{capacity0: four},
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, *resource.NewQuantity(2, resource.BinarySI))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyValidValues(one, map[resourceapi.QualifiedName]resource.Quantity{capacity0: three},
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, two)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, *resource.NewQuantity(6, resource.BinarySI))),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: *resource.NewQuantity(10, resource.BinarySI)}),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, two)),
			),
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					draapi.FullyQualifiedName{Domain: driverA, Identifier: string(capacity0)}: new(two),
				},
			},
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, two)),
			),
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					draapi.FullyQualifiedName{Domain: driverA, Identifier: string(capacity0)}: new(two),
				},
			},
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
					device(device2).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
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
					deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, two),
					deviceRequest(req1, classA, 1).withCapacityRequest(capacity0, two),
				),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, two)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device2).
						withAttribute("stringAttribute", "stringAttributeValue1").
						withAllowMultipleAllocations().
						withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
					device(device1).
						withAttribute("stringAttribute", "stringAttributeValue2").
						withAllowMultipleAllocations().
						withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
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
						subRequest(subReq0, classA, 1).withCapacityRequest(capacity0, four),
						subRequest(subReq1, classA, 1).withCapacityRequest(capacity0, two),
						subRequest(subReq1, classA, 1),
					),
				),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, two)),
			),
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					draapi.FullyQualifiedName{Domain: driverA, Identifier: string(capacity0)}: new(two),
				},
			},
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
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
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, two)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
					device(device2).withAllowMultipleAllocations(),
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
				claim(claim0).withRequests(allDeviceRequest(req0, classA).withCapacityRequest(capacity0, two)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
					device(device2).withCapacity(capacity0, one).withAllowMultipleAllocations(),
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
					deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one),
					allDeviceRequest(req1, classA).withCapacityRequest(capacity0, one),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
					device(device2).withCapacity(capacity0, one).withAllowMultipleAllocations(),
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
					draapi.FullyQualifiedName{Domain: driverA, Identifier: string(capacity0)}: new(one),
				},
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, two).withAllowMultipleAllocations(),
					device(device2).withCapacity(capacity0, one).withAllowMultipleAllocations(),
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
					deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one),
					allDeviceRequest(req1, classA).withCapacityRequest(capacity0, one),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, one).withAllowMultipleAllocations(),
					device(device2).withCapacity(capacity0, one).withAllowMultipleAllocations(),
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
					draapi.FullyQualifiedName{Domain: driverA, Identifier: string(capacity0)}: new(one),
				},
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					allDeviceRequest(req0, classA).withCapacityRequest(capacity0, one),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, one).withAllowMultipleAllocations(),
					device(device2).withCapacity(capacity0, one).withAllowMultipleAllocations(),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{}, // device-1 capacity-exhausted but still matches; All fails, not device-2 only
		},
		"consumable-capacity-dedicated-device-with-consumable-capacity-request": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, false)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, one),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
				claim(claim1).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, false)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"consumable-capacity-requests-are-not-modified": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, two),
					deviceRequest(req1, classA, 1).withCapacityRequest(capacity0, resource.MustParse("1000000000000000000000")),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).
						withCapacity(capacity0, resourceapi.DeviceCapacity{
							Value: resource.MustParse("1000000000000000000002"),
							RequestPolicy: &resourceapi.CapacityRequestPolicy{
								Default: ptr.To(resource.MustParse("1000000000000000000000")),
								ValidRange: &resourceapi.CapacityRequestPolicyRange{
									Min: &two,
								},
							},
						}).
						withAllowMultipleAllocations(),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
				deviceRequestAllocationResult(req1, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{capacity0: resource.MustParse("1000000000000000000000")}),
			)},
		},
		"allow-multiple-allocations-exclude-multi-allocatable-device-by-class-selector": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, false)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacity(capacity0, one).withAllowMultipleAllocations(),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
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
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one)),
			),
			allocatedCapacityDevices: map[DeviceID]ConsumedCapacity{
				MakeDeviceID(driverA, pool1, device1): {
					draapi.FullyQualifiedName{Domain: driverA, Identifier: string(capacity0)}: new(one),
				},
			},
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
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
					device(device1).withCapacity("memory", "4Gi").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("4Gi"),
							},
						),
					).withAllowMultipleAllocations(),
					device(device2).withCapacity("memory", "6Gi").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								"memory": resource.MustParse("6Gi"),
							},
						),
					).withAllowMultipleAllocations(),
					device(device3).withCapacity("memory", "4Gi").withDeviceCounterConsumption(
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
		"allow-multiple-allocations-with-shared-device-counter-cross-cycle": {
			features: Features{
				PartitionableDevices: true,
				ConsumableCapacity:   true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(
					deviceRequest(req0, classA, 1).
						withCapacityRequest(capacity0, one).
						withSelectors(resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.attributes["%s"].mode == "b"`, driverA),
							},
						}),
				),
			),
			allocatedSharedDeviceIDs: sets.New(
				MakeDeviceID(driverA, pool1, device1),
			),
			allocatedCapacityDevices: ConsumedCapacityCollection{
				MakeDeviceID(driverA, pool1, device1): ConsumedCapacity{
					draapi.FullyQualifiedName{Domain: driverA, Identifier: string(capacity0)}: new(one),
				},
			},
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).
						withAttribute("mode", "a").
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								counter0: one,
							})).
						withAllowMultipleAllocations(),
					device(device2).
						withAttribute("mode", "b").
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								counter0: one,
							})).
						withAllowMultipleAllocations(),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						counter0: one,
					}),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"partitionable-consumable-capacity-shared-device-counter-not-recharged": {
			features: Features{
				PartitionableDevices: true,
				ConsumableCapacity:   true,
			},
			// device1 already has a persisted shared allocation. It is represented
			// only by a share ID, with no AggregatedCapacity entry.
			allocatedSharedDeviceIDs: sets.New(
				MakeDeviceID(driverA, pool1, device1),
			),
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).
						withAllowMultipleAllocations().
						withDeviceCounterConsumption(
							deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
								"c": resource.MustParse("1"),
							}),
						),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"c": resource.MustParse("1"),
					}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, nil),
			)},
		},
		"consumable-capacity-dedicated-request-blocked-by-persisted-share": {
			// device1 still carries a live share after allowMultipleAllocations flipped to false,
			// so a dedicated request must not be handed the device on top of it.
			features: Features{ConsumableCapacity: true},
			allocatedSharedDeviceIDs: sets.New(
				MakeDeviceID(driverA, pool1, device1),
			),
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 1))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 1), driverA,
					device(device1),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"consumable-capacity-dedicated-request-falls-back-past-persisted-share": {
			// The guard drops device1 without aborting the search, so the request still lands on device2.
			features: Features{ConsumableCapacity: true},
			allocatedSharedDeviceIDs: sets.New(
				MakeDeviceID(driverA, pool1, device1),
			),
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 1))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 1), driverA,
					device(device1),
					device(device2),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"consumable-capacity-dedicated-request-falls-back-past-multiple-persisted-shares": {
			// The guard drops device1 and device2 without aborting the search, so the request lands on device3.
			features: Features{ConsumableCapacity: true},
			allocatedSharedDeviceIDs: sets.New(
				MakeDeviceID(driverA, pool1, device1),
				MakeDeviceID(driverA, pool1, device2),
			),
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 1))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 1), driverA,
					device(device1),
					device(device2),
					device(device3),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"consumable-capacity-with-admin-access-request-allowed-over-persisted-share": {
			// Admin access skips the availability checks, so the live share does not withhold device1.
			features: Features{ConsumableCapacity: true, AdminAccess: true},
			allocatedSharedDeviceIDs: sets.New(
				MakeDeviceID(driverA, pool1, device1),
			),
			claimsToAllocate: func() []wrapResourceClaim {
				c := claimWithRequest(claim0, req0, classA)
				c.Spec.Devices.Requests[0].Exactly.AdminAccess = new(true)
				return []wrapResourceClaim{c}
			}(),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 1), driverA,
					device(device1),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, true),
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
					deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one),
					deviceRequest(req2, classA, 1).withCapacityRequest(capacity0, two),
					requestWithPrioritizedList(req1,
						subRequest(subReq0, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"]["%s"].compareTo(quantity("4")) >= 0`, driverA, capacity0),
							}}).withCapacityRequest(capacity0, one), // Redundant, but not wrong.
						subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
							CEL: &resourceapi.CELDeviceSelector{
								Expression: fmt.Sprintf(`device.capacity["%s"]["%s"].compareTo(quantity("2")) >= 0`, driverA, capacity0),
							}}).withCapacityRequest(capacity0, one), // Redundant, but not wrong.
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
					device(device1).withCapacity(capacity0, "2").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								counter0: two,
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								counter1: four,
							},
						),
					).withAllowMultipleAllocations().withCapacityRequestPolicyRange((map[resourceapi.QualifiedName]resource.Quantity{capacity1: four})),
					device(device2).withCapacity(capacity0, "4").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								counter0: four,
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								counter1: four,
							},
						),
					).withAllowMultipleAllocations().withCapacityRequestPolicyRange((map[resourceapi.QualifiedName]resource.Quantity{capacity1: four})),
					device(device3).withCapacity(capacity0, "2").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1,
							map[string]resource.Quantity{
								counter0: two,
							},
						),
						deviceCounterConsumption(counterSet2,
							map[string]resource.Quantity{
								counter1: four,
							},
						),
					).withAllowMultipleAllocations().withCapacityRequestPolicyRange((map[resourceapi.QualifiedName]resource.Quantity{capacity1: four})),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 2), driverA,
					counterSet(counterSet1,
						map[string]resource.Quantity{
							counter0: four,
						},
					),
					counterSet(counterSet2,
						map[string]resource.Quantity{
							counter1: resource.MustParse("8"),
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
					device(device1).
						withAttribute("driverVersion", semver.MustParse("1.0.0")).
						withAttribute("numa", 0).
						withAttribute("boolAttribute", true).
						withAllowMultipleAllocations(),
					device(device2).
						withAttribute("driverVersion", semver.MustParse("1.0.1")).
						withAttribute("numa", 1).
						withAttribute("boolAttribute", false).
						withAllowMultipleAllocations(),
					device(device3).
						withAttribute("driverVersion", semver.MustParse("1.0.2")).
						withAttribute("numa", 2).
						withAttribute("boolAttribute", true).
						withAllowMultipleAllocations(),
					device(device4).
						withAttribute("driverVersion", semver.MustParse("1.0.0")).
						withAttribute("numa", 3).
						withAttribute("boolAttribute", true).
						withAllowMultipleAllocations(),
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
		"distinct-constraint-single-request-count-3-not-distinct": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA)),
			// req0 asks for 3 devices under a DistinctAttribute(stringAttribute)
			// constraint. Three devices exist with stringAttribute values
			// value1, value2, value1 -- only two distinct values.
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", "value2"),
				device(device3).withAttribute("stringAttribute", "value1"),
			)),
			node: node(node1, region1),
			// Expected results:
			//   - [req0] No allocation. Satisfying req0 requires 3 devices with
			//     distinct stringAttribute values, but device1 and device3 both
			//     have "value1", so no set of 3 distinct-valued devices exists.
			expectResults: nil,
		},
		"distinct-constraint-single-request-count-3-all-distinct": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA)),
			// req0 asks for 3 devices under a DistinctAttribute(stringAttribute)
			// constraint. Three devices exist with distinct values
			// value1, value2, value3.
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", "value2"),
				device(device3).withAttribute("stringAttribute", "value3"),
			)),
			node: node(node1, region1),
			// Expected results:
			//   - [req0] All three devices are allocated: their stringAttribute
			//     values (value1, value2, value3) are pairwise distinct, so the
			//     constraint holds.
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"distinct-constraint-two-requests-with-count-not-distinct": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(
					resourceapi.DeviceConstraint{DistinctAttribute: &stringAttribute, Requests: []string{req0, req1}},
				).withRequests(
					deviceRequest(req0, classA, 2),
					deviceRequest(req1, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			// req0 asks for 2 devices and req1 for 1 device; a single
			// DistinctAttribute(stringAttribute) constraint covers both requests,
			// so all 3 allocated devices must be distinct. Three devices exist
			// with values value1, value2, value1 -- only two distinct values.
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", "value2"),
				device(device3).withAttribute("stringAttribute", "value1"),
			)),
			node: node(node1, region1),
			// Expected results:
			//   - No allocation. The constraint spans req0 and req1, so the 3
			//     devices across both requests must be pairwise distinct. Because
			//     device1 and device3 share "value1", 3 distinct-valued devices
			//     cannot be chosen.
			expectResults: nil,
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
					device(device1).withAllowMultipleAllocations(),
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
					device(device1).withAttribute("boolAttribute", resourceapi.DeviceAttribute{}).withAllowMultipleAllocations(),
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
					device(device1).withAttribute("boolAttribute", true).withAllowMultipleAllocations(),
					device(device2).withAttribute("boolAttribute", false).withAllowMultipleAllocations(),
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
		"distinct-constraint-single-request-multiple-devices-attribute-not-set": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0,
					[]resourceapi.DeviceConstraint{{DistinctAttribute: &intAttribute}},
					request(req0, classA, 3),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1),
					device(device2),
					device(device3),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"distinct-constraint-single-request-not-enough-distinct-values": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0,
					[]resourceapi.DeviceConstraint{{DistinctAttribute: &intAttribute}},
					request(req0, classA, 3),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAttribute("numa", 1),
					device(device2).withAttribute("numa", 2),
					device(device3).withAttribute("numa", 2),
					device(device4).withAttribute("numa", 2),
				),
			),
			node:          node(node1, region1),
			expectResults: []any{},
		},
		"distinct-constraint-single-request-multiple-devices-attribute-set": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0,
					[]resourceapi.DeviceConstraint{{DistinctAttribute: &intAttribute}},
					request(req0, classA, 3),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAttribute("numa", 1),
					device(device2).withAttribute("numa", 2),
					device(device3).withAttribute("numa", 1),
					device(device4).withAttribute("numa", 3),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req0, driverA, pool1, device4, false),
			)},
		},
		"distinct-constraint-one-multi-allocatable-device-with-distinct-constraint": {
			features: Features{
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(resourceapi.DeviceConstraint{DistinctAttribute: &stringAttribute}).withRequests(
					deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, one),
					deviceRequest(req1, classA, 1).withCapacityRequest(capacity0, one),
				)),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).
						withAttribute("stringAttribute", "stringAttributeValue").
						withAllowMultipleAllocations().
						withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: two}),
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
					deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, two),
					deviceRequest(req1, classA, 1).withCapacityRequest(capacity0, two),
				),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).
						withAttribute("stringAttribute", "stringAttributeValue1").
						withAllowMultipleAllocations().
						withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
				),
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device2).
						withAttribute("stringAttribute", "stringAttributeValue2").
						withAllowMultipleAllocations().
						withCapacityRequestPolicyRange(map[resourceapi.QualifiedName]resource.Quantity{capacity0: four}),
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

		// Test cases for DRAFractionalCapacityRange feature
		"consumable-capacity-fractional-range-gate-disabled": {
			// FractionalCapacityRange is off: step arithmetic uses Value(), which rounds
			// sub-integer quantities up to 1.  With min=1, step=1, request=1:
			// added=0, n=0 → consumed = min + step*0 = 1 (DecimalSI, from step's format).
			features: Features{
				ConsumableCapacity: true,
				// FractionalCapacityRange intentionally not set
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, pointTwo)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withFractionalCapacityRequestPolicyRange(
						map[resourceapi.QualifiedName]resource.Quantity{capacity0: one},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					// Integer path: Value() of 200m and 100m both round up to 1,
					// so min=1, step=1, added=0 → result = NewQuantity(1, DecimalSI).
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{
						capacity0: *resource.NewQuantity(1, resource.DecimalSI),
					}),
				),
			},
		},
		"consumable-capacity-fractional-range-gate-enabled-round-up": {
			// FractionalCapacityRange on: 250m is between min=200m and next step=300m,
			// so it is rounded up to 300m (milli-arithmetic: (250-200)/100 rounds up to 1 step).
			features: Features{
				ConsumableCapacity:      true,
				FractionalCapacityRange: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withRequests(deviceRequest(req0, classA, 1).withCapacityRequest(capacity0, pointTwoFive)),
			),
			classes: objects(classWithAllowMultipleAllocations(classA, driverA, true)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).withAllowMultipleAllocations().withFractionalCapacityRequestPolicyRange(
						map[resourceapi.QualifiedName]resource.Quantity{capacity0: one},
					),
				),
			),
			node: node(node1, region1),
			expectResults: []any{
				allocationResult(
					localNodeSelector(node1),
					// Milli path: (250m - 200m) / 100m = 0.5 → ceil → 1 step → 200m + 100m = 300m.
					deviceRequestAllocationResult(req0, driverA, pool1, device1).withConsumedCapacity(&fixedShareID, map[resourceapi.QualifiedName]resource.Quantity{
						capacity0: pointThree,
					}),
				),
			},
		},

		// Test cases for DRAListTypeAttributes features (matchAttributes/distinctAttributes)
		"list-attributes-match-constaint-scalar-string-values": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", "value1"),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"list-attributes-match-constaint-list-of-string-values-with-common-elements": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", []string{"value1", "value2", "value3"}),
				device(device2).withAttribute("stringAttribute", []string{"value2", "value3", "value4"}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"list-attributes-match-constaint-list-of-string-values-with-partially-common-elements": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", []string{"value1", "value2"}),
				device(device2).withAttribute("stringAttribute", []string{"value2", "value3"}),
				device(device3).withAttribute("stringAttribute", []string{"value3", "value1"}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-match-constraint-backtrack-needs-intersection-restore": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 3),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", []string{"value1", "value2"}),
				device(device2).withAttribute("stringAttribute", []string{"value1"}),
				device(device3).withAttribute("stringAttribute", []string{"value2"}),
				device(device4).withAttribute("stringAttribute", []string{"value2"}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
				deviceAllocationResult(req0, driverA, pool1, device4, false),
			)},
		},
		"list-attributes-match-constaint-list-of-string-values-without-common-elements": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 1),
				request(req0, classA, 1),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", []string{"value1", "value2"}),
				device(device2).withAttribute("stringAttribute", []string{"value3", "value4"}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-match-constaint-mixed-scalar-and-list-of-string-values-with-common-elements": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 2),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", []string{"value1", "value2"}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"list-attributes-match-constaint-mixed-scalar-and-list-of-string-values-with-partially-common-elements": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", []string{"value1", "value2"}),
				device(device3).withAttribute("stringAttribute", []string{"value2", "value3"}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-match-constaint-mixed-scalar-and-list-of-string-values-without-common-elements": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", []string{"value2", "value3"}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-match-constaint-list-of-int-values-with-common-elements": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &intAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("numa", []int64{0, 1, 2}),
				device(device2).withAttribute("numa", []int64{1, 2, 3}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		// KEP-6072: standardized numaNode attribute, list form. Two devices whose
		// numaNode lists share an element (physical node + same-socket equidistant
		// nodes) are co-placed by matchAttribute set intersection.
		"numanode-matchattribute-overlapping-lists-are-co-placed": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &numaNodeAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("resource.kubernetes.io/numaNode", []int64{4}),
				device(device2).withAttribute("resource.kubernetes.io/numaNode", []int64{4, 5, 6, 7}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		// KEP-6072: numaNode lists with no common element (different NUMA domains)
		// cannot satisfy matchAttribute, so no allocation is made.
		"numanode-matchattribute-disjoint-lists-are-not-allocated": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &numaNodeAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("resource.kubernetes.io/numaNode", []int64{0}),
				device(device2).withAttribute("resource.kubernetes.io/numaNode", []int64{4, 5, 6, 7}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-match-constaint-list-of-bool-values-with-common-elements": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &boolAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("boolAttribute", []bool{true, false}),
				device(device2).withAttribute("boolAttribute", []bool{true}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"list-attributes-match-constaint-list-of-version-values-with-common-elements": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &versionAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("driverVersion", []semver.Version{semver.MustParse("1.0.0"), semver.MustParse("1.1.0"), semver.MustParse("2.0.0")}),
				device(device2).withAttribute("driverVersion", []semver.Version{semver.MustParse("1.1.0"), semver.MustParse("2.0.0"), semver.MustParse("2.1.0")}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"list-attributes-match-constaint-with-empty-or-unknown-type": {
			features: Features{
				ListTypeAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", resourceapi.DeviceAttribute{}),
				device(device2).withAttribute("stringAttribute", []string{"value"}),
			)),
			node:          node(node1, region1),
			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("unsupported attribute value")),
		},
		"list-attributes-disabled-match-constraint-with-lists": {
			features: Features{
				ListTypeAttributes: false,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", []string{"value1", "value2", "value3"}),
				device(device2).withAttribute("stringAttribute", []string{"value2", "value3", "value4"}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-distinct-constraint-scalar-string-values-all-distinct": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", "value2"),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"list-attributes-distinct-constraint-scalar-string-values-not-distinct": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", "value1"),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-distinct-constraint-single-request-count-3-not-distinct": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA)),
			// Same scenario as distinct-constraint-single-request-count-3-not-distinct
			// but with the ListTypeAttributes feature enabled: req0 asks for 3
			// devices under a DistinctAttribute constraint, and the 3 devices carry
			// values value1, value2, value1 -- only two distinct values.
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", "value2"),
				device(device3).withAttribute("stringAttribute", "value1"),
			)),
			node: node(node1, region1),
			// Expected results:
			//   - [req0] No allocation. device1 and device3 both have "value1",
			//     so 3 devices with distinct stringAttribute values cannot be found.
			expectResults: nil,
		},
		"list-attributes-distinct-constraint-single-request-count-3-all-distinct": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA)),
			// Same scenario as distinct-constraint-single-request-count-3-all-distinct
			// but with the ListTypeAttributes feature enabled: req0 asks for 3
			// devices under a DistinctAttribute constraint, and the 3 devices carry
			// distinct values value1, value2, value3.
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", "value2"),
				device(device3).withAttribute("stringAttribute", "value3"),
			)),
			node: node(node1, region1),
			// Expected results:
			//   - [req0] All three devices are allocated: their stringAttribute
			//     values (value1, value2, value3) are pairwise distinct.
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"list-attributes-distinct-constraint-list-of-string-values-all-distinct": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", []string{"value1", "value2"}),
				device(device2).withAttribute("stringAttribute", []string{"value2", "value3"}),
				device(device3).withAttribute("stringAttribute", []string{"value3", "value4"}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"list-attributes-distinct-constraint-list-of-string-values-with-common-elements": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", []string{"value1", "value2", "value3"}),
				device(device2).withAttribute("stringAttribute", []string{"value2", "value4", "value5"}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-distinct-constraint-mixed-scalar-and-list-all-distinct": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", []string{"value1", "value2"}),
				device(device3).withAttribute("stringAttribute", []string{"value2", "value3"}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"list-attributes-distinct-constraint-mixed-scalar-and-list-with-common-elements": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", "value1"),
				device(device2).withAttribute("stringAttribute", []string{"value1", "value2"}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-distinct-constraint-list-of-int-values-all-distinct": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &intAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("numa", []int64{0, 1}),
				device(device2).withAttribute("numa", []int64{1, 2}),
				device(device3).withAttribute("numa", []int64{2, 3}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"list-attributes-distinct-constraint-list-of-bool-values-all-distinct": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &boolAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("boolAttribute", []bool{true}),
				device(device2).withAttribute("boolAttribute", []bool{true, false}),
				device(device3).withAttribute("boolAttribute", []bool{false}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"list-attributes-distinct-constraint-list-of-version-values-all-distinct": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &versionAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("driverVersion", []string{"1.0.0", "1.1.0"}),
				device(device2).withAttribute("driverVersion", []string{"1.1.0", "2.0.0"}),
				device(device3).withAttribute("driverVersion", []string{"2.0.0", "2.1.0"}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
		"list-attributes-distinct-constraint-with-empty-or-unknown-type": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &versionAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("driverVersion", resourceapi.DeviceAttribute{}),
				device(device2).withAttribute("driverVersion", semver.MustParse("1.0.0")),
			)),
			node: node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("unsupported attribute value")),
		},
		"list-attributes-disabled-distinct-constraint-with-lists": {
			features: Features{
				ListTypeAttributes: false,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{DistinctAttribute: &stringAttribute}},
				request(req0, classA, 2)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", []string{"value1", "value2", "value3"}),
				device(device2).withAttribute("stringAttribute", []string{"value2", "value3", "value4"}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"list-attributes-distinct-constraint-backtrack-needs-remove": {
			features: Features{
				ListTypeAttributes: true,
				ConsumableCapacity: true,
			},
			claimsToAllocate: objects(
				claim(claim0).withConstraints(
					resourceapi.DeviceConstraint{
						Requests:          []string{req0, req1, req2},
						DistinctAttribute: &stringAttribute,
					},
				).withRequests(
					deviceRequest(req0, classA, 1),
					deviceRequest(req1, classA, 1),
					deviceRequest(req2, classA, 1),
				),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("stringAttribute", []string{"value1", "value2", "value3"}),
				device(device2).withAttribute("stringAttribute", []string{"value3", "value4", "value5"}),
				device(device3).withAttribute("stringAttribute", []string{"value1", "value2"}),
				device(device4).withAttribute("stringAttribute", []string{"value3", "value4"}),
				device(device0).withAttribute("stringAttribute", []string{"value5", "value6"}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
				deviceAllocationResult(req1, driverA, pool1, device4, false),
				deviceAllocationResult(req2, driverA, pool1, device0, false),
			)},
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
					device(device1),
				),
				sliceWithDevices(slice2, node2, resourcePool(pool1, 2), driverA,
					device(device2),
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
						device(device1),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node2, pool1, driverA,
						device(device2),
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
						device(device1),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node1, pool1, driverA,
						device(device2),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				// Node 2, generation 2.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node2, pool1, driverA,
						device(device1),
					)
					s.Spec.Pool.Generation = 2
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node2, pool1, driverA,
						device(device2),
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
						device(device1),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node2, pool1, driverA,
						device(device2),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				// Node 1, generation 2.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node1, pool1, driverA,
						device(device1),
					)
					s.Spec.Pool.Generation = 2
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node1, pool1, driverA,
						device(device2),
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
						device(device1),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node1, pool1, driverA,
						device(device2),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				// Node 2, generation 2.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node2, pool1, driverA,
						device(device1),
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
						device(device1),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice1, node2, pool1, driverA,
						device(device2),
					)
					s.Spec.Pool.Generation = 1
					s.Spec.Pool.ResourceSliceCount = 2
					return s
				}(),
				// Node 1, generation 2.
				func() wrapResourceSliceWithDevices {
					s := sliceWithDevices(slice2, node1, pool1, driverA,
						device(device1),
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
					device(device1),
				),
				sliceWithCounterSets(slice2, node1, resourcePool(pool1, 4), driverA,
					counterSet(counterSet1, nil),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool1, 4), driverA,
					device(device2),
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
					device(device1).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
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
					device(device1).withDeviceCounterConsumption(
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
					device(device1).withCapacity("memory", "6Gi").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("6Gi"),
						}),
					),
					device(device2).withCapacity("memory", "2Gi").withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("2Gi"),
						}),
					),
				),
				sliceWithDevices(slice3, node2, resourcePool(pool1, 3), driverA,
					device(device3).withDeviceCounterConsumption(
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
		"partitionable-devices-devices-consume-counter-set-in-different-nodes": {
			features: Features{
				PartitionableDevices: true,
			},
			claimsToAllocate: objects(
				claimWithRequests(claim0, nil, request(req0, classA, 1)),
			),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithCounterSets(slice1, node1, resourcePool(pool1, 3), driverA,
					counterSet(counterSet1, map[string]resource.Quantity{
						"memory": resource.MustParse("8Gi"),
					}),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 3), driverA,
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("6Gi"),
						}),
					),
				),
				sliceWithDevices(slice3, node2, resourcePool(pool1, 3), driverA,
					device(device2).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("6Gi"),
						}),
					),
					device(device3).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("2Gi"),
						}),
					),
				),
			),
			allocatedDevices: []DeviceID{
				MakeDeviceID(driverA, pool1, device1),
			},
			node: node(node2, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node2),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
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
					device(device1).withDeviceCounterConsumption(
						deviceCounterConsumption(counterSet1, map[string]resource.Quantity{
							"memory": resource.MustParse("6Gi"),
						}),
					),
				),
				sliceWithDevices(slice3, node2, resourcePool(pool1, 3), driverA,
					device(device3).withDeviceCounterConsumption(
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
					device(device1),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 2), driverA,
					device(device2),
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
					device(device1),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 2), driverA,
					device(device1),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool2, 1), driverA,
					device(device2),
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
					device(device1),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 2), driverA,
					device(device1),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool2, 1), driverA,
					device(device2),
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
					device(device1),
				),
			),
			node: node(node1, region1),
		},
		"check-combinations-within-single-request-single-pool-single-slice": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil,
				request(req0, classA, 5),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1),
				device(device2),
				device(device3),
				device(device4),
			)),
			node:                            node(node1, region1),
			expectResults:                   nil,
			expectNumAllocateOneInvocations: 16,
			expectNumAllocateOneInvocationsByChannel: map[internal.AllocatorChannel]int64{
				internal.Stable: 65,
			},
		},
		"check-combinations-within-single-request-single-pool-multiple-slices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil,
				request(req0, classA, 5),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1),
					device(device2),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 2), driverA,
					device(device3),
					device(device4),
				),
			),
			node:                            node(node1, region1),
			expectResults:                   nil,
			expectNumAllocateOneInvocations: 16,
			expectNumAllocateOneInvocationsByChannel: map[internal.AllocatorChannel]int64{
				internal.Stable: 65,
			},
		},
		"check-combinations-within-single-request-multiple-pools-multiple-slices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil,
				request(req0, classA, 5),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 2), driverA,
					device(device2),
				),
				sliceWithDevices(slice3, node1, pool2, driverA,
					device(device3),
					device(device4),
				),
			),
			node:                            node(node1, region1),
			expectResults:                   nil,
			expectNumAllocateOneInvocations: 16,
			expectNumAllocateOneInvocationsByChannel: map[internal.AllocatorChannel]int64{
				internal.Stable: 65,
			},
		},
		"check-combinations-within-single-request-many-pools": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil,
				request(req0, classA, 5),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1),
				),
				sliceWithDevices(slice2, node1, pool2, driverA,
					device(device2),
				),
				sliceWithDevices(slice3, node1, pool3, driverA,
					device(device3),
				),
				sliceWithDevices(slice4, node1, pool4, driverA,
					device(device4),
				),
			),
			node:                            node(node1, region1),
			expectResults:                   nil,
			expectNumAllocateOneInvocations: 16,
			expectNumAllocateOneInvocationsByChannel: map[internal.AllocatorChannel]int64{
				internal.Stable: 65,
			},
		},
		"check-combinations-with-backtracking": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				nil,
				// req-1 needs two generic devices.
				request(req1, classA, 2),
				// req-2 needs a specific device.
				request(req2, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.attributes["%s"].type == "X"`, driverA),
					},
				}),
			)),
			classes: objects(class(classA, driverA)),
			// The order of devices is chosen such that the allocator
			// will initially pick {device1, device2} for req1. This will fail
			// because req-2 needs device1. The allocator has to
			// backtrack. The correct solution is {device2, device3} for req-1
			// and {device1} for req-2. The optimized allocator avoids
			// testing {device2, device1} for req-1 and thus finds the solution
			// faster.
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("type", "X"),
				device(device2).withAttribute("type", "Y"),
				device(device3).withAttribute("type", "Y"),
			)),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req1, driverA, pool1, device2, false),
				deviceAllocationResult(req1, driverA, pool1, device3, false),
				deviceAllocationResult(req2, driverA, pool1, device1, false),
			)},
			expectNumAllocateOneInvocations: 12,
			expectNumAllocateOneInvocationsByChannel: map[internal.AllocatorChannel]int64{
				internal.Stable: 14,
			},
		},
		"check-combinations-with-backtracking-across-slices-and-pools": {
			features: Features{
				DeviceBindingAndStatus: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				nil,
				// req-1 needs two generic devices.
				request(req1, classA, 2),
				// req-2 needs a specific device.
				request(req2, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.attributes["%s"].type == "X"`, driverA),
					},
				}),
			)),
			classes: objects(class(classA, driverA)),
			// The order of devices is chosen such that the allocator
			// will initially pick {device1, device2} for req1. This will fail
			// because req-2 needs device1. The allocator has to
			// backtrack. The correct solution is {device2, device3} for req-1
			// and {device1} for req-2. The optimized allocator avoids
			// testing {device2, device1} for req-1 and thus finds the solution
			// faster.
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 2), driverA,
					device(device1).withAttribute("type", "X"),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 2), driverA,
					device(device2).withAttribute("type", "Y"),
				),
				// Use a binding condition here to make sure pool2 is searched after pool1 when
				// trying to allocate devices. This makes sure we see the same results every time.
				sliceWithDevices(slice3, node1, pool2, driverA,
					device(device3).
						withAttribute("type", "Y").
						withBindingConditions([]string{"IsPrepare"}, []string{}),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req1, driverA, pool1, device2, false),
				deviceRequestAllocationResultWithBindingConditions(req1, driverA, pool2, device3, []string{"IsPrepare"}, []string{}),
				deviceAllocationResult(req2, driverA, pool1, device1, false),
			)},
			expectNumAllocateOneInvocations: 12,
			expectNumAllocateOneInvocationsByChannel: map[internal.AllocatorChannel]int64{
				internal.Stable: 14,
			},
		},
		"check-combinations-with-prioritized-list": {
			features: Features{
				PrioritizedList: true,
			},
			claimsToAllocate: objects(claim(claim0).withRequests(
				// The first alternative can't be satisfied since there aren't enough devices
				// of type Y, but the allocator will try all devices before finding out.
				requestWithPrioritizedList(req1,
					subRequest(subReq0, classA, 3, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.attributes["%s"].type == "Y"`, driverA),
						},
					}),
					subRequest(subReq1, classA, 2),
				),
				// The first subrequest asks for too many devices, so the second
				// have to be chosen.
				requestWithPrioritizedList(req2,
					subRequest(subReq0, classA, 2),
					subRequest(subReq1, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.attributes["%s"].type == "X"`, driverA),
						},
					}),
				),
			)),
			classes: objects(class(classA, driverA)),
			// The order of devices is chosen such that the allocator
			// will initially pick {device1, device2} for req1. This will fail
			// because req-2 needs device1. The allocator has to
			// backtrack. The correct solution is {device2, device3} for req-1
			// and {device1} for req-2. The optimized allocator avoids
			// testing {device2, device1} for req-1 and thus finds the solution
			// faster.
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("type", "X"),
				device(device2).withAttribute("type", "Y"),
				device(device3).withAttribute("type", "Y"),
			)),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req1SubReq1, driverA, pool1, device2, false),
				deviceAllocationResult(req1SubReq1, driverA, pool1, device3, false),
				deviceAllocationResult(req2SubReq1, driverA, pool1, device1, false),
			)},
			expectNumAllocateOneInvocations: 26,
			expectNumAllocateOneInvocationsByChannel: map[internal.AllocatorChannel]int64{
				internal.Stable: 32,
			},
		},

		// "derived-attributes" is the primary test case. It mirrors the
		// cross-request hardware alignment example from the KEP.
		// It defines two requests and synthesizes a virtual grouping
		// key ("sharedNumaNode") across their differing
		// attribute names. The allocator must backtrack past the
		// incompatible first device to successfully co-allocate.
		"derived-attributes": {
			features: Features{
				DerivedAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{
					{
						Requests:       []string{req0, req1},
						MatchAttribute: ptr.To(resourceapi.FullyQualifiedName("derived/sharedNumaNode")),
					},
				},
				requestWithDerivedAttributes(
					req0,
					classA,
					1,
					[]resourceapi.DeviceDerivedAttribute{
						{
							Name:       "derived/sharedNumaNode",
							Expression: fmt.Sprintf(`device.attributes["%s"]["numa"]`, driverA),
						},
					},
				),
				requestWithDerivedAttributes(
					req1,
					classB,
					1,
					[]resourceapi.DeviceDerivedAttribute{
						{
							Name:       "derived/sharedNumaNode",
							Expression: fmt.Sprintf(`device.attributes["%s"]["numaNode"]`, driverB),
						},
					},
				),
			)),
			classes: objects(
				class(classA, driverA),
				class(classB, driverB),
			),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device2).withAttribute("numa", "numa-0"),
					device(device1).withAttribute("numa", "numa-1"),
				),
				sliceWithDevices(slice2, node1, pool2, driverB,
					device(device3).withAttribute("numaNode", "numa-1"),
					device(device4).withAttribute("numaNode", "numa-2"),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req1, driverB, pool2, device3, false),
			)},
		},
		// "derived-attributes-with-prioritized-list" verifies that
		// the allocator correctly evaluates derived attributes that are
		// defined differently across alternatives in a prioritized list.
		// It expects the allocator to skip the first alternative (because its
		// derived attribute violates the cross-request match constraint)
		// and successfully allocate using the second alternative which
		// correctly satisfies the constraint with the other request.
		"derived-attributes-with-prioritized-list": {
			features: Features{
				DerivedAttributes: true,
				PrioritizedList:   true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{
					{
						Requests:       []string{req0, req1},
						MatchAttribute: ptr.To(resourceapi.FullyQualifiedName("derived/match")),
					},
				},
				requestWithPrioritizedList(req0,
					// First alternative pulls physical attribute "foo" ("A").
					// This won't match req1 which pulls "baz" ("B").
					subRequest(subReq0, classA, 1, []resourceapi.DeviceDerivedAttribute{
						{
							Name:       resourceapi.FullyQualifiedName("derived/match"),
							Expression: fmt.Sprintf(`device.attributes["%s"]["foo"]`, driverA),
						},
					}),
					// Second alternative pulls physical attribute "bar" ("B").
					// This matches req1.
					subRequest(subReq1, classA, 1, []resourceapi.DeviceDerivedAttribute{
						{
							Name:       resourceapi.FullyQualifiedName("derived/match"),
							Expression: fmt.Sprintf(`device.attributes["%s"]["bar"]`, driverA),
						},
					}),
				).obj(),
				requestWithDerivedAttributes(
					req1,
					classB, // Notice req1 uses classB (driverB)
					1,
					[]resourceapi.DeviceDerivedAttribute{
						{
							Name:       resourceapi.FullyQualifiedName("derived/match"),
							Expression: fmt.Sprintf(`device.attributes["%s"]["baz"]`, driverB),
						},
					},
				),
			)),
			classes: objects(
				class(classA, driverA),
				class(classB, driverB),
			),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1).
						withAttribute("foo", "A").
						withAttribute("bar", "B"),
				),
				sliceWithDevices(slice2, node1, pool2, driverB,
					device(device2).
						withAttribute("baz", "B"),
				),
			),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0SubReq1, driverA, pool1, device1, false),
				deviceAllocationResult(req1, driverB, pool2, device2, false),
			)},
		},
		// "derived-attributes-override" verifies that derived
		// attributes shadow/override physical attributes of the same
		// name. It also proves that this shadowing is strictly
		// request-scoped and does not leak or affect other requests
		// (which continue to fall back to physical attributes).
		"derived-attributes-override": {
			features: Features{
				DerivedAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{
					{
						// Require match on "myAttr".
						Requests:       []string{req0, req1},
						MatchAttribute: ptr.To(resourceapi.FullyQualifiedName("dra.example.com/myAttr")),
					},
				},
				requestWithDerivedAttributes(
					req0,
					classA,
					1,
					[]resourceapi.DeviceDerivedAttribute{
						{
							// Overrides physical attribute to "value-2".
							Name:       resourceapi.FullyQualifiedName("dra.example.com/myAttr"),
							Expression: `"value-2"`,
						},
					},
				),
				// Falls back to physical attribute.
				request(req1, classA, 1),
			)),
			classes: objects(class(classA, "dra.example.com")),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, "dra.example.com",
				// Mismatched physical attributes.
				device(device1).withAttribute("myAttr", "value-1"),
				device(device2).withAttribute("myAttr", "value-2"),
			)),
			node: node(node1, region1),
			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				// req0 overridden to "value-2" -> device1. req1 uses physical "value-2" -> device2.
				deviceAllocationResult(req0, "dra.example.com", pool1, device1, false),
				deviceAllocationResult(req1, "dra.example.com", pool1, device2, false),
			)},
		},
		// "derived-attributes-cel-missing-attribute" verifies that
		// CEL runtime evaluation errors (such as looking up a key
		// that isn't defined on a candidate device) are caught
		// immediately. The allocator aborts allocation and
		// propagates the explicit CEL error, preventing silent
		// topology matching failures.
		"derived-attributes-cel-missing-attribute": {
			features: Features{
				DerivedAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{
					{
						Requests:       []string{req0},
						MatchAttribute: ptr.To(resourceapi.FullyQualifiedName("derived/missing")),
					},
				},
				requestWithDerivedAttributes(
					req0,
					classA,
					1,
					[]resourceapi.DeviceDerivedAttribute{
						{
							Name:       resourceapi.FullyQualifiedName("derived/missing"),
							Expression: fmt.Sprintf(`device.attributes["%s"]["missingAttr"]`, driverA),
						},
					},
				),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("existingAttr", "yes"),
			)),
			node:        node(node1, region1),
			expectError: gomega.MatchError(gomega.ContainSubstring("no such key: missingAttr")),
		},
		// "derived-attributes-type-mismatch" verifies that when CEL
		// dynamically evaluates to different value types for the same
		// attribute across candidate devices (e.g., string vs int),
		// the MatchAttribute constraint safely rejects the mismatch
		// rather than panicking or throwing an unhandled error.
		"derived-attributes-type-mismatch": {
			features: Features{
				DerivedAttributes: true,
			},
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{
					{
						Requests:       []string{req0},
						MatchAttribute: ptr.To(resourceapi.FullyQualifiedName("derived/myAttr")),
					},
				},
				requestWithDerivedAttributes(
					req0,
					classA,
					2,
					[]resourceapi.DeviceDerivedAttribute{
						{
							Name:       resourceapi.FullyQualifiedName("derived/myAttr"),
							Expression: fmt.Sprintf(`device.attributes["%s"]["static"]`, driverA),
						},
					},
				),
			)),
			classes: objects(class(classA, driverA)),
			slices: unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
				device(device1).withAttribute("static", "value"),
				device(device2).withAttribute("static", 1),
			)),
			node:          node(node1, region1),
			expectResults: nil,
		},
	}

	sharingAffinityCandidates(testcases)
	sharingAffinityPools(testcases)
	sharingAffinityGateAndState(testcases)
	sharingAffinityGateOff(testcases)
	sharingAffinityObsoleteGeneration(testcases)
	sharingAffinityMetadata(testcases)
	sharingAffinityScope(testcases)

	// Backends that receive the full allocated state must reject stale pools,
	// even when sharing affinity and consumable capacity are disabled.
	for variant, features := range map[string]Features{
		"gate-off":   {},
		"consumable": {AdminAccess: true, ConsumableCapacity: true, PrioritizedList: true},
		"enforced":   {AdminAccess: true, ConsumableCapacity: true, PrioritizedList: true, SharingAffinity: true},
	} {
		const driver = "sharing.example.com"
		for name, stalePools := range map[string]sets.Set[PoolID]{
			"nil":          nil,
			"other-pool":   sets.New(PoolID{Driver: driver, Pool: "other"}),
			"other-driver": sets.New(PoolID{Driver: "other.example.com", Pool: "pool"}),
			"stale-pool":   sets.New(PoolID{Driver: driver, Pool: "pool"}),
		} {
			claim := claimWithRequests("claim", nil, request("req", "network", 1))
			claim.Namespace = "default"
			claim.UID = "claim-uid"
			slice := sliceWithDevices("devices", nodeSelectionAll, resourceapi.ResourcePool{
				Name: "pool", Generation: 1, ResourceSliceCount: 1,
			}, driver, device("device-a"))
			result := resourceapi.DeviceRequestAllocationResult{
				Request: "req", Driver: driver, Pool: "pool", Device: "device-a",
			}
			var configs []resourceapi.DeviceAllocationConfiguration
			if features.ConsumableCapacity {
				slice.Spec.Devices[0].AllowMultipleAllocations = ptr.To(true)
				config := resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
					Driver: driver, Parameters: runtime.RawExtension{Raw: []byte(`{"tenant":"blue"}`)},
				}}
				claim.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{{DeviceConfiguration: config}}
				configs = []resourceapi.DeviceAllocationConfiguration{{
					Source: resourceapi.AllocationConfigSourceClaim, DeviceConfiguration: *config.DeepCopy(),
				}}
				result.ShareID = &fixedShareID
			}
			tc := AllocatorTestCase{
				features:                 features,
				claimsToAllocate:         objects(claim),
				classes:                  objects(class("network", driver)),
				slices:                   unwrapResourceSlices(slice),
				allocatedSharedDeviceIDs: sets.New[DeviceID](),
				allocatedCapacityDevices: internal.NewConsumedCapacityCollection(),
				affinityStates:           AffinityStateCollection{},
				stalePools:               stalePools,
			}
			expectResults := []any{allocationResultWithConfigs(nil,
				[]resourceapi.DeviceRequestAllocationResult{result}, configs)}
			if name != "stale-pool" {
				tc.expectResults = expectResults
			} else {
				// Stable's constructor does not receive sharing-affinity state.
				tc.expectResultsByChannel = map[internal.AllocatorChannel][]any{
					internal.Stable: expectResults,
				}
			}
			testcases["sharing-affinity-stale-pools-"+variant+"-"+name] = tc
		}
	}

	RunTestAllocator(t, supportedFeatures, newAllocator, testcases)

	t.Run("sharing-affinity-gate-off-entry-points", func(t *testing.T) {
		testSharingAffinityGateOffEntryPoints(t, newAllocator)
	})
	for _, tc := range []struct {
		name string
		run  func(*testing.T, allocatorFactory)
	}{
		{"sharing-affinity-allocation", testSharingAffinityAllocation},
		{"sharing-affinity-diagnostics", testSharingAffinityDiagnostics},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !supportedFeatures.SharingAffinity {
				t.Skip("sharing affinity is not supported")
			}
			tc.run(t, newAllocator)
		})
	}

	t.Run("sharing-affinity-state-transport", func(t *testing.T) {
		testAllocatedStateTransport(t, supportedFeatures, newAllocator)
	})

	t.Run("interrupt", func(t *testing.T) {
		for _, name := range []string{"off", "timeout", "deadline", "cancel"} {
			t.Run(name, func(t *testing.T) {
				_, ctx := ktesting.NewTestContext(t)
				g := gomega.NewWithT(t)

				// This testcase is a smaller variant of the one in https://github.com/kubernetes/kubernetes/issues/131730#issuecomment-2873598287.
				// That one took over 30 seconds, this one here only 0.07 seconds.
				// But even that is too long when we interrupt in the near future or
				// even before starting...
				deviceClass := class(classA, driverA)
				classLister := informerLister[resourceapi.DeviceClass]{
					objs: []*resourceapi.DeviceClass{deviceClass},
				}
				claimsToAllocate := unwrap(claimWithRequests(claim0, nil,
					request(req0, classA, 6),
				))
				slices := unwrapResourceSlices(sliceWithDevices(slice1, node1, pool1, driverA,
					device(device1),
					device(device2),
					device(device3),
					device(device4),
					device("device-5"),
				))
				node := node(node1, region1)

				t.Logf("ResourceSlices:\n%s\n\nResourceClaims:\n%s\n\nDeviceClass:\n%s",
					format.Object(slices, 1),
					format.Object(claimsToAllocate, 1),
					format.Object(deviceClass, 1),
				)

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

func RunTestAllocator(t *testing.T,
	supportedFeatures Features,
	newAllocator func(
		ctx context.Context,
		features Features,
		allocateState AllocatedState,
		classLister DeviceClassLister,
		slices []*resourceapi.ResourceSlice,
		celCache *cel.Cache,
	) (Allocator, error),
	testcases map[string]AllocatorTestCase) {
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
			for i, claim := range claimsToAllocate {
				claimsToAllocate[i] = wrapResourceClaim{claim.DeepCopy()}
			}
			allocatedDevices := slices.Clone(tc.allocatedDevices)
			var slices []*resourceapi.ResourceSlice
			if tc.slices != nil {
				slices = make([]*resourceapi.ResourceSlice, len(tc.slices))
				for i, slice := range tc.slices {
					slices[i] = slice.DeepCopy()
				}
			}
			allocatedState := snapshotAllocatedState(AllocatedState{
				AllocatedDevices:         sets.New(allocatedDevices...),
				AllocatedSharedDeviceIDs: tc.allocatedSharedDeviceIDs,
				AggregatedCapacity:       tc.allocatedCapacityDevices,
				DeviceSharingAffinity: SharingAffinityState{
					DeviceStates: tc.affinityStates,
					StalePools:   tc.stalePools,
				},
			})
			before := snapshotAllocatedState(allocatedState)
			t.Logf("ResourceSlices:\n%s\n\nResourceClaims:\n%s\n\nDeviceClasses:\n%s\nallocated capacity:\n%s\nDeviceSharingAffinity:\n%s",
				format.Object(slices, 1),
				format.Object(claimsToAllocate, 1),
				format.Object(classLister.objs, 1),
				format.Object(allocatedState.AggregatedCapacity, 1),
				format.Object(allocatedState.DeviceSharingAffinity, 1),
			)
			allocator, err := newAllocator(ctx, tc.features, allocatedState, classLister, slices, cel.NewCache(1, cel.Features{
				EnableConsumableCapacity: tc.features.ConsumableCapacity,
				EnableListTypeAttributes: tc.features.ListTypeAttributes,
			}))
			g.Expect(err).ToNot(gomega.HaveOccurred())

			if _, ok := allocator.(internal.AllocatorExtended); tc.expectNumAllocateOneInvocations > 0 && !ok {
				t.Skipf("%T does not support the AllocatorStats interface", allocator)
			}
			if tc.node == nil {
				tc.node = node(node1, region1)
			}
			results, err := allocator.Allocate(ctx, tc.node, unwrap(claimsToAllocate...))
			matchError := tc.expectError
			if matchError == nil {
				matchError = gomega.Not(gomega.HaveOccurred())
			}
			g.Expect(err).To(matchError)
			if tc.expectErrorIs != nil {
				g.Expect(err).To(gomega.MatchError(tc.expectErrorIs))
			}

			t.Logf("name: %s", name)
			// replace any share id with fixed value for testing
			for ri, result := range results {
				for ai, allocation := range result.Devices.Results {
					if allocation.ShareID != nil {
						g.Expect(*allocation.ShareID).ToNot(gomega.BeEmpty())
						results[ri].Devices.Results[ai].ShareID = &fixedShareID
					}
					t.Logf("allocated capacity: %v", allocation.ConsumedCapacity)
				}
			}
			expectResults := tc.expectResults
			if override, ok := tc.expectResultsByChannel[allocator.Channel()]; ok {
				expectResults = override
			}
			g.Expect(results).To(gomega.ConsistOf(expectResults...))

			// Objects that the allocator had access to should not have been modified.
			g.Expect(claimsToAllocate).To(gomega.Equal(tc.claimsToAllocate))
			g.Expect(allocatedDevices).To(gomega.HaveExactElements(tc.allocatedDevices))
			g.Expect(slices).To(gomega.Equal(tc.slices))
			g.Expect(classLister.objs).To(gomega.ConsistOf(tc.classes))
			if !equalAllocatedStateTransport(t, allocatedState, before) {
				t.Fatal("allocator modified its input allocated state")
			}

			if expectNumAllocateOneInvocations := tc.expectNumAllocateOneInvocations; expectNumAllocateOneInvocations > 0 {
				stats := allocator.(internal.AllocatorExtended).GetStats()
				if override, ok := tc.expectNumAllocateOneInvocationsByChannel[allocator.Channel()]; ok {
					expectNumAllocateOneInvocations = override
				}
				g.Expect(stats.NumAllocateOneInvocations).To(gomega.Equal(expectNumAllocateOneInvocations))
			}
			if tc.inspectAllocatedState != nil {
				tc.inspectAllocatedState(t, allocatedState)
			}
		})
	}
}

// Each constructor call and its immutability oracle need independent nested
// maps. Capacity retains the runner's existing nil-to-empty normalization.
func snapshotAllocatedState(state AllocatedState) AllocatedState {
	return AllocatedState{
		AllocatedDevices:         cloneSet(state.AllocatedDevices),
		AllocatedSharedDeviceIDs: cloneSet(state.AllocatedSharedDeviceIDs),
		AggregatedCapacity:       state.AggregatedCapacity.Clone(),
		DeviceSharingAffinity:    state.DeviceSharingAffinity.Clone(),
	}
}

func cloneSet[T comparable](s sets.Set[T]) sets.Set[T] {
	if s == nil {
		return nil
	}
	return s.Clone()
}

func testAllocatedStateTransport(t *testing.T, supportedFeatures Features,
	newAllocator func(context.Context, Features, AllocatedState, DeviceClassLister, []*resourceapi.ResourceSlice, *cel.Cache) (Allocator, error)) {
	const driver = "sharing.example.com"
	deviceID := func(name string) DeviceID { return MakeDeviceID(driver, "pool", name) }
	lockedDevice := deviceID("device-9")
	extraDevice := deviceID("device-x")
	slotsKey := draapi.FullyQualifiedName{Domain: driver, Identifier: "slots"}
	insertCapacity := func(capacity ConsumedCapacityCollection, id DeviceID, amount string) {
		capacity.Insert(schedulerapi.NewDeviceConsumedCapacity(id, map[resourceapi.QualifiedName]resource.Quantity{
			"slots": resource.MustParse(amount),
		}))
	}
	for _, shape := range []string{"nil", "empty", "populated"} {
		t.Run(shape, func(t *testing.T) {
			// Build the oracle independently: cloning the fixture here could
			// mask the same shallow-copy defect that this test must detect.
			fixture := func() AllocatorTestCase {
				tc := AllocatorTestCase{
					classes:          objects(class("network", driver)),
					slices:           unwrapResourceSlices(sliceWithDevices("devices", nodeSelectionAll, "pool", driver, device("device-1"))),
					claimsToAllocate: []wrapResourceClaim{claimWithRequests("claim", nil, request("req", "network", 1))},
					expectResults: []any{allocationResult(nil, resourceapi.DeviceRequestAllocationResult{
						Request: "req", Driver: driver, Pool: "pool", Device: "device-1",
					})},
				}
				if shape == "nil" {
					return tc
				}
				tc.affinityStates = AffinityStateCollection{}
				tc.stalePools = sets.New[PoolID]()
				tc.allocatedSharedDeviceIDs = sets.New[DeviceID]()
				tc.allocatedCapacityDevices = internal.NewConsumedCapacityCollection()
				if shape == "populated" {
					tc.allocatedDevices = []DeviceID{deviceID("device-8")}
					tc.allocatedSharedDeviceIDs.Insert(lockedDevice)
					tc.affinityStates[lockedDevice] = DeviceAffinityState{
						Status: internal.AffinityStatusLocked, LockedAffinity: map[string]string{"tenant": "blue"},
					}
					tc.stalePools.Insert(PoolID{Driver: "other.example.com", Pool: "x"})
					insertCapacity(tc.allocatedCapacityDevices, lockedDevice, "3")
				}
				return tc
			}
			tc, pristine := fixture(), fixture()
			stateOf := func(tc AllocatorTestCase) AllocatedState {
				return AllocatedState{
					AllocatedDevices:         sets.New(tc.allocatedDevices...),
					AllocatedSharedDeviceIDs: tc.allocatedSharedDeviceIDs,
					AggregatedCapacity:       tc.allocatedCapacityDevices,
					DeviceSharingAffinity: SharingAffinityState{
						DeviceStates: tc.affinityStates,
						StalePools:   tc.stalePools,
					},
				}
			}
			want := stateOf(pristine)
			// The existing runner always constructs an empty capacity collection.
			if want.AggregatedCapacity == nil {
				want.AggregatedCapacity = internal.NewConsumedCapacityCollection()
			}
			var received [2]AllocatedState
			inspections := 0
			for i, name := range []string{"first", "second"} {
				tc.inspectAllocatedState = func(t *testing.T, state AllocatedState) {
					inspections++
					// Validate shape before any map writes or pointer access.
					if !equalAllocatedStateTransport(t, state, want) {
						t.Fatal("runner did not transport the fixture unchanged; cannot check isolation")
					}
					if i != 0 {
						return
					}
					if state.AllocatedDevices == nil || state.AggregatedCapacity == nil {
						t.Fatal("runner must construct non-nil allocated-device and capacity collections")
					}
					if shape == "populated" {
						lock, ok := state.DeviceSharingAffinity.DeviceStates[lockedDevice]
						if !ok || lock.LockedAffinity == nil {
							t.Fatal("populated fixture lost device-9's nested affinity lock")
						}
						capacity, ok := state.AggregatedCapacity[lockedDevice]
						if !ok {
							t.Fatal("populated fixture lost device-9's consumed capacity")
						}
						q, ok := capacity[slotsKey]
						if !ok || q == nil {
							t.Fatal("populated fixture lost device-9's sharing.example.com/slots quantity")
						}
						if q.Cmp(resource.MustParse("3")) != 0 {
							t.Fatalf("device-9 consumed slots = %s, want 3", q.String())
						}
						lock.LockedAffinity["tenant"] = "mutated"
						q.Add(resource.MustParse("1"))
					}
					state.AllocatedDevices.Insert(extraDevice)
					insertCapacity(state.AggregatedCapacity, extraDevice, "1")
					if shape != "nil" {
						if state.DeviceSharingAffinity.DeviceStates == nil || state.DeviceSharingAffinity.StalePools == nil || state.AllocatedSharedDeviceIDs == nil {
							t.Fatal("non-nil fixture collections became nil")
						}
						state.DeviceSharingAffinity.DeviceStates[extraDevice] = DeviceAffinityState{
							Status: internal.AffinityStatusLocked, LockedAffinity: map[string]string{"tenant": "mutated"},
						}
						state.DeviceSharingAffinity.StalePools.Insert(PoolID{Driver: driver, Pool: "extra"})
						state.AllocatedSharedDeviceIDs.Insert(extraDevice)
					}
				}
				recordingAllocator := func(ctx context.Context, features Features, state AllocatedState, classes DeviceClassLister, slices []*resourceapi.ResourceSlice, cache *cel.Cache) (Allocator, error) {
					received[i] = state
					return newAllocator(ctx, features, state, classes, slices, cache)
				}
				RunTestAllocator(t, supportedFeatures, recordingAllocator, map[string]AllocatorTestCase{name: tc})
			}
			if inspections != 2 {
				t.Errorf("post-assertion inspections = %d, want 2; both runner calls must invoke the inspection hook", inspections)
			}
			if !equalAllocatedStateTransport(t, stateOf(tc), stateOf(pristine)) {
				t.Error("mutating a received snapshot changed the original fixture")
			}
			if !equalAllocatedStateTransport(t, received[1], want) {
				t.Error("second constructor did not receive an independent pristine snapshot")
			}
		})
	}
}

func equalAllocatedStateTransport(t *testing.T, got, want AllocatedState) bool {
	t.Helper()
	equal := true
	for _, field := range []struct {
		name      string
		got, want any
	}{
		{"AllocatedDevices", got.AllocatedDevices, want.AllocatedDevices},
		{"AllocatedSharedDeviceIDs", got.AllocatedSharedDeviceIDs, want.AllocatedSharedDeviceIDs},
		{"DeviceSharingAffinity", got.DeviceSharingAffinity, want.DeviceSharingAffinity},
	} {
		if !reflect.DeepEqual(field.got, field.want) {
			t.Errorf("%s = %#v, want %#v (including nil/empty shape)", field.name, field.got, field.want)
			equal = false
		}
	}
	if (got.AggregatedCapacity == nil) != (want.AggregatedCapacity == nil) || len(got.AggregatedCapacity) != len(want.AggregatedCapacity) {
		t.Errorf("AggregatedCapacity shape = %#v, want %#v", got.AggregatedCapacity, want.AggregatedCapacity)
		equal = false
	}
	for id, wantCapacity := range want.AggregatedCapacity {
		gotCapacity, ok := got.AggregatedCapacity[id]
		if !ok || (gotCapacity == nil) != (wantCapacity == nil) || len(gotCapacity) != len(wantCapacity) {
			t.Errorf("AggregatedCapacity[%s] = %#v, want %#v", id, gotCapacity, wantCapacity)
			equal = false
		}
		for key, wantQuantity := range wantCapacity {
			gotQuantity, ok := gotCapacity[key]
			if !ok || gotQuantity == nil || wantQuantity == nil {
				t.Errorf("AggregatedCapacity[%s][%s] = %v, want non-nil quantity %v", id, key, gotQuantity, wantQuantity)
				equal = false
				continue
			}
			// Quantity's cached string is not part of capacity equality.
			if gotQuantity.Cmp(*wantQuantity) != 0 {
				t.Errorf("AggregatedCapacity[%s][%s] = %s, want %s", id, key, gotQuantity.String(), wantQuantity.String())
				equal = false
			}
		}
	}
	return equal
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
