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

package structured

import (
	"context"
	"errors"
	"fmt"
	"math"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha3"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/cel/environment"
	resourcelisters "k8s.io/client-go/listers/resource/v1alpha3"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/klog/v2"
)

// ClaimLister returns a subset of the claims that a
// resourcelisters.ResourceClaimLister would return.
type ClaimLister interface {
	// ListAllAllocated returns only claims which are allocated.
	ListAllAllocated() ([]*resourceapi.ResourceClaim, error)
}

// Allocator calculates how to allocate a set of unallocated claims which use
// structured parameters.
//
// It needs as input the node where the allocated claims are meant to be
// available and the current state of the cluster (claims, classes, resource
// slices).
type Allocator struct {
	claimsToAllocate []*resourceapi.ResourceClaim
	claimLister      ClaimLister
	classLister      resourcelisters.DeviceClassLister
	sliceLister      resourcelisters.ResourceSliceLister
}

// NewAllocator returns an allocator for a certain set of claims or an error if
// some problem was detected which makes it impossible to allocate claims.
func NewAllocator(ctx context.Context,
	claimsToAllocate []*resourceapi.ResourceClaim,
	claimLister ClaimLister,
	classLister resourcelisters.DeviceClassLister,
	sliceLister resourcelisters.ResourceSliceLister,
) (*Allocator, error) {
	return &Allocator{
		claimsToAllocate: claimsToAllocate,
		claimLister:      claimLister,
		classLister:      classLister,
		sliceLister:      sliceLister,
	}, nil
}

// ClaimsToAllocate returns the claims that the allocated was created for.
func (a *Allocator) ClaimsToAllocate() []*resourceapi.ResourceClaim {
	return a.claimsToAllocate
}

// Allocate calculates the allocation(s) for one particular node.
//
// It returns an error only if some fatal problem occurred. These are errors
// caused by invalid input data, like for example errors in CEL selectors, so a
// scheduler should abort and report that problem instead of trying to find
// other nodes where the error doesn't occur.
//
// In the future, special errors will be defined which enable the caller to
// identify which object (like claim or class) caused the problem. This will
// enable reporting the problem as event for those objects.
//
// If the claims cannot be allocated, it returns nil. This includes the
// situation where the resource slices are incomplete at the moment.
//
// If the claims can be allocated, then it prepares one allocation result for
// each unallocated claim. It is the responsibility of the caller to persist
// those allocations, if desired.
//
// Allocate is thread-safe. If the caller wants to get the node name included
// in log output, it can use contextual logging and add the node as an
// additional value. A name can also be useful because log messages do not
// have a common prefix. V(4) is used for one-time log entries, V(5) for important
// progress reports, and V(6) for detailed debug output.
func (sharedAllocator *Allocator) Allocate(ctx context.Context, node *v1.Node) ([]*resourceapi.AllocationResult, error) {
	alloc := &allocator{
		Allocator:   sharedAllocator,
		ctx:         ctx, // all methods share the same a and thus ctx
		logger:      klog.FromContext(ctx),
		constraints: make([][]constraint, 0, len(sharedAllocator.claimsToAllocate)),
		requestData: make(map[requestIndices]requestData),
		allocated:   make(map[DeviceID]bool),
		result:      make([]*resourceapi.AllocationResult, len(sharedAllocator.claimsToAllocate)),
	}

	// First determine all eligible pools.
	pools, err := GatherPools(ctx, alloc.sliceLister, node)
	if err != nil {
		return nil, fmt.Errorf("gather pool information: %w", err)
	}
	alloc.pools = pools
	if loggerV := alloc.logger.V(6); loggerV.Enabled() {
		loggerV.Info("Gathered pool information", "numPools", len(pools), "pools", pools)
	} else {
		alloc.logger.V(4).Info("Gathered pool information", "numPools", len(pools))
	}

	// We allocate one claim after the other and for each claim, all of
	// its requests. For each individual device we pick one possible
	// candidate after the other, checking constraints as we go.
	// Each chosen candidate is marked as "in use" and the process
	// continues, recursively. This way, all requests get matched against
	// all candidates in all possible orders.
	//
	// The first full solution is chosen.
	//
	// In other words, this is an exhaustive search. This is okay because
	// it aborts early. Once scoring gets added, more intelligence may be
	// needed to avoid trying "equivalent" solutions (two identical
	// requests, two identical devices, two solutions that are the same in
	// practice).

	// This is where we sanity check that we can actually handle the claims
	// and their requests. For each claim we determine how many devices
	// need to be allocated. If not all can be stored in the result, the
	// claim cannot be allocated.
	for claimIndex, claim := range alloc.claimsToAllocate {
		numDevices := 0

		// If we have any any request that wants "all" devices, we need to
		// figure out how much "all" is. If some pool is incomplete, we stop
		// here because allocation cannot succeed. Once we do scoring, we should
		// stop in all cases, not just when "all" devices are needed, because
		// pulling from an incomplete might not pick the best solution and it's
		// better to wait. This does not matter yet as long the incomplete pool
		// has some matching device.
		for requestIndex := range claim.Spec.Devices.Requests {
			request := &claim.Spec.Devices.Requests[requestIndex]
			details := request.DeviceRequestDetails
			if details == nil || details.DeviceClassName == "" {
				// Unknown future request type!
				return nil, fmt.Errorf("claim %s, request %s: unsupported request type", klog.KObj(claim), request.Name)
			}
			for i, selector := range details.Selectors {
				if selector.CEL == nil {
					// Unknown future selector type!
					return nil, fmt.Errorf("claim %s, request %s, selector #%d: unsupported selector type", klog.KObj(claim), request.Name, i)
				}
			}

			// Should be set. If it isn't, something changed and we should refuse to proceed.
			if details.DeviceClassName == "" {
				return nil, fmt.Errorf("claim %s, request %s: missing device class name", klog.KObj(claim), request.Name)
			}
			class, err := alloc.classLister.Get(details.DeviceClassName)
			if err != nil {
				return nil, fmt.Errorf("claim %s, request %s: could not retrieve device class %s: %w", klog.KObj(claim), request.Name, details.DeviceClassName, err)
			}

			requestData := requestData{
				class: class,
			}

			switch request.CountMode {
			case resourceapi.DeviceCountModeExact:
				numDevices := request.Count
				if numDevices > math.MaxInt {
					// Allowed by API validation, but doesn't make sense.
					return nil, fmt.Errorf("claim %s, request %s: exact count %d is too large", klog.KObj(claim), request.Name, numDevices)
				}
				requestData.numDevices = int(numDevices)
			case resourceapi.DeviceCountModeAll:
				requestData.allDevices = make([]deviceWithID, 0, resourceapi.AllocationResultsMaxSize)
				for _, pool := range pools {
					if pool.IsIncomplete {
						return nil, fmt.Errorf("claim %s, request %s: asks for all devices, but resource pool %s is currently being updated", klog.KObj(claim), request.Name, pool.PoolID)
					}

					for _, slice := range pool.Slices {
						for deviceIndex := range slice.Spec.Devices {
							selectable, err := alloc.isSelectable(requestIndices{claimIndex: claimIndex, requestIndex: requestIndex}, slice, deviceIndex)
							if err != nil {
								return nil, err
							}
							if selectable {
								requestData.allDevices = append(requestData.allDevices, deviceWithID{device: &slice.Spec.Devices[deviceIndex], DeviceID: DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: slice.Spec.Devices[deviceIndex].Name}})
							}
						}
					}
				}
				requestData.numDevices = len(requestData.allDevices)
				alloc.logger.V(5).Info("Request for 'all' devices", "claim", klog.KObj(claim), "request", request.Name, "numDevicesPerRequest", requestData.numDevices)
			default:
				return nil, fmt.Errorf("claim %s, request %s: unsupported count mode %s", klog.KObj(claim), request.Name, request.CountMode)
			}
			alloc.requestData[requestIndices{claimIndex: claimIndex, requestIndex: requestIndex}] = requestData
			numDevices += requestData.numDevices
		}
		alloc.logger.Info("Checked claim", "claim", klog.KObj(claim), "numDevices", numDevices)

		// Check that we don't end up with too many results.
		if numDevices > resourceapi.AllocationResultsMaxSize {
			return nil, fmt.Errorf("claim %s: number of requested devices %d exceeds the claim limit of %d", klog.KObj(claim), numDevices, resourceapi.AllocationResultsMaxSize)
		}

		// If we don't, then we can pre-allocate the result slices for
		// appending the actual results later.
		alloc.result[claimIndex] = &resourceapi.AllocationResult{
			Devices: resourceapi.DeviceAllocationResult{
				Results: make([]resourceapi.DeviceRequestAllocationResult, 0, numDevices),
			},
		}

		// Constraints are assumed to be monotonic: once a constraint returns
		// false, adding more devices will not cause it to return true. This
		// allows the search to stop early once a constraint returns false.
		var constraints = make([]constraint, len(claim.Spec.Devices.Constraints))
		for i, constraint := range claim.Spec.Devices.Constraints {
			switch {
			case constraint.MatchAttribute != nil:
				logger := alloc.logger
				if loggerV := alloc.logger.V(5); loggerV.Enabled() {
					logger = klog.LoggerWithName(logger, "matchAttributeConstraint")
					logger = klog.LoggerWithValues(logger, "matchAttribute", *constraint.MatchAttribute)
				}
				m := &matchAttributeConstraint{
					logger:        logger,
					requestNames:  sets.New(constraint.Requests...),
					attributeName: *constraint.MatchAttribute,
				}
				constraints = append(constraints, m)
			default:
				// Unknown constraint type!
				return nil, fmt.Errorf("claim %s, constraint #%d: unsupported constraint type", klog.KObj(claim), i)
			}
		}
		alloc.constraints = append(alloc.constraints, constraints)
	}

	// Selecting a device for a request is independent of what has been
	// allocated already. Therefore the result of checking a request against
	// a device instance in the pool can be cached. The pointer to both
	// can serve as key because they are static for the duration of
	// the Allocate call and can be compared in Go.
	alloc.deviceMatchesRequest = make(map[matchKey]bool)

	// Some of the existing devices are probably already allocated by
	// claims...
	claims, err := alloc.claimLister.ListAllAllocated()
	numAllocated := 0
	if err != nil {
		return nil, fmt.Errorf("list allocated claims: %w", err)
	}
	for _, claim := range claims {
		// Sanity check..
		if claim.Status.Allocation == nil {
			continue
		}
		for _, result := range claim.Status.Allocation.Devices.Results {
			deviceID := DeviceID{Driver: result.Driver, Pool: result.Pool, Device: result.Device}
			alloc.allocated[deviceID] = true
			numAllocated++
		}
	}
	alloc.logger.V(6).Info("Gathered information about allocated devices", "numAllocated", numAllocated)

	// In practice, there aren't going to be many different CEL
	// expressions. Most likely, there is going to be handful of different
	// device classes that get used repeatedly. Different requests may all
	// use the same selector. Therefore compiling CEL expressions on demand
	// could be a useful performance enhancement. It's not implemented yet
	// because the key is more complex (just the string?) and the memory
	// for both key and cached content is larger than for device matches.
	//
	// We may also want to cache this in the shared [Allocator] instance,
	// which implies adding locking.

	// All errors get created such that they can be returned by Allocate
	// without further wrapping.
	done, err := alloc.allocateOne(deviceIndices{})
	if err != nil {
		return nil, err
	}
	if errors.Is(err, errStop) || !done {
		return nil, nil
	}

	// Populate configs.
	for claimIndex, allocationResult := range alloc.result {
		claim := alloc.claimsToAllocate[claimIndex]
		for requestIndex := range claim.Spec.Devices.Requests {
			class := alloc.requestData[requestIndices{claimIndex: claimIndex, requestIndex: requestIndex}].class
			if class != nil {
				for _, config := range class.Spec.Config {
					allocationResult.Devices.Config = append(allocationResult.Devices.Config, resourceapi.DeviceAllocationConfiguration{
						Source:              resourceapi.AllocationConfigSourceClass,
						Requests:            nil, // All of them...
						DeviceConfiguration: config.DeviceConfiguration,
					})
				}
			}
		}
		for _, config := range claim.Spec.Devices.Config {
			allocationResult.Devices.Config = append(allocationResult.Devices.Config, resourceapi.DeviceAllocationConfiguration{
				Source:              resourceapi.AllocationConfigSourceClaim,
				Requests:            config.Requests,
				DeviceConfiguration: config.DeviceConfiguration,
			})
		}
	}

	return alloc.result, nil
}

// errStop is a special error that gets returned by allocateOne if it detects
// that allocation cannot succeed.
var errStop = errors.New("stop allocation")

// allocator is used while an [Allocator.Allocate] is running. Only a single
// goroutine works with it, so there is no need for locking.
type allocator struct {
	*Allocator
	ctx                  context.Context
	logger               klog.Logger
	pools                []*Pool
	deviceMatchesRequest map[matchKey]bool
	constraints          [][]constraint                 // one list of constraints per claim
	requestData          map[requestIndices]requestData // one entry per request
	allocated            map[DeviceID]bool
	result               []*resourceapi.AllocationResult
}

// matchKey identifies a device/request pair.
type matchKey struct {
	DeviceID
	requestIndices
}

// requestIndices identifies one specific request by its
// claim and request index.
type requestIndices struct {
	claimIndex, requestIndex int
}

// deviceIndices identifies one specific required device inside
// a request of a certain claim.
type deviceIndices struct {
	claimIndex, requestIndex, deviceIndex int
}

type requestData struct {
	class      *resourceapi.DeviceClass
	numDevices int

	// pre-determined set of devices for allocating "all" devices
	allDevices []deviceWithID
}

type deviceWithID struct {
	DeviceID
	device *resourceapi.Device
}

type constraint interface {
	// add is called whenever a device is about to be allocated. It must
	// check whether the device matches the constraint and if yes,
	// track that it is allocated.
	add(requestName string, device *resourceapi.Device, deviceID DeviceID) bool

	// For every successful add there is exactly one matching removed call
	// with the exact same parameters.
	remove(requestName string, device *resourceapi.Device, deviceID DeviceID)
}

// matchAttributeConstraint compares an attribute value across devices.
// All devices must share the same value. When the set of devices is
// empty, any device that has the attribute can be added. After that,
// only matching devices can be added.
//
// We don't need to track *which* devices are part of the set, only
// how many.
type matchAttributeConstraint struct {
	logger        klog.Logger // Includes name and attribute name, so no need to repeat in log messages.
	requestNames  sets.Set[string]
	attributeName string

	attribute  *resourceapi.DeviceAttribute
	numDevices int
}

func (m *matchAttributeConstraint) add(requestName string, device *resourceapi.Device, deviceID DeviceID) bool {
	if m.requestNames.Len() > 0 && !m.requestNames.Has(requestName) {
		// Device not affected by constraint.
		m.logger.V(6).Info("Constraint does not apply to request", "request", requestName)
		return true
	}

	attribute := lookupAttribute(device, m.attributeName)
	if attribute == nil {
		// Doesn't have the attribute.
		m.logger.V(6).Info("Constraint not satisfied, attribute not set")
		return false
	}

	if m.numDevices == 0 {
		// The first device can always get picked.
		m.attribute = attribute
		m.numDevices = 1
		m.logger.V(6).Info("First in set")
		return true
	}

	switch {
	case attribute.StringValue != nil:
		if m.attribute.StringValue == nil || *attribute.StringValue != *m.attribute.StringValue {
			m.logger.V(6).Info("String values different")
			return false
		}
	case attribute.IntValue != nil:
		if m.attribute.IntValue == nil || *attribute.IntValue != *m.attribute.IntValue {
			m.logger.V(6).Info("Int values different")
			return false
		}
	case attribute.BoolValue != nil:
		if m.attribute.BoolValue == nil || *attribute.BoolValue != *m.attribute.BoolValue {
			m.logger.V(6).Info("Bool values different")
			return false
		}
	case attribute.VersionValue != nil:
		if m.attribute.VersionValue == nil || *attribute.VersionValue != *m.attribute.VersionValue {
			// TODO: should this be a semver-based comparison? We
			// could require that vendors use a normalized
			// representation instead of "01.02.03" in one device
			// and "1.2.3" in another.
			m.logger.V(6).Info("Version values different")
			return false
		}
	default:
		// Unknown value type, cannot match.
		m.logger.V(6).Info("Match attribute type unknown")
		return false
	}

	m.numDevices++
	m.logger.V(6).Info("Constraint satisfied by device", "device", deviceID, "numDevices", m.numDevices)
	return true
}

func (m *matchAttributeConstraint) remove(requestName string, device *resourceapi.Device, deviceID DeviceID) {
	if m.requestNames.Len() > 0 && !m.requestNames.Has(requestName) {
		// Device not affected by constraint.
		return
	}

	m.numDevices--
	m.logger.V(6).Info("Device removed from constraint set", "device", deviceID, "numDevices", m.numDevices)
}

func lookupAttribute(device *resourceapi.Device, attributeName string) *resourceapi.DeviceAttribute {
	for i := range device.Attributes {
		if device.Attributes[i].Name == attributeName {
			return &device.Attributes[i]
		}
	}

	return nil
}

// allocateOne iterates over all eligible devices (not in use, match selector,
// satisfy constraints) for a specific required device. It returns true if
// everything got allocated, an error if allocation needs to stop.
func (alloc *allocator) allocateOne(r deviceIndices) (bool, error) {
	if r.claimIndex >= len(alloc.claimsToAllocate) {
		// Done! If we were doing scoring, we would compare the current allocation result
		// against the previous one, keep the best, and continue. Without scoring, we stop
		// and use the first solution.
		alloc.logger.V(4).Info("Allocation result found")
		return true, nil
	}

	claim := alloc.claimsToAllocate[r.claimIndex]
	if r.requestIndex >= len(claim.Spec.Devices.Requests) {
		// Done with the claim, continue with the next one.
		return alloc.allocateOne(deviceIndices{claimIndex: r.claimIndex + 1})
	}

	// We already know how many devices per request are needed.
	// Ready to move on to the next request?
	requestData := alloc.requestData[requestIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex}]
	if r.deviceIndex >= requestData.numDevices {
		return alloc.allocateOne(deviceIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex + 1})
	}

	request := &alloc.claimsToAllocate[r.claimIndex].Spec.Devices.Requests[r.requestIndex]
	doAllDevices := request.CountMode == resourceapi.DeviceCountModeAll
	alloc.logger.V(5).Info("Allocating one device", "currentClaim", r.claimIndex, "totalClaims", len(alloc.claimsToAllocate), "currentRequest", r.requestIndex, "totalRequestsPerClaim", len(claim.Spec.Devices.Requests), "currentDevice", r.deviceIndex, "devicesPerRequest", requestData.numDevices, "allDevices", doAllDevices, "adminAccess", request.AdminAccess)

	if doAllDevices {
		// For "all" devices we already know which ones we need. We
		// just need to check whether we can use them.
		deviceWithID := requestData.allDevices[r.deviceIndex]
		_, _, err := alloc.allocateDevice(r, deviceWithID.device, deviceWithID.DeviceID, true)
		if err != nil {
			return false, err
		}
		done, err := alloc.allocateOne(deviceIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex, deviceIndex: r.deviceIndex + 1})
		if err != nil {
			return false, err
		}

		// The order in which we allocate "all" devices doesn't matter,
		// so we only try with the one which was up next. If we couldn't
		// get all of them, then there is no solution and we have to stop.
		if !done {
			return false, errStop
		}
		return done, nil
	}

	// We need to find suitable devices.
	for _, pool := range alloc.pools {
		for _, slice := range pool.Slices {
			for deviceIndex := range slice.Spec.Devices {
				deviceID := DeviceID{Driver: pool.Driver, Pool: pool.Pool, Device: slice.Spec.Devices[deviceIndex].Name}

				// Checking for "in use" is cheap and thus gets done first.
				if !request.AdminAccess && alloc.allocated[deviceID] {
					alloc.logger.V(6).Info("Device in use", "device", deviceID)
					continue
				}

				// Next check selectors.
				selectable, err := alloc.isSelectable(requestIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex}, slice, deviceIndex)
				if err != nil {
					return false, err
				}
				if !selectable {
					alloc.logger.V(6).Info("Device not selectable", "device", deviceID)
					continue
				}

				// Finally treat as allocated and move on to the next device.
				allocated, deallocate, err := alloc.allocateDevice(r, &slice.Spec.Devices[deviceIndex], deviceID, false)
				if err != nil {
					return false, err
				}
				if !allocated {
					// In use or constraint violated...
					alloc.logger.V(6).Info("Device not usable", "device", deviceID)
					continue
				}
				done, err := alloc.allocateOne(deviceIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex, deviceIndex: r.deviceIndex + 1})
				if err != nil {
					return false, err
				}

				// If we found a solution, then we can stop.
				if done {
					return done, nil
				}

				// Otherwise try some other device after rolling back.
				deallocate()
			}
		}
	}

	// If we get here without finding a solution, then there is none.
	return false, nil
}

// isSelectable checks whether a device satisfies the request and class selectors.
func (alloc *allocator) isSelectable(r requestIndices, slice *resourceapi.ResourceSlice, deviceIndex int) (bool, error) {
	device := &slice.Spec.Devices[deviceIndex]
	deviceID := DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name}
	matchKey := matchKey{DeviceID: deviceID, requestIndices: r}
	if matches, ok := alloc.deviceMatchesRequest[matchKey]; ok {
		// No need to check again.
		return matches, nil
	}

	requestData := alloc.requestData[r]
	if requestData.class != nil {
		match, err := alloc.selectorsMatch(r, device, deviceID, requestData.class, requestData.class.Spec.Selectors)
		if err != nil {
			return false, err
		}
		if !match {
			alloc.deviceMatchesRequest[matchKey] = false
			return false, nil
		}
	}

	request := &alloc.claimsToAllocate[r.claimIndex].Spec.Devices.Requests[r.requestIndex]
	match, err := alloc.selectorsMatch(r, device, deviceID, nil, request.Selectors)
	if err != nil {
		return false, err
	}
	if !match {
		alloc.deviceMatchesRequest[matchKey] = false
		return false, nil
	}

	alloc.deviceMatchesRequest[matchKey] = true
	return true, nil

}

func (alloc *allocator) selectorsMatch(r requestIndices, device *resourceapi.Device, deviceID DeviceID, class *resourceapi.DeviceClass, selectors []resourceapi.DeviceSelector) (bool, error) {
	for i, selector := range selectors {
		expr := cel.Compiler.CompileCELExpression(selector.CEL.Expression, environment.StoredExpressions)
		if expr.Error != nil {
			// Could happen if some future apiserver accepted some
			// future expression and then got downgraded. Normally
			// the "stored expression" mechanism prevents that, but
			// this code here might be more than one release older
			// than the cluster it runs in.
			if class != nil {
				return false, fmt.Errorf("class %s: selector #%d: CEL compile error: %w", class.Name, i, expr.Error)
			}
			return false, fmt.Errorf("claim %s: selector #%d: CEL compile error: %w", klog.KObj(alloc.claimsToAllocate[r.claimIndex]), i, expr.Error)
		}

		matches, err := expr.DeviceMatches(alloc.ctx, cel.Device{Driver: deviceID.Driver, Attributes: device.Attributes, Capacities: device.Capacities})
		if class != nil {
			alloc.logger.V(6).Info("CEL result", "device", deviceID, "class", klog.KObj(class), "selector", i, "expression", selector.CEL.Expression, "matches", matches, "err", err)
		} else {
			alloc.logger.V(6).Info("CEL result", "device", deviceID, "claim", klog.KObj(alloc.claimsToAllocate[r.claimIndex]), "selector", i, "expression", selector.CEL.Expression, "matches", matches, "err", err)
		}

		if err != nil {
			// TODO (future): more detailed errors which reference class resp. claim.
			if class != nil {
				return false, fmt.Errorf("class %s: selector #%d: CEL runtime error: %w", class.Name, i, err)
			}
			return false, fmt.Errorf("claim %s: selector #%d: CEL runtime error: %w", klog.KObj(alloc.claimsToAllocate[r.claimIndex]), i, err)
		}
		if !matches {
			return false, nil
		}
	}

	// All of them match.
	return true, nil
}

// allocateDevice checks device availability and constraints for one
// candidate. If that candidate works out okay, the shared state gets updated
// as if that candidate had been allocated. If allocation cannot continue later
// and must try something else, then the rollback function can be invoked to
// restore the previous state.
func (alloc *allocator) allocateDevice(r deviceIndices, device *resourceapi.Device, deviceID DeviceID, must bool) (bool, func(), error) {
	claim := alloc.claimsToAllocate[r.claimIndex]
	request := &claim.Spec.Devices.Requests[r.requestIndex]
	adminAccess := request.AdminAccess
	if !adminAccess && alloc.allocated[deviceID] {
		alloc.logger.V(6).Info("Device in use", "device", deviceID)
		return false, nil, nil
	}

	// It's available. Now check constraints.
	for i, constraint := range alloc.constraints[r.claimIndex] {
		added := constraint.add(request.Name, device, deviceID)
		if !added {
			if must {
				// It does not make sense to declare a claim where a constraint prevents getting
				// all devices. Treat this as an error.
				return false, nil, fmt.Errorf("claim %s, request %s: cannot add device %s because a claim constraint would not be satisfied", klog.KObj(claim), request.Name, deviceID)
			}

			// Roll back for all previous constraints before we return.
			for e := 0; e < i; e++ {
				alloc.constraints[r.claimIndex][e].remove(request.Name, device, deviceID)
			}
			return false, nil, nil
		}
	}

	// All constraints satisfied. Mark as in use (unless we do admin access)
	// and record the result.
	alloc.logger.V(6).Info("Device allocated", "device", deviceID)
	if !adminAccess {
		alloc.allocated[deviceID] = true
	}
	result := resourceapi.DeviceRequestAllocationResult{
		Request: request.Name,
		Driver:  deviceID.Driver,
		Pool:    deviceID.Pool,
		Device:  deviceID.Device,
	}
	previousNumResults := len(alloc.result[r.claimIndex].Devices.Results)
	alloc.result[r.claimIndex].Devices.Results = append(alloc.result[r.claimIndex].Devices.Results, result)

	return true, func() {
		for _, constraint := range alloc.constraints[r.claimIndex] {
			constraint.remove(request.Name, device, deviceID)
		}
		if !adminAccess {
			alloc.allocated[deviceID] = false
		}
		// Truncate, but keep the underlying slice.
		alloc.result[r.claimIndex].Devices.Results = alloc.result[r.claimIndex].Devices.Results[:previousNumResults]
		alloc.logger.V(6).Info("Device deallocated", "device", deviceID)
	}, nil
}
