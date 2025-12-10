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

package incubating

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/dynamic-resource-allocation/structured/internal"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

type DeviceClassLister = internal.DeviceClassLister
type Features = internal.Features
type DeviceID = internal.DeviceID
type Stats = internal.Stats

func MakeDeviceID(driver, pool, device string) DeviceID {
	return internal.MakeDeviceID(driver, pool, device)
}

// SupportedFeatures includes all additional features,
// making this the variant that is used when any of those
// are enabled.
var SupportedFeatures = internal.Features{
	AdminAccess:          true,
	PrioritizedList:      true,
	PartitionableDevices: true,
	DeviceTaints:         true,
}

type Allocator struct {
	features         Features
	allocatedDevices sets.Set[DeviceID]
	classLister      DeviceClassLister
	slices           []*resourceapi.ResourceSlice
	celCache         *cel.Cache
	// availableCounters contains the available counters for each
	// resource pool. It acts as a cache that is updated the first time
	// the available counters are needed for each pool. The information
	// about each pool is never updated once set the first time.
	// This is computed bsed on information on the Allocator, so it will
	// be correct even for multiple usages of the Allocator.
	// The keys in the map are resource pool names.
	// The allocator might be accessed by different goroutines, so
	// access to this map must be synchronized.
	availableCounters map[draapi.UniqueString]counterSets
	mutex             sync.RWMutex
	// numAllocateOneInvocations counts the number of times the allocateOne
	// function is called for the allocator. This is a measurement of the
	// amount of work the allocator had to do to allocate devices
	// for the claims.
	numAllocateOneInvocations atomic.Int64
}

var _ internal.AllocatorExtended = &Allocator{}

// NewAllocator returns an allocator for a certain set of claims or an error if
// some problem was detected which makes it impossible to allocate claims.
//
// The returned Allocator can be used multiple times and is thread-safe.
func NewAllocator(ctx context.Context,
	features Features,
	allocatedDevices sets.Set[DeviceID],
	classLister DeviceClassLister,
	slices []*resourceapi.ResourceSlice,
	celCache *cel.Cache,
) (*Allocator, error) {
	return &Allocator{
		features:          features,
		allocatedDevices:  allocatedDevices,
		classLister:       classLister,
		slices:            slices,
		celCache:          celCache,
		availableCounters: make(map[draapi.UniqueString]counterSets),
	}, nil
}

func (a *Allocator) Allocate(ctx context.Context, node *v1.Node, claims []*resourceapi.ResourceClaim) (finalResult []resourceapi.AllocationResult, finalErr error) {
	alloc := &allocator{
		Allocator:            a,
		ctx:                  ctx, // all methods share the same a and thus ctx
		logger:               klog.FromContext(ctx),
		node:                 node,
		claimsToAllocate:     claims,
		deviceMatchesRequest: make(map[matchKey]bool),
		constraints:          make([][]constraint, len(claims)),
		consumedCounters:     make(map[draapi.UniqueString]counterSets),
		requestData:          make(map[requestIndices]requestData),
		result:               make([]internalAllocationResult, len(claims)),
	}
	alloc.logger.V(5).Info("Starting allocation", "numClaims", len(alloc.claimsToAllocate), "numSlices", len(alloc.slices))
	defer func() {
		alloc.logger.V(5).Info("Done with allocation", "success", len(finalResult) == len(alloc.claimsToAllocate), "err", finalErr)
	}()

	// First determine all eligible pools.
	pools, err := GatherPools(ctx, alloc.slices, node, a.features)
	if err != nil {
		return nil, fmt.Errorf("gather pool information: %w", err)
	}
	alloc.pools = pools
	if loggerV := alloc.logger.V(7); loggerV.Enabled() {
		loggerV.Info("Gathered pool information", "numPools", len(pools), "pools", pools)
	} else {
		alloc.logger.V(5).Info("Gathered pool information", "numPools", len(pools))
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
	minDevicesTotal := 0
	for claimIndex, claim := range alloc.claimsToAllocate {
		minDevicesPerClaim := 0

		// If we have any any request that wants "all" devices, we need to
		// figure out how much "all" is. If some pool is incomplete, we stop
		// here because allocation cannot succeed. Once we do scoring, we should
		// stop in all cases, not just when "all" devices are needed, because
		// pulling from an incomplete might not pick the best solution and it's
		// better to wait. This does not matter yet as long the incomplete pool
		// has some matching device.
		for requestIndex := range claim.Spec.Devices.Requests {
			request := &claim.Spec.Devices.Requests[requestIndex]
			requestKey := requestIndices{claimIndex: claimIndex, requestIndex: requestIndex}
			hasSubRequests := len(request.FirstAvailable) > 0

			// Error out if the prioritizedList feature is not enabled and the request
			// has subrequests. This is to avoid surprising behavior for users.
			if !a.features.PrioritizedList && hasSubRequests {
				return nil, fmt.Errorf("claim %s, request %s: has subrequests, but the DRAPrioritizedList feature is disabled", klog.KObj(claim), request.Name)
			}

			if hasSubRequests {
				// We need to find the minimum number of devices that can be allocated
				// for the request, so setting this to a high number so we can do the
				// easy comparison in the loop.
				minDevicesPerRequest := math.MaxInt

				// A request with subrequests gets one entry per subrequest in alloc.requestData.
				// We can only predict a lower number of devices because it depends on which
				// subrequest gets chosen.
				for i, subReq := range request.FirstAvailable {
					reqData, err := alloc.validateDeviceRequest(&deviceSubRequestAccessor{subRequest: &subReq},
						&exactDeviceRequestAccessor{request: request}, requestKey, pools)
					if err != nil {
						return nil, err
					}
					requestKey.subRequestIndex = i
					alloc.requestData[requestKey] = reqData
					if reqData.numDevices < minDevicesPerRequest {
						minDevicesPerRequest = reqData.numDevices
					}
				}
				minDevicesPerClaim += minDevicesPerRequest
			} else {
				reqData, err := alloc.validateDeviceRequest(&exactDeviceRequestAccessor{request: request}, nil, requestKey, pools)
				if err != nil {
					return nil, err
				}
				alloc.requestData[requestKey] = reqData
				minDevicesPerClaim += reqData.numDevices
			}
		}
		alloc.logger.V(6).Info("Checked claim", "claim", klog.KObj(claim), "minDevices", minDevicesPerClaim)
		// Check that we don't end up with too many results.
		// This isn't perfectly reliable because numDevicesPerClaim is
		// only a lower bound, so allocation also has to check this.
		if minDevicesPerClaim > resourceapi.AllocationResultsMaxSize {
			return nil, fmt.Errorf("claim %s: number of requested devices %d exceeds the claim limit of %d", klog.KObj(claim), minDevicesPerClaim, resourceapi.AllocationResultsMaxSize)
		}

		// If we don't, then we can pre-allocate the result slices for
		// appending the actual results later.
		alloc.result[claimIndex].devices = make([]internalDeviceResult, 0, minDevicesPerClaim)

		// Constraints are assumed to be monotonic: once a constraint returns
		// false, adding more devices will not cause it to return true. This
		// allows the search to stop early once a constraint returns false.
		constraints := make([]constraint, len(claim.Spec.Devices.Constraints))
		for i, constraint := range claim.Spec.Devices.Constraints {
			switch {
			case constraint.MatchAttribute != nil:
				matchAttribute := resourceapi.FullyQualifiedName(*constraint.MatchAttribute)
				logger := alloc.logger
				if loggerV := alloc.logger.V(6); loggerV.Enabled() {
					logger = klog.LoggerWithName(logger, "matchAttributeConstraint")
					logger = klog.LoggerWithValues(logger, "matchAttribute", matchAttribute)
				}
				m := &matchAttributeConstraint{
					logger:        logger,
					requestNames:  sets.New(constraint.Requests...),
					attributeName: matchAttribute,
				}
				constraints[i] = m
			default:
				// Unknown constraint type!
				return nil, fmt.Errorf("claim %s, constraint #%d: empty constraint (unsupported constraint type?)", klog.KObj(claim), i)
			}
		}
		alloc.constraints[claimIndex] = constraints
		minDevicesTotal += minDevicesPerClaim
	}

	// Selecting a device for a request is independent of what has been
	// allocated already. Therefore the result of checking a request against
	// a device instance in the pool can be cached. The pointer to both
	// can serve as key because they are static for the duration of
	// the Allocate call and can be compared in Go.
	alloc.deviceMatchesRequest = make(map[matchKey]bool)

	// We can estimate the size based on what we need to allocate.
	alloc.allocatingDevices = make(map[DeviceID]sets.Set[int], minDevicesTotal)

	alloc.logger.V(6).Info("Gathered information about devices", "numAllocated", len(alloc.allocatedDevices), "minDevicesToBeAllocated", minDevicesTotal)

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
	done, err := alloc.allocateOne(deviceIndices{}, false)
	if errors.Is(err, errStop) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if !done {
		// If no devices could be allocated, but we found one or more
		// invalid pools, return an error here. We didn't do it during
		// allocation since there might be valid pools from which the
		// claims could be satisfied.
		for _, pool := range pools {
			if pool.IsInvalid {
				// Not a fatal error, allocation on other nodes may proceed.
				// The error is only surfaced if allocation fails on all nodes.
				return nil, fmt.Errorf("invalid resource pools were encountered%w", internal.ErrFailedAllocationOnNode)
			}
		}
		return nil, nil
	}

	result := make([]resourceapi.AllocationResult, len(alloc.result))
	for claimIndex, internalResult := range alloc.result {
		claim := alloc.claimsToAllocate[claimIndex]
		allocationResult := &result[claimIndex]
		allocationResult.Devices.Results = make([]resourceapi.DeviceRequestAllocationResult, len(internalResult.devices))
		for i, internal := range internalResult.devices {
			allocationResult.Devices.Results[i] = resourceapi.DeviceRequestAllocationResult{
				Request:     internal.requestName(),
				Driver:      internal.id.Driver.String(),
				Pool:        internal.id.Pool.String(),
				Device:      internal.id.Device.String(),
				AdminAccess: internal.adminAccess,
				Tolerations: internal.lookupRequest(claim).tolerations(),
			}
		}

		// Populate configs.

		// Each class config gets added only once.
		// We need to keep track of which class configs have already been added and at which position in the allocationResult.Devices.Config.
		type configRange struct{ start, end int }
		configIndexesForClass := make(map[string]configRange) // Key: class name / Value: position of the configs for the class in allocationResult.Devices.Config.
		for requestIndex := range claim.Spec.Devices.Requests {
			requestKey := requestIndices{claimIndex: claimIndex, requestIndex: requestIndex}
			requestData := alloc.requestData[requestKey]
			if requestData.parentRequest != nil {
				// We need the class of the selected subrequest.
				requestKey.subRequestIndex = requestData.selectedSubRequestIndex
				requestData = alloc.requestData[requestKey]
			}

			class := requestData.class
			if class == nil {
				continue
			}
			configIndexes, exists := configIndexesForClass[class.Name]
			if exists {
				// The configs for the class have already been added.
				// Just append the request name for the request class.
				for i := configIndexes.start; i < configIndexes.end; i++ {
					allocationResult.Devices.Config[i].Requests = append(allocationResult.Devices.Config[i].Requests, requestData.requestName())
				}
				continue
			}

			// Add all configs for the class once.
			initialConfigLen := len(allocationResult.Devices.Config)
			for _, config := range class.Spec.Config {
				allocationResult.Devices.Config = append(allocationResult.Devices.Config, resourceapi.DeviceAllocationConfiguration{
					Source:              resourceapi.AllocationConfigSourceClass,
					Requests:            []string{requestData.requestName()},
					DeviceConfiguration: config.DeviceConfiguration,
				})
			}
			configIndexesForClass[class.Name] = configRange{start: initialConfigLen, end: len(allocationResult.Devices.Config)}
		}
		for _, config := range claim.Spec.Devices.Config {
			// If Requests are empty, it applies to all. So it can just be included.
			if len(config.Requests) == 0 {
				allocationResult.Devices.Config = append(allocationResult.Devices.Config, resourceapi.DeviceAllocationConfiguration{
					Source:              resourceapi.AllocationConfigSourceClaim,
					Requests:            config.Requests,
					DeviceConfiguration: config.DeviceConfiguration,
				})
				continue
			}

			for i, request := range claim.Spec.Devices.Requests {
				if slices.Contains(config.Requests, request.Name) {
					allocationResult.Devices.Config = append(allocationResult.Devices.Config, resourceapi.DeviceAllocationConfiguration{
						Source:              resourceapi.AllocationConfigSourceClaim,
						Requests:            config.Requests,
						DeviceConfiguration: config.DeviceConfiguration,
					})
					continue
				}

				requestKey := requestIndices{claimIndex: claimIndex, requestIndex: i}
				requestData := alloc.requestData[requestKey]
				if requestData.parentRequest == nil {
					continue
				}

				subRequest := request.FirstAvailable[requestData.selectedSubRequestIndex]
				subRequestName := fmt.Sprintf("%s/%s", request.Name, subRequest.Name)
				if slices.Contains(config.Requests, subRequestName) {
					allocationResult.Devices.Config = append(allocationResult.Devices.Config, resourceapi.DeviceAllocationConfiguration{
						Source:              resourceapi.AllocationConfigSourceClaim,
						Requests:            config.Requests,
						DeviceConfiguration: config.DeviceConfiguration,
					})
				}
			}
		}

		// Determine node selector.
		nodeSelector, err := alloc.createNodeSelector(internalResult.devices)
		if err != nil {
			return nil, fmt.Errorf("create NodeSelector for claim %s: %w", claim.Name, err)
		}
		allocationResult.NodeSelector = nodeSelector
	}

	return result, nil
}

func (a *Allocator) GetStats() Stats {
	s := Stats{
		NumAllocateOneInvocations: a.numAllocateOneInvocations.Load(),
	}
	return s
}

func (alloc *allocator) validateDeviceRequest(request requestAccessor, parentRequest requestAccessor, requestKey requestIndices, pools []*Pool) (requestData, error) {
	claim := alloc.claimsToAllocate[requestKey.claimIndex]
	requestData := requestData{
		request:       request,
		parentRequest: parentRequest,
	}
	for i, selector := range request.selectors() {
		if selector.CEL == nil {
			// Unknown future selector type!
			return requestData, fmt.Errorf("claim %s, request %s, selector #%d: CEL expression empty (unsupported selector type?)", klog.KObj(claim), request.name(), i)
		}
	}

	if !alloc.features.AdminAccess && request.hasAdminAccess() {
		return requestData, fmt.Errorf("claim %s, request %s: admin access is requested, but the feature is disabled", klog.KObj(claim), request.name())
	}

	// Should be set. If it isn't, something changed and we should refuse to proceed.
	if request.deviceClassName() == "" {
		return requestData, fmt.Errorf("claim %s, request %s: missing device class name (unsupported request type?)", klog.KObj(claim), request.name())
	}
	class, err := alloc.classLister.Get(request.deviceClassName())
	if err != nil {
		return requestData, fmt.Errorf("claim %s, request %s: could not retrieve device class %s: %w", klog.KObj(claim), request.name(), request.deviceClassName(), err)
	}

	// Start collecting information about the request.
	// The class must be set and stored before calling isSelectable.
	requestData.class = class

	switch request.allocationMode() {
	case resourceapi.DeviceAllocationModeExactCount:
		numDevices := request.count()
		if numDevices > math.MaxInt {
			// Allowed by API validation, but doesn't make sense.
			return requestData, fmt.Errorf("claim %s, request %s: exact count %d is too large", klog.KObj(claim), request.name(), numDevices)
		}
		requestData.numDevices = int(numDevices)
	case resourceapi.DeviceAllocationModeAll:
		// If we have any any request that wants "all" devices, we need to
		// figure out how much "all" is. If some pool is incomplete, we stop
		// here because allocation cannot succeed. Once we do scoring, we should
		// stop in all cases, not just when "all" devices are needed, because
		// pulling from an incomplete might not pick the best solution and it's
		// better to wait. This does not matter yet as long the incomplete pool
		// has some matching device.
		requestData.allDevices = make([]deviceWithID, 0, resourceapi.AllocationResultsMaxSize)
		for _, pool := range pools {
			if pool.IsIncomplete {
				return requestData, fmt.Errorf("claim %s, request %s: asks for all devices, but resource pool %s is currently being updated", klog.KObj(claim), request.name(), pool.PoolID)
			}
			if pool.IsInvalid {
				return requestData, fmt.Errorf("claim %s, request %s: asks for all devices, but resource pool %s is currently invalid", klog.KObj(claim), request.name(), pool.PoolID)
			}

			for _, slice := range pool.DeviceSlicesTargetingNode {
				for deviceIndex := range slice.Spec.Devices {
					selectable, err := alloc.isSelectable(requestKey, requestData, slice, deviceIndex)
					if err != nil {
						return requestData, err
					}
					if selectable {
						device := deviceWithID{
							id:     DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: slice.Spec.Devices[deviceIndex].Name},
							Device: &slice.Spec.Devices[deviceIndex],
							slice:  slice,
						}
						requestData.allDevices = append(requestData.allDevices, device)
					}
				}
			}
		}
		requestData.numDevices = len(requestData.allDevices)
		alloc.logger.V(6).Info("Request for 'all' devices", "claim", klog.KObj(claim), "request", request.name(), "numDevicesPerRequest", requestData.numDevices)
	default:
		return requestData, fmt.Errorf("claim %s, request %s: unsupported count mode %s", klog.KObj(claim), request.name(), request.allocationMode())
	}
	return requestData, nil
}

// errStop is a special error that gets returned by allocateOne if it detects
// that allocation cannot succeed.
var errStop = errors.New("stop allocation")

// errAllocationResultMaxSizeExceeded is a special error that gets return by
// allocatedOne when the number of allocated devices exceeds the max number
// allowed. This is checked by earlier invocations in the recursion and used
// to do more aggressive backtracking and avoid attempting allocations that
// we know can not succeed.
var errAllocationResultMaxSizeExceeded = errors.New("allocation max size exceeded")

// allocator is used while an [Allocator.Allocate] is running. Only a single
// goroutine works with it, so there is no need for locking.
type allocator struct {
	*Allocator
	ctx                  context.Context
	logger               klog.Logger
	node                 *v1.Node
	claimsToAllocate     []*resourceapi.ResourceClaim
	pools                []*Pool
	deviceMatchesRequest map[matchKey]bool
	constraints          [][]constraint // one list of constraints per claim
	// consumedCounters keeps track of the counters consumed by all devices
	// that are in the process of being allocated.
	// The keys in the map are resource pool names.
	consumedCounters map[draapi.UniqueString]counterSets
	requestData      map[requestIndices]requestData // one entry per request with no subrequests and one entry per subrequest
	// allocatingDevices tracks which devices will be newly allocated for a
	// particular attempt to find a solution. The map is indexed by device
	// and its values represent for which of a pod's claims the device will
	// be allocated.
	// Claims are identified by their index in claimsToAllocate.
	allocatingDevices map[DeviceID]sets.Set[int]
	result            []internalAllocationResult
}

// counterSets is a map with the name of counter sets to the counters in
// the set.
type counterSets map[draapi.UniqueString]map[string]resourceapi.Counter

// matchKey identifies a device/request pair.
type matchKey struct {
	DeviceID
	requestIndices
}

// requestIndices identifies one specific request
// or subrequest by three properties:
//
// - claimIndex: The index of the claim in the requestData map.
// - requestIndex: The index of the request in the claim.
// - subRequestIndex: The index of the subrequest in the parent request.
type requestIndices struct {
	claimIndex, requestIndex int
	subRequestIndex          int
}

// deviceIndices identifies one specific required device inside
// a request or subrequest of a certain claim.
type deviceIndices struct {
	claimIndex      int // The index of the claim in the allocator.
	requestIndex    int // The index of the request in the claim.
	subRequestIndex int // The index of the subrequest within the request (ignored if subRequest is false).
	deviceIndex     int // The index of a device within a request or subrequest.
}

type requestData struct {
	// The request or subrequest which needs to be allocated.
	// Never nil.
	request requestAccessor
	// The parent of a subrequest, nil if not a subrequest.
	parentRequest requestAccessor
	class         *resourceapi.DeviceClass
	numDevices    int

	// selectedSubRequestIndex is set for the entry with requestIndices.subRequestIndex == 0.
	// It is the index of the subrequest which got picked during allocation.
	selectedSubRequestIndex int

	// pre-determined set of devices for allocating "all" devices
	allDevices []deviceWithID
}

func (rd *requestData) requestName() string {
	if rd.parentRequest != nil {
		return fmt.Sprintf("%s/%s", rd.parentRequest.name(), rd.request.name())
	}
	return rd.request.name()
}

type deviceWithID struct {
	*draapi.Device
	id    DeviceID
	slice *draapi.ResourceSlice
	pool  *Pool
}

type internalAllocationResult struct {
	devices []internalDeviceResult
}

type internalDeviceResult struct {
	*draapi.Device
	request       string // name of the request (if no subrequests) or the subrequest
	parentRequest string // name of the request which contains the subrequest, empty otherwise
	id            DeviceID
	slice         *draapi.ResourceSlice
	adminAccess   *bool
}

func (idr internalDeviceResult) requestName() string {
	if idr.parentRequest == "" {
		return idr.request
	}
	return fmt.Sprintf("%s/%s", idr.parentRequest, idr.request)
}

func (idr internalDeviceResult) lookupRequest(claim *resourceapi.ResourceClaim) requestAccessor {
	requestName := idr.request
	if idr.parentRequest != "" {
		requestName = idr.parentRequest
	}
	for i := range claim.Spec.Devices.Requests {
		request := &claim.Spec.Devices.Requests[i]
		if request.Name != requestName {
			continue
		}
		if idr.parentRequest == "" {
			// No need to check sub-requests.
			return &exactDeviceRequestAccessor{request}
		}
		for j := range request.FirstAvailable {
			subRequest := &request.FirstAvailable[j]
			if subRequest.Name != idr.request {
				continue
			}
			return &deviceSubRequestAccessor{subRequest}
		}
	}
	return nil
}

type constraint interface {
	// add is called whenever a device is about to be allocated. It must
	// check whether the device matches the constraint and if yes,
	// track that it is allocated.
	add(requestName, subRequestName string, device *draapi.Device, deviceID DeviceID) bool

	// For every successful add there is exactly one matching removed call
	// with the exact same parameters.
	remove(requestName, subRequestName string, device *draapi.Device, deviceID DeviceID)
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
	attributeName resourceapi.FullyQualifiedName

	attribute  *resourceapi.DeviceAttribute
	numDevices int
}

func (m *matchAttributeConstraint) add(requestName, subRequestName string, device *draapi.Device, deviceID DeviceID) bool {
	if m.requestNames.Len() > 0 && !m.matches(requestName, subRequestName) {
		// Device not affected by constraint.
		m.logger.V(7).Info("Constraint does not apply to request", "request", requestName)
		return true
	}

	attribute := lookupAttribute(device, deviceID, m.attributeName)
	if attribute == nil {
		// Doesn't have the attribute.
		m.logger.V(7).Info("Constraint not satisfied, attribute not set")
		return false
	}

	if m.numDevices == 0 {
		// The first device can always get picked.
		m.attribute = attribute
		m.numDevices = 1
		m.logger.V(7).Info("First in set")
		return true
	}

	switch {
	case attribute.StringValue != nil:
		if m.attribute.StringValue == nil || *attribute.StringValue != *m.attribute.StringValue {
			m.logger.V(7).Info("String values different")
			return false
		}
	case attribute.IntValue != nil:
		if m.attribute.IntValue == nil || *attribute.IntValue != *m.attribute.IntValue {
			m.logger.V(7).Info("Int values different")
			return false
		}
	case attribute.BoolValue != nil:
		if m.attribute.BoolValue == nil || *attribute.BoolValue != *m.attribute.BoolValue {
			m.logger.V(7).Info("Bool values different")
			return false
		}
	case attribute.VersionValue != nil:
		// semver 2.0.0 requires that version strings are in their
		// minimal form (in particular, no leading zeros). Therefore a
		// strict "exact equal" check can do a string comparison.
		if m.attribute.VersionValue == nil || *attribute.VersionValue != *m.attribute.VersionValue {
			m.logger.V(7).Info("Version values different")
			return false
		}
	default:
		// Unknown value type, cannot match.
		m.logger.V(7).Info("Match attribute type unknown")
		return false
	}

	m.numDevices++
	m.logger.V(7).Info("Constraint satisfied by device", "device", deviceID, "numDevices", m.numDevices)
	return true
}

func (m *matchAttributeConstraint) remove(requestName, subRequestName string, device *draapi.Device, deviceID DeviceID) {
	if m.requestNames.Len() > 0 && !m.matches(requestName, subRequestName) {
		// Device not affected by constraint.
		return
	}

	m.numDevices--
	m.logger.V(7).Info("Device removed from constraint set", "device", deviceID, "numDevices", m.numDevices)
}

func (m *matchAttributeConstraint) matches(requestName, subRequestName string) bool {
	if subRequestName == "" {
		return m.requestNames.Has(requestName)
	} else {
		fullSubRequestName := fmt.Sprintf("%s/%s", requestName, subRequestName)
		return m.requestNames.Has(requestName) || m.requestNames.Has(fullSubRequestName)
	}
}

func lookupAttribute(device *draapi.Device, deviceID DeviceID, attributeName resourceapi.FullyQualifiedName) *resourceapi.DeviceAttribute {
	// Fully-qualified match?
	if attr, ok := device.Attributes[resourceapi.QualifiedName(attributeName)]; ok {
		return &attr
	}
	index := strings.Index(string(attributeName), "/")
	if index < 0 {
		// Should not happen for a valid fully qualified name.
		return nil
	}

	if string(attributeName[0:index]) != deviceID.Driver.String() {
		// Not an attribute of the driver and not found above,
		// so it is not available.
		return nil
	}

	// Domain matches the driver, so let's check just the ID.
	if attr, ok := device.Attributes[resourceapi.QualifiedName(attributeName[index+1:])]; ok {
		return &attr
	}

	return nil
}

// allocateOne iterates over all eligible devices (not in use, match selector,
// satisfy constraints) for a specific required device. It returns true if
// everything got allocated, an error if allocation needs to stop.
//
// allocateSubRequest is true when trying to allocate one particular subrequest.
// This allows the logic for subrequests to call allocateOne with the same
// device index without causing infinite recursion.
func (alloc *allocator) allocateOne(r deviceIndices, allocateSubRequest bool) (bool, error) {
	alloc.numAllocateOneInvocations.Add(1)

	if alloc.ctx.Err() != nil {
		return false, fmt.Errorf("filter operation aborted: %w", context.Cause(alloc.ctx))
	}

	if r.claimIndex >= len(alloc.claimsToAllocate) {
		// Done! If we were doing scoring, we would compare the current allocation result
		// against the previous one, keep the best, and continue. Without scoring, we stop
		// and use the first solution.
		alloc.logger.V(6).Info("Allocation result found")
		return true, nil
	}

	claim := alloc.claimsToAllocate[r.claimIndex]
	if r.requestIndex >= len(claim.Spec.Devices.Requests) {
		// Done with the claim, continue with the next one.
		success, err := alloc.allocateOne(deviceIndices{claimIndex: r.claimIndex + 1}, false)
		if errors.Is(err, errAllocationResultMaxSizeExceeded) {
			// We don't need to propagate this further because
			// this is not a fatal error. Retrying the claim under
			// different circumstances may succeed if it uses
			// subrequests and changing the allocation of some
			// prior claim enables allocating a subrequest here
			// which needs fewer devices.
			return false, nil
		}
		return success, err
	}

	// r.subRequestIndex is zero unless the for loop below is in the
	// recursion chain.
	requestKey := requestIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex, subRequestIndex: r.subRequestIndex}
	requestData := alloc.requestData[requestKey]

	// Subrequests are special: we only need to allocate one of them, then
	// we can move on to the next request. We enter this for loop when
	// hitting the first subrequest, but not if we are already working on a
	// specific subrequest.
	if !allocateSubRequest && requestData.parentRequest != nil {
		// Keep track of whether all attempts to do allocation with the
		// subrequests results in the allocation result limit exceeded.
		// If so, there is no need to make attempts with other devices
		// in the previous request (if any), except when
		// it is a firstAvailable request where some sub-requests
		// need less devices than others.
		allAllocationExceeded := true
		for subRequestIndex := 0; ; subRequestIndex++ {
			nextSubRequestKey := requestKey
			nextSubRequestKey.subRequestIndex = subRequestIndex
			if _, ok := alloc.requestData[nextSubRequestKey]; !ok {
				// Past the end of the subrequests without finding a solution -> give up.
				//
				// Return errAllocationResultMaxSizeExceeded if all
				// attempts for the subrequests failed to due to reaching
				// the max size limit. This would mean that there are no
				// solution that involves the previous request (if any).
				if allAllocationExceeded {
					return false, errAllocationResultMaxSizeExceeded
				}
				return false, nil
			}

			r.subRequestIndex = subRequestIndex
			success, err := alloc.allocateOne(r, true /* prevent infinite recusion */)
			// If we reached the allocation result limit, we can try
			// with the next subrequest if there is one. It might request
			// fewer devices, so it might succeed.
			if errors.Is(err, errAllocationResultMaxSizeExceeded) {
				continue
			}
			// If we get here, at least one of the subrequests failed for a
			// different reason than errAllocationResultMaxSizeExceeded.
			allAllocationExceeded = false
			if err != nil {
				return false, err
			}
			// If allocation with a subrequest succeeds, return without
			// attempting the remaining subrequests.
			if success {
				// Store the index of the selected subrequest
				requestData.selectedSubRequestIndex = subRequestIndex
				alloc.requestData[requestKey] = requestData
				return true, nil
			}
		}
		// This is unreachable, so no need to have a return statement here.
	}

	// Look up the current request that we are attempting to satisfy. This can
	// be either a request or a subrequest.
	request := requestData.request
	doAllDevices := request.allocationMode() == resourceapi.DeviceAllocationModeAll

	// At least one device is required for 'All' allocation mode.
	if doAllDevices && len(requestData.allDevices) == 0 {
		alloc.logger.V(6).Info("Allocation for 'all' devices didn't succeed: no devices found", "claim", klog.KObj(claim), "request", requestData.request.name())
		return false, nil
	}

	// We already know how many devices per request are needed.
	if r.deviceIndex >= requestData.numDevices {
		// Done with request, continue with next one. We have completed the work for
		// the request or subrequest, so we can no longer be allocating devices for
		// a subrequest.
		success, err := alloc.allocateOne(deviceIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex + 1}, false)
		// We want to propagate any errAllocationResultMaxSizeExceeded to the caller. If
		// that error is returned here, it means none of the requests/subrequests after this one
		// could be allocated while staying within the limit on the number of devices, so there
		// are no solution in the current request/subrequest that would work.
		return success, err
	}

	// Before trying to allocate devices, check if allocating the devices
	// in the current request will put us over the threshold.
	// We can calculate this by adding the number of already allocated devices with the number
	// of devices in the current request, and then finally subtract the deviceIndex since we
	// don't want to double count any devices already allocated for the current request.
	numDevicesAfterAlloc := len(alloc.result[r.claimIndex].devices) + requestData.numDevices - r.deviceIndex
	if numDevicesAfterAlloc > resourceapi.AllocationResultsMaxSize {
		// Return a special error so we can identify this situation in the
		// callers and do more aggressive backtracking.
		return false, errAllocationResultMaxSizeExceeded
	}

	alloc.logger.V(6).Info("Allocating one device", "currentClaim", r.claimIndex, "totalClaims", len(alloc.claimsToAllocate), "currentRequest", r.requestIndex, "currentSubRequest", r.subRequestIndex, "totalRequestsPerClaim", len(claim.Spec.Devices.Requests), "currentDevice", r.deviceIndex, "devicesPerRequest", requestData.numDevices, "allDevices", doAllDevices, "adminAccess", request.adminAccess())
	if doAllDevices {
		// For "all" devices we already know which ones we need. We
		// just need to check whether we can use them.
		deviceWithID := requestData.allDevices[r.deviceIndex]
		success, deallocate, err := alloc.allocateDevice(r, deviceWithID, true)
		if err != nil {
			return false, err
		}
		if !success {
			// The order in which we allocate "all" devices doesn't matter,
			// so we only try with the one which was up next. If we couldn't
			// get all of them, then there is no solution and we have to stop.
			return false, nil
		}
		done, err := alloc.allocateOne(deviceIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex, deviceIndex: r.deviceIndex + 1}, allocateSubRequest)
		if err != nil || !done {
			// If we get an error or didn't complete, we need to backtrack. Depending
			// on the situation we might be able to retry, so we make sure we
			// deallocate.
			deallocate()
			return false, err
		}
		return done, nil
	}

	// We need to find suitable devices.
	for _, pool := range alloc.pools {
		// We don't allocate devices from invalid or incomplete pools, but
		// don't error out here since there might be available devices in other
		// pools.
		if pool.IsIncomplete || pool.IsInvalid {
			continue
		}
		for _, slice := range pool.DeviceSlicesTargetingNode {
			for deviceIndex := range slice.Spec.Devices {
				deviceID := DeviceID{Driver: pool.Driver, Pool: pool.Pool, Device: slice.Spec.Devices[deviceIndex].Name}

				// Checking for "in use" is cheap and thus gets done first.
				if request.adminAccess() && alloc.allocatingDeviceForClaim(deviceID, r.claimIndex) {
					alloc.logger.V(7).Info("Device in use in same claim", "device", deviceID)
					continue
				}
				if !request.adminAccess() && alloc.deviceInUse(deviceID) {
					alloc.logger.V(7).Info("Device in use", "device", deviceID)
					continue
				}

				// Next check selectors.
				requestKey := requestIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex, subRequestIndex: r.subRequestIndex}
				selectable, err := alloc.isSelectable(requestKey, requestData, slice, deviceIndex)
				if err != nil {
					return false, err
				}
				if !selectable {
					alloc.logger.V(7).Info("Device not selectable", "device", deviceID)
					continue
				}

				// Finally treat as allocated and move on to the next device.
				device := deviceWithID{
					id:     deviceID,
					Device: &slice.Spec.Devices[deviceIndex],
					slice:  slice,
					pool:   pool,
				}
				allocated, deallocate, err := alloc.allocateDevice(r, device, false)
				if err != nil {
					return false, err
				}
				if !allocated {
					// In use or constraint violated...
					alloc.logger.V(7).Info("Device not usable", "device", deviceID)
					continue
				}
				deviceKey := deviceIndices{
					claimIndex:      r.claimIndex,
					requestIndex:    r.requestIndex,
					subRequestIndex: r.subRequestIndex,
					deviceIndex:     r.deviceIndex + 1,
				}
				done, err := alloc.allocateOne(deviceKey, allocateSubRequest)
				// If we found a solution, we can stop.
				if err == nil && done {
					return done, nil
				}

				// Otherwise we didn't find a solution, and we need to deallocate
				// so the temporary allocation is correct for trying other devices.
				deallocate()

				if err != nil {
					// If we hit an error, we return. This might be that we reached
					// the allocation size limit, and if so, it will be caught further
					// up the stack and other subrequests will be attempted if there
					// are any.
					return false, err
				}
			}
		}
	}

	// If we get here without finding a solution, then there is none.
	return false, nil
}

// isSelectable checks whether a device satisfies the request and class selectors.
func (alloc *allocator) isSelectable(r requestIndices, requestData requestData, slice *draapi.ResourceSlice, deviceIndex int) (bool, error) {
	device := &slice.Spec.Devices[deviceIndex]
	if !alloc.features.DeviceBindingAndStatus &&
		len(device.BindingConditions) > 0 {
		// Devices with binding conditions are not supported, feature is off.
		return false, nil
	}

	deviceID := DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: slice.Spec.Devices[deviceIndex].Name}
	matchKey := matchKey{DeviceID: deviceID, requestIndices: r}
	if matches, ok := alloc.deviceMatchesRequest[matchKey]; ok {
		// No need to check again.
		return matches, nil
	}

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

	request := requestData.request
	match, err := alloc.selectorsMatch(r, device, deviceID, nil, request.selectors())
	if err != nil {
		return false, err
	}
	if !match {
		alloc.deviceMatchesRequest[matchKey] = false
		return false, nil
	}

	if ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
		matches, err := NodeMatches(alloc.node, ptr.Deref(device.NodeName, ""), ptr.Deref(device.AllNodes, false), device.NodeSelector)
		if err != nil {
			return false, err
		}
		if !matches {
			alloc.deviceMatchesRequest[matchKey] = false
			return false, nil
		}
	}

	alloc.deviceMatchesRequest[matchKey] = true
	return true, nil

}

func (alloc *allocator) selectorsMatch(r requestIndices, device *draapi.Device, deviceID DeviceID, class *resourceapi.DeviceClass, selectors []resourceapi.DeviceSelector) (bool, error) {
	for i, selector := range selectors {
		expr := alloc.celCache.GetOrCompile(selector.CEL.Expression)
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

		// If this conversion turns out to be expensive, the CEL package could be converted
		// to use unique strings.
		var d resourceapi.Device
		if err := draapi.Convert_api_Device_To_v1_Device(device, &d, nil); err != nil {
			return false, fmt.Errorf("convert Device: %w", err)
		}
		matches, details, err := expr.DeviceMatches(alloc.ctx, cel.Device{Driver: deviceID.Driver.String(), Attributes: d.Attributes, Capacity: d.Capacity})
		if class != nil {
			alloc.logger.V(7).Info("CEL result", "device", deviceID, "class", klog.KObj(class), "selector", i, "expression", selector.CEL.Expression, "matches", matches, "actualCost", ptr.Deref(details.ActualCost(), 0), "err", err)
		} else {
			alloc.logger.V(7).Info("CEL result", "device", deviceID, "claim", klog.KObj(alloc.claimsToAllocate[r.claimIndex]), "selector", i, "expression", selector.CEL.Expression, "actualCost", ptr.Deref(details.ActualCost(), 0), "matches", matches, "err", err)
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
// candidate. The device must be selectable.
//
// If that candidate works out okay, the shared state gets updated
// as if that candidate had been allocated. If allocation cannot continue later
// and must try something else, then the rollback function can be invoked to
// restore the previous state.
//
// The rollback function is only provided in case of a successful allocation
// (true and no error).
func (alloc *allocator) allocateDevice(r deviceIndices, device deviceWithID, must bool) (bool, func(), error) {
	claim := alloc.claimsToAllocate[r.claimIndex]
	requestKey := requestIndices{claimIndex: r.claimIndex, requestIndex: r.requestIndex, subRequestIndex: r.subRequestIndex}
	requestData := alloc.requestData[requestKey]
	request := requestData.request
	if request.adminAccess() && alloc.allocatingDeviceForClaim(device.id, r.claimIndex) {
		alloc.logger.V(7).Info("Device in use in same claim", "device", device.id)
		return false, nil, nil
	}
	if !request.adminAccess() && alloc.deviceInUse(device.id) {
		alloc.logger.V(7).Info("Device in use", "device", device.id)
		return false, nil, nil
	}

	// Devices that consume counters can not be allocated if the PartitionableDevices feature
	// is not enabled.
	if !alloc.features.PartitionableDevices && len(device.ConsumesCounters) > 0 {
		alloc.logger.V(7).Info("Device consumes counters, but the partitionable devices feature is not enabled", "device", device.id)
		return false, nil, nil
	}

	// The API validation logic has checked the ConsumesCounters referred should exist inside SharedCounters.
	if len(device.ConsumesCounters) > 0 {
		// If a device consumes counters from a counter set, verify that
		// there is sufficient counters available.
		ok, err := alloc.checkAvailableCounters(device)
		if err != nil {
			return false, nil, err
		}
		if !ok {
			alloc.logger.V(7).Info("Insufficient counters", "device", device.id)
			return false, nil, nil
		}
	}

	var parentRequestName string
	var baseRequestName string
	var subRequestName string
	if requestData.parentRequest == nil {
		baseRequestName = requestData.request.name()
	} else {
		parentRequestName = requestData.parentRequest.name()
		baseRequestName = parentRequestName
		subRequestName = requestData.request.name()
	}

	// Might be tainted, in which case the taint has to be tolerated.
	// The check is skipped if the feature is disabled.
	if alloc.features.DeviceTaints && taintPreventsAllocation(device.Device, request) {
		return false, nil, nil
	}

	// It's available. Now check constraints.
	for i, constraint := range alloc.constraints[r.claimIndex] {
		added := constraint.add(baseRequestName, subRequestName, device.Device, device.id)
		if !added {
			if must {
				// It does not make sense to declare a claim where a constraint prevents getting
				// all devices. Treat this as an error.
				return false, nil, fmt.Errorf("claim %s, request %s: cannot add device %s because a claim constraint would not be satisfied", klog.KObj(claim), request.name(), device.id)
			}

			// Roll back for all previous constraints before we return.
			for e := 0; e < i; e++ {
				alloc.constraints[r.claimIndex][e].remove(baseRequestName, subRequestName, device.Device, device.id)
			}
			return false, nil, nil
		}
	}

	// All constraints satisfied. Mark as in use (unless we do admin access)
	// and record the result.
	alloc.logger.V(7).Info("Device allocated", "device", device.id)

	if alloc.allocatingDevices[device.id] == nil {
		alloc.allocatingDevices[device.id] = make(sets.Set[int])
	}
	alloc.allocatingDevices[device.id].Insert(r.claimIndex)

	result := internalDeviceResult{
		request:       request.name(),
		parentRequest: parentRequestName,
		id:            device.id,
		Device:        device.Device,
		slice:         device.slice,
	}
	if request.adminAccess() {
		result.adminAccess = ptr.To(request.adminAccess())
	}
	previousNumResults := len(alloc.result[r.claimIndex].devices)
	alloc.result[r.claimIndex].devices = append(alloc.result[r.claimIndex].devices, result)

	return true, func() {
		for _, constraint := range alloc.constraints[r.claimIndex] {
			constraint.remove(baseRequestName, subRequestName, device.Device, device.id)
		}
		alloc.allocatingDevices[device.id].Delete(r.claimIndex)
		if alloc.features.PartitionableDevices && len(device.ConsumesCounters) > 0 {
			alloc.deallocateCountersForDevice(device)
		}
		// Truncate, but keep the underlying slice.
		alloc.result[r.claimIndex].devices = alloc.result[r.claimIndex].devices[:previousNumResults]
		alloc.logger.V(7).Info("Device deallocated", "device", device.id)
	}, nil
}

func taintPreventsAllocation(device *draapi.Device, request requestAccessor) bool {
	for _, taint := range device.Taints {
		switch taint.Effect {
		// Only known effects prevent allocation, others (including None) are ignored.
		case resourceapi.DeviceTaintEffectNoExecute, resourceapi.DeviceTaintEffectNoSchedule:
			if !taintTolerated(taint, request) {
				return true
			}
		}
	}
	return false
}

func taintTolerated(taint resourceapi.DeviceTaint, request requestAccessor) bool {
	for _, toleration := range request.tolerations() {
		if resourceclaim.ToleratesTaint(toleration, taint) {
			return true
		}
	}
	return false
}

// checkAvailableCounters checks if there are enough counters available to allocate
// the specified device.
//
// Gets called only if the partitionable devices feature is enabled and the device
// consumes counters.
func (alloc *allocator) checkAvailableCounters(device deviceWithID) (bool, error) {
	pool := device.pool
	poolName := pool.PoolID.Pool

	// Check first if the available counters for this pool have already been
	// calculated.
	alloc.mutex.RLock()
	availableCountersForPool, found := alloc.availableCounters[poolName]
	alloc.mutex.RUnlock()
	// If not, we need to do it now. But we store the result so it doesn't need
	// to be calculated again.
	// Since this is computed without holding the lock on the mutex, other goroutines
	// might also do this work. But the input will be the same to all of them, so
	// the result will also always be the same.
	if !found {
		availableCountersForPool = make(counterSets, len(pool.CounterSets))
		for _, counterSet := range pool.CounterSets {
			availableCountersForCounterSet := make(map[string]resourceapi.Counter, len(counterSet.Counters))
			for name, c := range counterSet.Counters {
				availableCountersForCounterSet[name] = c
			}
			availableCountersForPool[counterSet.Name] = availableCountersForCounterSet
		}

		// Update the data structure to reflect counters already consumed by allocated devices. This
		// only includes devices where the allocation process has completed, so this will never
		// change during the allocation process.
		for _, resourceSlices := range [][]*draapi.ResourceSlice{pool.DeviceSlicesTargetingNode, pool.DeviceSlicesNotTargetingNode} {
			for _, slice := range resourceSlices {
				for _, device := range slice.Spec.Devices {
					deviceID := DeviceID{
						Driver: slice.Spec.Driver,
						Pool:   slice.Spec.Pool.Name,
						Device: device.Name,
					}
					// Devices that aren't allocated doesn't consume any counters, so we don't
					// need to consider them.
					if !alloc.allocatedDevices.Has(deviceID) {
						continue
					}
					for _, deviceCounterConsumption := range device.ConsumesCounters {
						availableCountersForCounterSet := availableCountersForPool[deviceCounterConsumption.CounterSet]
						for name, c := range deviceCounterConsumption.Counters {
							existingCounter, ok := availableCountersForCounterSet[name]
							if !ok {
								// the API validation logic has been added to make sure the counters referred should exist in counter sets.
								continue
							}
							// This can potentially result in negative available counters. That is fine,
							// we just treat it as no counters available.
							existingCounter.Value.Sub(c.Value)
							availableCountersForCounterSet[name] = existingCounter
						}
					}
					// Note that we don't include devices in the alloc.allocatingDevices here since
					// counters consumed by devices for the current claims are tracked in
					// alloc.consumedCounters
				}
			}
		}

		// Set the available counters on the allocator so we don't have to
		// compute this again.
		alloc.mutex.Lock()
		alloc.availableCounters[poolName] = availableCountersForPool
		alloc.mutex.Unlock()
	}

	// Update the consumedCounters data structure with the counters consumed
	// by the current device.
	consumedCountersForPool, found := alloc.consumedCounters[poolName]
	// If no devices in the allocating state have consumed any counters from the current
	// pool, initialize the data structure.
	if !found {
		consumedCountersForPool = make(counterSets)
		alloc.consumedCounters[poolName] = consumedCountersForPool
	}
	for _, deviceCounterConsumption := range device.ConsumesCounters {
		consumedCountersForCounterSet, found := consumedCountersForPool[deviceCounterConsumption.CounterSet]
		if !found {
			consumedCountersForCounterSet = make(map[string]resourceapi.Counter)
			consumedCountersForPool[deviceCounterConsumption.CounterSet] = consumedCountersForCounterSet
		}
		for name, c := range deviceCounterConsumption.Counters {
			consumedCounters, found := consumedCountersForCounterSet[name]
			if !found {
				consumedCountersForCounterSet[name] = c
				continue
			}
			consumedCounters.Value.Add(c.Value)
			consumedCountersForCounterSet[name] = consumedCounters
		}
	}

	// Check that we didn't exceed the availability of any counters by allocating
	// the current device. If we did, the current set of devices doesn't work, so we
	// update the consumed counters to no longer reflect the current device.
	for availableCounterSetName, availableCounters := range availableCountersForPool {
		consumedCounters := consumedCountersForPool[availableCounterSetName]
		for availableCounterName, availableCounter := range availableCounters {
			consumedCounter := consumedCounters[availableCounterName]
			if availableCounter.Value.Cmp(consumedCounter.Value) < 0 {
				alloc.deallocateCountersForDevice(device)
				return false, nil
			}
		}
	}

	return true, nil
}

func (alloc *allocator) deviceInUse(deviceID DeviceID) bool {
	return alloc.allocatedDevices.Has(deviceID) || alloc.allocatingDeviceForAnyClaim(deviceID)
}

func (alloc *allocator) allocatingDeviceForAnyClaim(deviceID DeviceID) bool {
	return alloc.allocatingDevices[deviceID].Len() > 0
}

func (alloc *allocator) allocatingDeviceForClaim(deviceID DeviceID, claimIndex int) bool {
	return alloc.allocatingDevices[deviceID].Has(claimIndex)
}

// deallocateCountersForDevice subtracts the consumed counters of the provided
// device from the consumedCounters data structure.
func (alloc *allocator) deallocateCountersForDevice(device deviceWithID) {
	poolName := device.pool.PoolID.Pool

	consumedCountersForPool := alloc.consumedCounters[poolName]
	for _, deviceCounterConsumption := range device.ConsumesCounters {
		counterSetName := deviceCounterConsumption.CounterSet
		consumedCounterSet := consumedCountersForPool[counterSetName]
		for name, c := range deviceCounterConsumption.Counters {
			consumedCounter := consumedCounterSet[name]
			consumedCounter.Value.Sub(c.Value)
			consumedCounterSet[name] = consumedCounter
		}
	}
}

// createNodeSelector constructs a node selector for the allocation, if needed,
// otherwise it returns nil.
func (alloc *allocator) createNodeSelector(result []internalDeviceResult) (*v1.NodeSelector, error) {
	// Selector with one term. That term gets extended with additional
	// requirements from the different devices.
	ns := &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{{}},
	}

	for i := range result {
		slice := result[i].slice
		var nodeName *string
		var nodeSelector *v1.NodeSelector
		if ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
			nodeName = result[i].NodeName
			nodeSelector = result[i].NodeSelector
		} else {
			nodeName = slice.Spec.NodeName
			nodeSelector = slice.Spec.NodeSelector
		}
		if nodeName != nil {
			// At least one device is local to one node. This
			// restricts the allocation to that node.
			return &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{{
					MatchFields: []v1.NodeSelectorRequirement{{
						Key:      "metadata.name",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{*nodeName},
					}},
				}},
			}, nil
		}
		if nodeSelector != nil {
			switch len(nodeSelector.NodeSelectorTerms) {
			case 0:
				// Nothing?
			case 1:
				// Add all terms if they are not present already.
				addNewNodeSelectorRequirements(nodeSelector.NodeSelectorTerms[0].MatchFields, &ns.NodeSelectorTerms[0].MatchFields)
				addNewNodeSelectorRequirements(nodeSelector.NodeSelectorTerms[0].MatchExpressions, &ns.NodeSelectorTerms[0].MatchExpressions)
			default:
				// This shouldn't occur, validation must prevent creation of such slices.
				return nil, fmt.Errorf("unsupported ResourceSlice.NodeSelector with %d terms", len(nodeSelector.NodeSelectorTerms))
			}
		}
	}

	if len(ns.NodeSelectorTerms[0].MatchFields) > 0 || len(ns.NodeSelectorTerms[0].MatchExpressions) > 0 {
		// We have a valid node selector.
		return ns, nil
	}

	// Available everywhere.
	return nil, nil
}

// requestAccessor is an interface for accessing either
// DeviceRequests or DeviceSubRequests. It lets most
// of the allocator code work with either DeviceRequests
// or DeviceSubRequests.
type requestAccessor interface {
	name() string
	deviceClassName() string
	allocationMode() resourceapi.DeviceAllocationMode
	count() int64
	adminAccess() bool
	hasAdminAccess() bool
	selectors() []resourceapi.DeviceSelector
	tolerations() []resourceapi.DeviceToleration
}

// exactDeviceRequestAccessor is an implementation of the
// requestAccessor interface for ExactDeviceRequests.
type exactDeviceRequestAccessor struct {
	request *resourceapi.DeviceRequest
}

func (d *exactDeviceRequestAccessor) name() string {
	return d.request.Name
}

func (d *exactDeviceRequestAccessor) deviceClassName() string {
	return d.request.Exactly.DeviceClassName
}

func (d *exactDeviceRequestAccessor) allocationMode() resourceapi.DeviceAllocationMode {
	return d.request.Exactly.AllocationMode
}

func (d *exactDeviceRequestAccessor) count() int64 {
	return d.request.Exactly.Count
}

func (d *exactDeviceRequestAccessor) adminAccess() bool {
	return ptr.Deref(d.request.Exactly.AdminAccess, false)
}

func (d *exactDeviceRequestAccessor) hasAdminAccess() bool {
	return d.request.Exactly.AdminAccess != nil
}

func (d *exactDeviceRequestAccessor) selectors() []resourceapi.DeviceSelector {
	return d.request.Exactly.Selectors
}

func (d *exactDeviceRequestAccessor) tolerations() []resourceapi.DeviceToleration {
	return d.request.Exactly.Tolerations
}

// deviceSubRequestAccessor is an implementation of the
// requestAccessor interface for DeviceSubRequests.
type deviceSubRequestAccessor struct {
	subRequest *resourceapi.DeviceSubRequest
}

func (d *deviceSubRequestAccessor) name() string {
	return d.subRequest.Name
}

func (d *deviceSubRequestAccessor) deviceClassName() string {
	return d.subRequest.DeviceClassName
}

func (d *deviceSubRequestAccessor) allocationMode() resourceapi.DeviceAllocationMode {
	return d.subRequest.AllocationMode
}

func (d *deviceSubRequestAccessor) count() int64 {
	return d.subRequest.Count
}

func (d *deviceSubRequestAccessor) adminAccess() bool {
	return false
}

func (d *deviceSubRequestAccessor) hasAdminAccess() bool {
	return false
}

func (d *deviceSubRequestAccessor) selectors() []resourceapi.DeviceSelector {
	return d.subRequest.Selectors
}

func (d *deviceSubRequestAccessor) tolerations() []resourceapi.DeviceToleration {
	return d.subRequest.Tolerations
}

func addNewNodeSelectorRequirements(from []v1.NodeSelectorRequirement, to *[]v1.NodeSelectorRequirement) {
	for _, requirement := range from {
		if !containsNodeSelectorRequirement(*to, requirement) {
			*to = append(*to, requirement)
		}
	}
}

func containsNodeSelectorRequirement(requirements []v1.NodeSelectorRequirement, requirement v1.NodeSelectorRequirement) bool {
	values := sets.New(requirement.Values...)
	for _, existingRequirement := range requirements {
		if existingRequirement.Key != requirement.Key {
			continue
		}
		if existingRequirement.Operator != requirement.Operator {
			continue
		}
		if !sets.New(existingRequirement.Values...).Equal(values) {
			continue
		}
		return true
	}
	return false
}
