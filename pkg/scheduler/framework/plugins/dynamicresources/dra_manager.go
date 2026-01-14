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

package dynamicresources

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/go-logr/logr"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	resourcelisters "k8s.io/client-go/listers/resource/v1"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

var _ fwk.SharedDRAManager = &DefaultDRAManager{}

// DefaultDRAManager is the default implementation of SharedDRAManager. It obtains the DRA objects
// from API informers, and uses an AssumeCache and a map of in-flight allocations in order
// to avoid race conditions when modifying ResourceClaims.
type DefaultDRAManager struct {
	resourceClaimTracker  *claimTracker
	resourceSliceLister   *resourceSliceLister
	deviceClassLister     *deviceClassLister
	extendedResourceCache *extendedresourcecache.ExtendedResourceCache
}

func NewDRAManager(ctx context.Context, claimsCache *assumecache.AssumeCache, resourceSliceTracker *resourceslicetracker.Tracker, informerFactory informers.SharedInformerFactory) *DefaultDRAManager {
	logger := klog.FromContext(ctx)
	manager := &DefaultDRAManager{
		resourceClaimTracker: &claimTracker{
			cache:               claimsCache,
			inFlightAllocations: &sync.Map{},
			allocatedDevices:    newAllocatedDevices(logger),
			logger:              logger,
		},
		resourceSliceLister: &resourceSliceLister{tracker: resourceSliceTracker},
		deviceClassLister:   &deviceClassLister{classLister: informerFactory.Resource().V1().DeviceClasses().Lister()},
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.DRAExtendedResource) {
		manager.extendedResourceCache = extendedresourcecache.NewExtendedResourceCache(logger)
	}

	// Reacting to events is more efficient than iterating over the list
	// repeatedly in PreFilter.
	manager.resourceClaimTracker.cache.AddEventHandler(manager.resourceClaimTracker.allocatedDevices.handlers())

	return manager
}

func (s *DefaultDRAManager) ResourceClaims() fwk.ResourceClaimTracker {
	return s.resourceClaimTracker
}

func (s *DefaultDRAManager) ResourceSlices() fwk.ResourceSliceLister {
	return s.resourceSliceLister
}

func (s *DefaultDRAManager) DeviceClasses() fwk.DeviceClassLister {
	return s.deviceClassLister
}

// DeviceClassResolver will always return a valid interface implementation. It
// wraps a nil extendedresourcecache.ExtendedResourceCache if the feature is
// disabled.
//
// That's okay, extendedresourcecache.ExtendedResourceCache.GetDeviceClass
// returns nil if called for nil.
func (s *DefaultDRAManager) DeviceClassResolver() fwk.DeviceClassResolver {
	return s.extendedResourceCache
}

var _ fwk.ResourceSliceLister = &resourceSliceLister{}

type resourceSliceLister struct {
	tracker *resourceslicetracker.Tracker
}

func (l *resourceSliceLister) ListWithDeviceTaintRules() ([]*resourceapi.ResourceSlice, error) {
	return l.tracker.ListPatchedResourceSlices()
}

var _ fwk.DeviceClassLister = &deviceClassLister{}

type deviceClassLister struct {
	classLister resourcelisters.DeviceClassLister
}

func (l *deviceClassLister) Get(className string) (*resourceapi.DeviceClass, error) {
	return l.classLister.Get(className)
}

func (l *deviceClassLister) List() ([]*resourceapi.DeviceClass, error) {
	return l.classLister.List(labels.Everything())
}

var _ fwk.ResourceClaimTracker = &claimTracker{}

type claimTracker struct {
	// cache enables temporarily storing a newer claim object
	// while the scheduler has allocated it and the corresponding object
	// update from the apiserver has not been processed by the claim
	// informer callbacks. ResourceClaimTracker get added here in PreBind and removed by
	// the informer callback (based on the "newer than" comparison in the
	// assume cache).
	//
	// It uses cache.MetaNamespaceKeyFunc to generate object names, which
	// therefore are "<namespace>/<name>".
	//
	// This is necessary to ensure that reconstructing the resource usage
	// at the start of a pod scheduling cycle doesn't reuse the resources
	// assigned to such a claim. Alternatively, claim allocation state
	// could also get tracked across pod scheduling cycles, but that
	// - adds complexity (need to carefully sync state with informer events
	//   for claims and ResourceSlices)
	// - would make integration with cluster autoscaler harder because it would need
	//   to trigger informer callbacks.
	cache *assumecache.AssumeCache
	// inFlightAllocations is a map from claim UUIDs to claim objects for those claims
	// for which allocation was triggered during a scheduling cycle and the
	// corresponding claim status update call in PreBind has not been done
	// yet. If another pod needs the claim, the pod is treated as "not
	// schedulable yet". The cluster event for the claim status update will
	// make it schedulable.
	//
	// This mechanism avoids the following problem:
	// - Pod A triggers allocation for claim X.
	// - Pod B shares access to that claim and gets scheduled because
	//   the claim is assumed to be allocated.
	// - PreBind for pod B is called first, tries to update reservedFor and
	//   fails because the claim is not really allocated yet.
	//
	// We could avoid the ordering problem by allowing either pod A or pod B
	// to set the allocation. But that is more complicated and leads to another
	// problem:
	// - Pod A and B get scheduled as above.
	// - PreBind for pod A gets called first, then fails with a temporary API error.
	//   It removes the updated claim from the assume cache because of that.
	// - PreBind for pod B gets called next and succeeds with adding the
	//   allocation and its own reservedFor entry.
	// - The assume cache is now not reflecting that the claim is allocated,
	//   which could lead to reusing the same resource for some other claim.
	//
	// A sync.Map is used because in practice sharing of a claim between
	// pods is expected to be rare compared to per-pod claim, so we end up
	// hitting the "multiple goroutines read, write, and overwrite entries
	// for disjoint sets of keys" case that sync.Map is optimized for.
	inFlightAllocations *sync.Map
	allocatedDevices    *allocatedDevices
	logger              klog.Logger
}

func (c *claimTracker) ClaimHasPendingAllocation(claimUID types.UID) bool {
	_, found := c.inFlightAllocations.Load(claimUID)
	return found
}

func (c *claimTracker) SignalClaimPendingAllocation(claimUID types.UID, allocatedClaim *resourceapi.ResourceClaim) error {
	c.inFlightAllocations.Store(claimUID, allocatedClaim)
	// This is the same verbosity as the corresponding log in the assume cache.
	c.logger.V(5).Info("Added in-flight claim", "claim", klog.KObj(allocatedClaim), "uid", claimUID, "version", allocatedClaim.ResourceVersion)
	// There's no reason to return an error in this implementation, but the error is helpful for other implementations.
	// For example, implementations that have to deal with fake claims might want to return an error if the allocation
	// is for an invalid claim.
	return nil
}

func (c *claimTracker) RemoveClaimPendingAllocation(claimUID types.UID) (deleted bool) {
	claim, found := c.inFlightAllocations.LoadAndDelete(claimUID)
	// The assume cache doesn't log this, but maybe it should.
	if found {
		claim := claim.(*resourceapi.ResourceClaim)
		c.logger.V(5).Info("Removed in-flight claim", "claim", klog.KObj(claim), "uid", claimUID, "version", claim.ResourceVersion)
	} else {
		c.logger.V(5).Info("Redundant remove of in-flight claim, not found", "uid", claimUID)
	}
	return found
}

func (c *claimTracker) Get(namespace, claimName string) (*resourceapi.ResourceClaim, error) {
	obj, err := c.cache.Get(namespace + "/" + claimName)
	if err != nil {
		return nil, err
	}
	claim, ok := obj.(*resourceapi.ResourceClaim)
	if !ok {
		return nil, fmt.Errorf("unexpected object type %T for assumed object %s/%s", obj, namespace, claimName)
	}
	return claim, nil
}

func (c *claimTracker) List() ([]*resourceapi.ResourceClaim, error) {
	var result []*resourceapi.ResourceClaim
	// Probably not worth adding an index for?
	objs := c.cache.List(nil)
	for _, obj := range objs {
		claim, ok := obj.(*resourceapi.ResourceClaim)
		if ok {
			result = append(result, claim)
		}
	}
	return result, nil
}

// errClaimTrackerConcurrentModification gets returned if ListAllAllocatedDevices
// or GatherAllocatedState need to be retried.
//
// There is a rare race when a claim is initially in-flight:
// - allocated is created from cache (claim not there)
// - someone removes from the in-flight claims and adds to the cache
// - we start checking in-flight claims (claim not there anymore)
// => claim ignored
//
// A proper fix would be to rewrite the assume cache, allocatedDevices,
// and the in-flight map so that they are under a single lock. But that's
// a pretty big change and prevents reusing the assume cache. So instead
// we check for changes in the set of allocated devices and keep trying
// until we get an attempt with no concurrent changes.
//
// A claim being first in the cache, then only in-flight cannot happen,
// so we don't need to re-check the in-flight claims.
var errClaimTrackerConcurrentModification = errors.New("conflicting concurrent modification")

func (c *claimTracker) ListAllAllocatedDevices() (a sets.Set[structured.DeviceID], err error) {
	c.logger.V(6).Info("Starting ListAllAllocatedDevices")
	defer func() {
		c.logger.V(6).Info("Finished ListAllAllocatedDevices", "allocatedDevices", logAllocatedDevices(c.logger, a), "err", err)
	}()

	// Start with a fresh set that matches the current known state of the
	// world according to the informers.
	allocated, revision := c.allocatedDevices.Get()

	// Whatever is in flight also has to be checked.
	c.inFlightAllocations.Range(func(key, value any) bool {
		claim := value.(*resourceapi.ResourceClaim)
		foreachAllocatedDevice(claim, func(deviceID structured.DeviceID) {
			c.logger.V(6).Info("Device is in flight for allocation", "device", deviceID, "claim", klog.KObj(claim))
			allocated.Insert(deviceID)
		}, false, func(structured.SharedDeviceID) {}, func(structured.DeviceConsumedCapacity) {})
		return true
	})

	if revision == c.allocatedDevices.Revision() {
		// Our current result is valid, nothing changed in the meantime.
		return allocated, nil
	}

	return nil, errClaimTrackerConcurrentModification
}

func (c *claimTracker) GatherAllocatedState() (s *structured.AllocatedState, err error) {
	c.logger.V(6).Info("Starting GatherAllocatedState")
	defer func() {
		var a sets.Set[structured.DeviceID]
		if s != nil {
			a = s.AllocatedDevices
		}
		c.logger.V(6).Info("Finished GatherAllocatedState", "allocatedDevices", logAllocatedDevices(c.logger, a), "err", err)
	}()

	// Start with a fresh set that matches the current known state of the
	// world according to the informers.
	allocated, revision1 := c.allocatedDevices.Get()
	allocatedSharedDeviceIDs := sets.New[structured.SharedDeviceID]()
	aggregatedCapacity, revision2 := c.allocatedDevices.Capacities()

	if revision1 != revision2 {
		// Already not consistent. Try again.
		return nil, errClaimTrackerConcurrentModification
	}

	enabledConsumableCapacity := utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity)

	// Whatever is in flight also has to be checked.
	c.inFlightAllocations.Range(func(key, value any) bool {
		claim := value.(*resourceapi.ResourceClaim)
		foreachAllocatedDevice(claim, func(deviceID structured.DeviceID) {
			c.logger.V(6).Info("Device is in flight for allocation", "device", deviceID, "claim", klog.KObj(claim))
			allocated.Insert(deviceID)
		},
			enabledConsumableCapacity,
			func(sharedDeviceID structured.SharedDeviceID) {
				c.logger.V(6).Info("Device is in flight for allocation", "shared device", sharedDeviceID, "claim", klog.KObj(claim))
				allocatedSharedDeviceIDs.Insert(sharedDeviceID)
			}, func(capacity structured.DeviceConsumedCapacity) {
				c.logger.V(6).Info("Device is in flight for allocation", "consumed capacity", capacity, "claim", klog.KObj(claim))
				aggregatedCapacity.Insert(capacity)
			})
		return true
	})

	if revision1 == c.allocatedDevices.Revision() {
		// Our current result is valid, nothing changed in the meantime.
		return &structured.AllocatedState{
			AllocatedDevices:         allocated,
			AllocatedSharedDeviceIDs: allocatedSharedDeviceIDs,
			AggregatedCapacity:       aggregatedCapacity,
		}, nil
	}

	return nil, errClaimTrackerConcurrentModification
}

func (c *claimTracker) AssumeClaimAfterAPICall(claim *resourceapi.ResourceClaim) error {
	return c.cache.Assume(claim)
}

func (c *claimTracker) AssumedClaimRestore(namespace, claimName string) {
	c.cache.Restore(namespace + "/" + claimName)
}

// At V(6), log only a limited number of devices to avoid blowing up logs. For
// many E2E tests, 10 devices is enough for all devices without having to
// truncate, at least when running the tests sequentially.
const maxDevicesLevel6 = 10

// logAllocatedDevices returns a handle for the value in a structured log call which
// includes varying amounts of information about the allocated devices, depending on
// the verbosity of the logger.
func logAllocatedDevices(logger klog.Logger, allocatedDevices sets.Set[structured.DeviceID]) any {
	// We need to check verbosity here because our caller's source code
	// location may be relevant (-vmodule !).
	helper, logger := logger.WithCallStackHelper()
	helper()

	// We always produce the same output at V <= 5. 6 adds all IDs.
	verbosity := 5
	for i := 7; i > verbosity; i-- {
		if loggerV := logger.V(i); loggerV.Enabled() {
			verbosity = i
			break
		}
	}

	return &allocatedDevicesLogger{verbosity, allocatedDevices}
}

type allocatedDevicesLogger struct {
	verbosity int
	devices   sets.Set[structured.DeviceID]
}

var _ logr.Marshaler = &allocatedDevicesLogger{}

func (a *allocatedDevicesLogger) MarshalLog() any {
	if a.verbosity < 6 {
		return nil
	}

	info := struct {
		Count   int      `json:"count"`
		Devices []string `json:"devices"`
	}{
		Count: len(a.devices),
	}
	ids := make([]string, 0, len(a.devices))
	for id := range a.devices {
		ids = append(ids, id.String())
	}
	slices.Sort(ids)
	if a.verbosity == 6 && len(ids) > maxDevicesLevel6 {
		truncated := make([]string, 0, maxDevicesLevel6+1)
		truncated = append(truncated, ids[:maxDevicesLevel6/2]...)
		truncated = append(truncated, "...")
		truncated = append(truncated, ids[len(ids)-maxDevicesLevel6/2:]...)
		ids = truncated
	}
	info.Devices = ids

	return info
}
