/*
Copyright 2019 The Kubernetes Authors.

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

package benchmark

import (
	"context"
	"fmt"
	"math/rand/v2"
	"reflect"
	"sync"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/dynamic-resource-allocation/cel"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

// createResourceClaimsOp defines an op where resource claims are created.
type createResourceClaimsOp struct {
	// Must be createResourceClaimsOpcode.
	Opcode operationCode
	// Number of claims to create. Parameterizable through CountParam.
	Count int
	// Template parameter for Count.
	CountParam string
	// Namespace the claims should be created in.
	Namespace string
	// Path to spec file describing the claims to create.
	TemplatePath string
}

var _ realOp = &createResourceClaimsOp{}
var _ runnableOp = &createResourceClaimsOp{}

func (op *createResourceClaimsOp) isValid(allowParameterization bool) error {
	if !isValidCount(allowParameterization, op.Count, op.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", op.Count, op.CountParam)
	}
	if op.Namespace == "" {
		return fmt.Errorf("Namespace must be set")
	}
	if op.TemplatePath == "" {
		return fmt.Errorf("TemplatePath must be set")
	}
	return nil
}

func (op *createResourceClaimsOp) collectsMetrics() bool {
	return false
}
func (op *createResourceClaimsOp) patchParams(w *Workload) (realOp, error) {
	if op.CountParam != "" {
		var err error
		op.Count, err = w.Params.get(op.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	return op, op.isValid(false)
}

func (op *createResourceClaimsOp) requiredNamespaces() []string {
	return []string{op.Namespace}
}

func (op *createResourceClaimsOp) run(tCtx ktesting.TContext) {
	tCtx.Logf("creating %d claims in namespace %q", op.Count, op.Namespace)

	var claimTemplate *resourceapi.ResourceClaim
	if err := getSpecFromFile(&op.TemplatePath, &claimTemplate); err != nil {
		tCtx.Fatalf("parsing ResourceClaim %q: %v", op.TemplatePath, err)
	}
	var createErr error
	var mutex sync.Mutex
	create := func(i int) {
		err := func() error {
			if _, err := tCtx.Client().ResourceV1().ResourceClaims(op.Namespace).Create(tCtx, claimTemplate.DeepCopy(), metav1.CreateOptions{}); err != nil {
				return fmt.Errorf("create claim: %v", err)
			}
			return nil
		}()
		if err != nil {
			mutex.Lock()
			defer mutex.Unlock()
			createErr = err
		}
	}

	workers := min(op.Count, 30)
	workqueue.ParallelizeUntil(tCtx, workers, op.Count, create)
	if createErr != nil {
		tCtx.Fatal(createErr.Error())
	}
}

// allocResourceClaimsOp defines an op where resource claims with structured
// parameters get allocated without being associated with a pod.
type allocResourceClaimsOp struct {
	// Must be allocResourceClaimsOpcode.
	Opcode operationCode
	// Namespace where claims are to be allocated, all namespaces if empty.
	Namespace string
}

var _ realOp = &allocResourceClaimsOp{}
var _ runnableOp = &allocResourceClaimsOp{}

func (op *allocResourceClaimsOp) isValid(allowParameterization bool) error {
	return nil
}

func (op *allocResourceClaimsOp) collectsMetrics() bool {
	return false
}
func (op *allocResourceClaimsOp) patchParams(w *Workload) (realOp, error) {
	return op, op.isValid(false)
}

func (op *allocResourceClaimsOp) requiredNamespaces() []string { return nil }

func (op *allocResourceClaimsOp) run(tCtx ktesting.TContext) {
	claims, err := tCtx.Client().ResourceV1().ResourceClaims(op.Namespace).List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list claims")
	tCtx.Logf("allocating %d ResourceClaims", len(claims.Items))
	tCtx = tCtx.WithCancel()
	defer tCtx.Cancel("allocResourceClaimsOp.run is done")

	// Track cluster state.
	informerFactory := informers.NewSharedInformerFactory(tCtx.Client(), 0)
	claimInformer := informerFactory.Resource().V1().ResourceClaims().Informer()
	nodeLister := informerFactory.Core().V1().Nodes().Lister()
	resourceSliceTrackerOpts := resourceslicetracker.Options{
		EnableDeviceTaintRules:   utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceTaintRules),
		EnableConsumableCapacity: utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity),
		SliceInformer:            informerFactory.Resource().V1().ResourceSlices(),
		KubeClient:               tCtx.Client(),
	}
	if resourceSliceTrackerOpts.EnableDeviceTaintRules {
		resourceSliceTrackerOpts.TaintInformer = informerFactory.Resource().V1().DeviceTaintRules()
	}
	resourceSliceTracker, err := resourceslicetracker.StartTracker(tCtx, resourceSliceTrackerOpts)
	tCtx.ExpectNoError(err, "start resource slice tracker")
	assumeCache := assumecache.NewAssumeCache(tCtx.Logger(), claimInformer, "ResourceClaim", "", nil)
	draManager := dynamicresources.NewDRAManager(tCtx, assumeCache, resourceSliceTracker, informerFactory)
	informerFactory.Start(tCtx.Done())
	defer func() {
		tCtx.Cancel("allocResourceClaimsOp.run is shutting down")
		informerFactory.Shutdown()
	}()
	syncResult := informerFactory.WaitForCacheSyncWithContext(tCtx)
	expectSyncResult := cache.SyncResult{
		Synced: map[reflect.Type]bool{
			reflect.TypeFor[*resourceapi.DeviceClass]():   true,
			reflect.TypeFor[*resourceapi.ResourceClaim](): true,
			reflect.TypeFor[*resourceapi.ResourceSlice](): true,
			reflect.TypeFor[*v1.Node]():                   true,
		},
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceTaintRules) {
		expectSyncResult.Synced[reflect.TypeFor[*resourceapi.DeviceTaintRule]()] = true
	}
	if diff := cmp.Diff(expectSyncResult, syncResult,
		cmp.Transformer("TypeOf", func(t reflect.Type) string {
			return t.String()
		}),
	); diff != "" {
		tCtx.Fatalf("unexpected informer sync result (- expected, + actual):\n%s", diff)
	}

	// The ResourceSlice tracker lags behind informers when DeviceTaintRules are enabled.
	if !cache.WaitForCacheSync(tCtx.Done(), resourceSliceTracker.HasSynced) {
		tCtx.Fatalf("resource slice tracker failed to sync: %v", context.Cause(tCtx))
	}

	celCache := cel.NewCache(10, cel.Features{
		EnableConsumableCapacity: utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity),
		EnableListTypeAttributes: utilfeature.DefaultFeatureGate.Enabled(features.DRAListTypeAttributes),
	})

	// Also wait for the assume cache to catch up.
	// Without this we cannot reliably store the result of
	// the UpdateStatus call below.
	// Has to be done indirectly, the assume cache itself has
	// no HasSynced method (maybe it should).
	handle := assumeCache.AddEventHandler(cache.ResourceEventHandlerFuncs{})
	if !cache.WaitForCacheSync(tCtx.Done(), handle.HasSynced) {
		tCtx.Fatalf("assume cache failed to sync: %v", context.Cause(tCtx))
	}

	// The set of nodes is assumed to be fixed at this point.
	nodes, err := nodeLister.List(labels.Everything())
	tCtx.ExpectNoError(err, "list nodes")
	slices, err := draManager.ResourceSlices().ListWithDeviceTaintRules()
	tCtx.ExpectNoError(err, "list slices")

	// Allocate one claim at a time, picking nodes randomly. Each
	// allocation is stored immediately, using the claim cache to avoid
	// having to wait for the actual informer update.
claims:
	for i := range claims.Items {
		claim := &claims.Items[i]
		if claim.Status.Allocation != nil {
			continue
		}

		claims, err := draManager.ResourceClaims().List()
		tCtx.ExpectNoError(err, "list claims")
		allocatedDevices := sets.New[structured.DeviceID]()
		allocatedSharedDeviceIDs := sets.New[structured.SharedDeviceID]()
		aggregatedCapacity := structured.NewConsumedCapacityCollection()
		for _, claim := range claims {
			if claim.Status.Allocation == nil {
				continue
			}
			for _, result := range claim.Status.Allocation.Devices.Results {
				deviceID := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
				allocatedDevices.Insert(deviceID)
				if result.ShareID == nil {
					allocatedDevices.Insert(deviceID)
					continue
				}
				sharedDeviceID := structured.MakeSharedDeviceID(deviceID, result.ShareID)
				allocatedSharedDeviceIDs.Insert(sharedDeviceID)
				claimedCapacity := result.ConsumedCapacity
				if claimedCapacity != nil {
					allocatedCapacity := structured.NewDeviceConsumedCapacity(deviceID, claimedCapacity)
					aggregatedCapacity.Insert(allocatedCapacity)
				}
			}
		}
		allocatedState := structured.AllocatedState{
			AllocatedDevices:         allocatedDevices,
			AllocatedSharedDeviceIDs: allocatedSharedDeviceIDs,
			AggregatedCapacity:       aggregatedCapacity,
		}
		allocator, err := structured.NewAllocator(tCtx, structured.Features{
			PrioritizedList:      utilfeature.DefaultFeatureGate.Enabled(features.DRAPrioritizedList),
			AdminAccess:          utilfeature.DefaultFeatureGate.Enabled(features.DRAAdminAccess),
			DeviceTaints:         utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceTaints),
			PartitionableDevices: utilfeature.DefaultFeatureGate.Enabled(features.DRAPartitionableDevices),
			ConsumableCapacity:   utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity),
		}, allocatedState, draManager.DeviceClasses(), slices, celCache)
		tCtx.ExpectNoError(err, "create allocator")

		rand.Shuffle(len(nodes), func(i, j int) {
			nodes[i], nodes[j] = nodes[j], nodes[i]
		})
		for _, node := range nodes {
			result, err := allocator.Allocate(tCtx, node, []*resourceapi.ResourceClaim{claim})
			tCtx.ExpectNoError(err, "allocate claim")
			if result != nil {
				claim = claim.DeepCopy()
				claim.Status.Allocation = &result[0]
				claim, err := tCtx.Client().ResourceV1().ResourceClaims(claim.Namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
				tCtx.ExpectNoError(err, "update claim status with allocation")
				tCtx.ExpectNoError(draManager.ResourceClaims().AssumeClaimAfterAPICall(claim), "assume claim")
				continue claims
			}
		}
		tCtx.Fatalf("Could not allocate claim %d out of %d", i, len(claims))
	}
}
