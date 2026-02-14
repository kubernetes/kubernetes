package topologyaware

import (
	"context"
	"fmt"
	"reflect"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"

	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/dynamic-resource-allocation/structured"

	fwk "k8s.io/kube-scheduler/framework"
)

// DRATestPlugin is a test plugin that implements PlacementGeneratorPlugin and PlacementStatePlugin.
// It is implemented only as a PoC, showing that early implementation of TAS can work with DRA based
// topologies.
type DRATestPlugin struct {
	draManager fwk.SharedDRAManager
	handle     fwk.Handle
}

var _ fwk.PlacementGeneratorPlugin = &DRATestPlugin{}
var _ fwk.PlacementStatePlugin = &DRATestPlugin{}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
	return &DRATestPlugin{handle: h, draManager: h.SharedDRAManager()}, nil
}

// Name returns name of the plugin.
func (pl *DRATestPlugin) Name() string {
	return "DRATestPlugin"
}

// GeneratePlacements generates a list of potential Placements for the given PodGroup.
// It finds all resource claims used by pods in a pod group and for each of them
// tries to find possible allocations, returning placements that can possibly
// satisfy all claims.
func (pl *DRATestPlugin) GeneratePlacements(ctx context.Context, state *fwk.CycleState, podGroup *fwk.PodGroupInfo, parentPlacements []*fwk.ParentPlacement) ([]*fwk.Placement, *fwk.Status) {
	// Identify all unique ResourceClaims in the PodGroup.
	claims, status := pl.findUniqueClaims(ctx, podGroup)
	if status != nil {
		return nil, status
	}

	claimsToAllocate := make([]*resourceapi.ResourceClaim, 0, len(claims))
	for _, claim := range claims {
		claimsToAllocate = append(claimsToAllocate, claim)
	}

	// Prepare allocator.
	allocatedDevices, err := pl.draManager.ResourceClaims().ListAllAllocatedDevices()
	if err != nil {
		return []*fwk.Placement{}, fwk.AsStatus(err)
	}
	allocatedState := structured.AllocatedState{
		AllocatedDevices:         allocatedDevices,
		AllocatedSharedDeviceIDs: sets.New[structured.SharedDeviceID](),
		AggregatedCapacity:       structured.NewConsumedCapacityCollection(),
	}
	slices, err := pl.draManager.ResourceSlices().ListWithDeviceTaintRules()
	if err != nil {
		return nil, fwk.AsStatus(err)
	}
	allocator, err := structured.NewAllocator(ctx, structured.Features{}, allocatedState, pl.draManager.DeviceClasses(), slices, cel.NewCache(1000, cel.Features{}))
	if err != nil {
		return nil, fwk.AsStatus(err)
	}

	placements := make([]*fwk.Placement, 0)
	for _, parent := range parentPlacements {
		// Find all possible allocations for each claim, limiting the search to nodes in the parent placement.
		claimToAllocations, ok := pl.findPossibleAllocations(ctx, allocator, parent.PlacementNodes, claimsToAllocate)
		if !ok {
			continue
		}

		// We need to form a combination containing one Option per Claim.
		// Then we intersect their valid nodes.
		// If intersection is not empty, we create a Placement.

		// Convert map to slice of slices for easier recursion
		var claimOrder []string
		var allocationsList [][]resourceapi.AllocationResult
		for _, claim := range claimsToAllocate {
			uid := string(claim.UID)
			allocations := claimToAllocations[uid]
			claimOrder = append(claimOrder, claim.Name)
			allocationsList = append(allocationsList, allocations)
		}

		// Closure for recursion
		var combineAllocations func(index int, currentAllocations []resourceapi.AllocationResult)
		combineAllocations = func(index int, currentAllocations []resourceapi.AllocationResult) {
			if index == len(allocationsList) {
				var finalSelector *v1.NodeSelector = parent.NodeSelector
				for _, opt := range currentAllocations {
					finalSelector = intersectNodeSelectors(finalSelector, opt.NodeSelector)
				}

				draAllocations := make([]fwk.DraClaimAllocation, len(claimsToAllocate))
				for i, opt := range currentAllocations {
					draAllocations[i] = fwk.DraClaimAllocation{
						ResourceClaimName: claimOrder[i],
						Allocation:        opt,
					}
				}

				placements = append(placements, &fwk.Placement{
					NodeSelector:   finalSelector,
					DRAAllocations: draAllocations,
				})
				return
			}

			for _, opt := range allocationsList[index] {
				combineAllocations(index+1, append(currentAllocations, opt))
			}
		}

		combineAllocations(0, nil)
	}

	return placements, nil
}

// findUniqueClaims returns a map of unique ResourceClaims in the PodGroup.
func (pl *DRATestPlugin) findUniqueClaims(ctx context.Context, podGroup *fwk.PodGroupInfo) (map[string]*resourceapi.ResourceClaim, *fwk.Status) {
	logger := klog.FromContext(ctx)
	claims := make(map[string]*resourceapi.ResourceClaim)
	for _, pod := range podGroup.UnscheduledPods {
		for _, podClaim := range pod.Spec.ResourceClaims {
			claimName, mustCheckOwner, err := resourceclaim.Name(pod, &podClaim)
			if err != nil {
				return nil, fwk.AsStatus(fmt.Errorf("failed to get claim name: %v", err))
			}
			// The claim name might be nil if no underlying resource claim
			// was generated for the referenced claim. There are valid use
			// cases when this might happen, so we simply skip it.
			if claimName == nil {
				continue
			}
			claim, err := pl.draManager.ResourceClaims().Get(pod.Namespace, *claimName)
			if err != nil {
				return nil, fwk.AsStatus(fmt.Errorf("failed to get claim: %v", err))
			}

			if claim.DeletionTimestamp != nil {
				logger.V(3).Info("resourceclaim is being deleted", "claim", claim.Name)
				continue
			}

			if mustCheckOwner {
				if err := resourceclaim.IsForPod(pod, claim); err != nil {
					return nil, fwk.AsStatus(fmt.Errorf("failed to check claim owner: %v", err))
				}
			}
			claims[string(pod.Namespace)+"/"+podClaim.Name] = claim
		}
	}
	return claims, nil
}

// findPossibleAllocations finds all possible allocations for each claim on the given nodes.
// It returns a map from claim UID to a list of possible allocation results.
// If any claim cannot be satisfied on any of the nodes, it returns false.
func (pl *DRATestPlugin) findPossibleAllocations(ctx context.Context, allocator structured.Allocator, nodes []*v1.Node, claims []*resourceapi.ResourceClaim) (map[string][]resourceapi.AllocationResult, bool) {
	logger := klog.FromContext(ctx)
	claimToAllocations := make(map[string][]resourceapi.AllocationResult)
	for _, claim := range claims {
		var possibleAllocations []resourceapi.AllocationResult
		for _, node := range nodes {
			if node == nil {
				continue
			}

			// allocator.Allocate() returns first possible allocation that satisfies the claim
			// on a given node.
			// For PoC this should be good enough.
			results, err := allocator.Allocate(ctx, node, []*resourceapi.ResourceClaim{claim})
			if err != nil {
				logger.Error(err, "Error checking allocation", "node", node.Name, "claim", claim.Name)
				continue
			}

			if len(results) == 0 {
				continue
			}

			found := false
			for _, opt := range possibleAllocations {
				if isSameAllocationResult(opt, results[0]) {
					found = true
					break
				}
			}
			if !found {
				possibleAllocations = append(possibleAllocations, results[0])
			}
		}
		// If we did not find any possible allocation for given claim on any node from the parent placement
		// we cannot satisfy the whole group.
		if len(possibleAllocations) == 0 {
			return nil, false
		}

		claimToAllocations[string(claim.UID)] = possibleAllocations
	}
	return claimToAllocations, true
}

// AssumePlacement temporarily configures the scheduling context to evaluate the feasibility of the given Placement.
func (pl *DRATestPlugin) AssumePlacement(ctx context.Context, state *fwk.CycleState, podGroup *fwk.PodGroupInfo, placement *fwk.Placement) *fwk.Status {
	// Find unique claim names within the PodGroup context.
	claims := make(map[string]*resourceapi.ResourceClaim)
	for _, pod := range podGroup.UnscheduledPods {
		for _, podClaim := range pod.Spec.ResourceClaims {
			claimName, _, err := resourceclaim.Name(pod, &podClaim)
			if err != nil || claimName == nil {
				continue
			}
			claim, err := pl.draManager.ResourceClaims().Get(pod.Namespace, *claimName)
			if err != nil {
				continue
			}
			claims[claim.Name] = claim
		}
	}

	for _, alloc := range placement.DRAAllocations {
		claim, ok := claims[alloc.ResourceClaimName]
		if !ok {
			// This shouldn't happen if Placement matches PodGroup
			continue
		}

		// Create a copy with the allocation
		assumedClaim := claim.DeepCopy()
		assumedClaim.Status.Allocation = &alloc.Allocation

		if err := pl.draManager.ResourceClaims().SignalClaimPendingAllocation(claim.UID, assumedClaim); err != nil {
			return fwk.AsStatus(err)
		}
	}
	return nil
}

// RevertPlacement reverts the temporary scheduling context changes made by AssumePlacement.
func (pl *DRATestPlugin) RevertPlacement(ctx context.Context, state *fwk.CycleState, podGroup *fwk.PodGroupInfo, placement *fwk.Placement) *fwk.Status {
	claims := make(map[string]*resourceapi.ResourceClaim)
	for _, pod := range podGroup.UnscheduledPods {
		for _, podClaim := range pod.Spec.ResourceClaims {
			claimName, _, err := resourceclaim.Name(pod, &podClaim)
			if err != nil || claimName == nil {
				continue
			}
			claim, err := pl.draManager.ResourceClaims().Get(pod.Namespace, *claimName)
			if err != nil {
				continue
			}
			claims[claim.Name] = claim
		}
	}

	for _, alloc := range placement.DRAAllocations {
		claim, ok := claims[alloc.ResourceClaimName]
		if !ok {
			continue
		}
		pl.draManager.ResourceClaims().RemoveClaimPendingAllocation(claim.UID)
	}
	return nil
}

func isSameAllocationResult(a, b resourceapi.AllocationResult) bool {
	return reflect.DeepEqual(a, b)
}

func intersectNodeSelectors(a, b *v1.NodeSelector) *v1.NodeSelector {
	if a == nil || len(a.NodeSelectorTerms) == 0 {
		return b
	}
	if b == nil || len(b.NodeSelectorTerms) == 0 {
		return a
	}

	// NodeSelectorTerms are ORed
	// MatchFields/MatchExpressions are ANDed
	// So if A selector has X OR Y and B selector has O OR P
	// We need (X AND B) OR (X AND P) OR (Y AND O) OR (Y AND P)
	var newTerms []v1.NodeSelectorTerm
	for _, t1 := range a.NodeSelectorTerms {
		for _, t2 := range b.NodeSelectorTerms {
			// Merge t1 and t2
			// t1 AND t2
			// Concatenate MatchExpressions and MatchFields
			newTerm := v1.NodeSelectorTerm{
				MatchExpressions: append(append([]v1.NodeSelectorRequirement{}, t1.MatchExpressions...), t2.MatchExpressions...),
				MatchFields:      append(append([]v1.NodeSelectorRequirement{}, t1.MatchFields...), t2.MatchFields...),
			}
			newTerms = append(newTerms, newTerm)
		}
	}

	return &v1.NodeSelector{NodeSelectorTerms: newTerms}
}
