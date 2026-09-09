package internal

import (
	"math/rand"
	"sort"

	"github.com/onsi/ginkgo/v2/types"
)

type SortableSpecs struct {
	Specs   Specs
	Indexes []int
}

func NewSortableSpecs(specs Specs) *SortableSpecs {
	indexes := make([]int, len(specs))
	for i := range specs {
		indexes[i] = i
	}
	return &SortableSpecs{
		Specs:   specs,
		Indexes: indexes,
	}
}
func (s *SortableSpecs) Len() int      { return len(s.Indexes) }
func (s *SortableSpecs) Swap(i, j int) { s.Indexes[i], s.Indexes[j] = s.Indexes[j], s.Indexes[i] }
func (s *SortableSpecs) Less(i, j int) bool {
	a, b := s.Specs[s.Indexes[i]], s.Specs[s.Indexes[j]]

	aNodes, bNodes := a.Nodes.WithType(types.NodeTypesForContainerAndIt), b.Nodes.WithType(types.NodeTypesForContainerAndIt)

	firstOrderedAIdx, firstOrderedBIdx := aNodes.IndexOfFirstNodeMarkedOrdered(), bNodes.IndexOfFirstNodeMarkedOrdered()
	if firstOrderedAIdx > -1 && firstOrderedBIdx > -1 && aNodes[firstOrderedAIdx].ID == bNodes[firstOrderedBIdx].ID {
		// strictly preserve order within an ordered containers.  ID will track this as IDs are generated monotonically
		return aNodes.FirstNodeWithType(types.NodeTypeIt).ID < bNodes.FirstNodeWithType(types.NodeTypeIt).ID
	}

	// if either spec is in an ordered container - only use the nodes up to the outermost ordered container
	if firstOrderedAIdx > -1 {
		aNodes = aNodes[:firstOrderedAIdx+1]
	}
	if firstOrderedBIdx > -1 {
		bNodes = bNodes[:firstOrderedBIdx+1]
	}

	for i := 0; i < len(aNodes) && i < len(bNodes); i++ {
		aCL, bCL := aNodes[i].CodeLocation, bNodes[i].CodeLocation
		if aCL.FileName != bCL.FileName {
			return aCL.FileName < bCL.FileName
		}
		if aCL.LineNumber != bCL.LineNumber {
			return aCL.LineNumber < bCL.LineNumber
		}
	}
	// either everything is equal or we have different lengths of CLs
	if len(aNodes) != len(bNodes) {
		return len(aNodes) < len(bNodes)
	}
	// ok, now we are sure everything was equal. so we use the spec text to break ties
	for i := 0; i < len(aNodes); i++ {
		if aNodes[i].Text != bNodes[i].Text {
			return aNodes[i].Text < bNodes[i].Text
		}
	}
	// ok, all those texts were equal.  we'll use the ID of the most deeply nested node as a last resort
	return aNodes[len(aNodes)-1].ID < bNodes[len(bNodes)-1].ID
}

type GroupedSpecIndices []SpecIndices
type SpecIndices []int

func OrderSpecs(specs Specs, suiteConfig types.SuiteConfig) (GroupedSpecIndices, GroupedSpecIndices) {
	/*
		Ginkgo has sophisticated support for randomizing specs.  Specs are guaranteed to have the same
		order for a given seed across test runs.

		By default only top-level containers and specs are shuffled - this makes for a more intuitive debugging
		experience - specs within a given container run in the order they appear in the file.

		Developers can set -randomizeAllSpecs to shuffle _all_ specs.

		In addition, spec containers can be marked as Ordered.  Specs within an Ordered container are never shuffled.

		Finally, specs and spec containers can be marked as Serial.  When running in parallel, serial specs run on Process #1 _after_ all other processes have finished.
	*/

	// Seed a new random source based on thee configured random seed.
	r := rand.New(rand.NewSource(suiteConfig.RandomSeed))

	// first, we sort the entire suite to ensure a deterministic order.  the sort is performed by filename, then line number, and then spec text.  this ensures every parallel process has the exact same spec order and is only necessary to cover the edge case where the user iterates over a map to generate specs.
	sortableSpecs := NewSortableSpecs(specs)
	sort.Sort(sortableSpecs)

	// then we break things into execution groups
	// a group represents a single unit of execution and is a collection of SpecIndices
	// usually a group is just a single spec, however ordered containers must be preserved as a single group
	executionGroupIDs := []uint{}
	executionGroups := map[uint]SpecIndices{}
	for _, idx := range sortableSpecs.Indexes {
		spec := specs[idx]
		groupNode := spec.Nodes.FirstNodeMarkedOrdered()
		if groupNode.IsZero() {
			groupNode = spec.Nodes.FirstNodeWithType(types.NodeTypeIt)
		}
		executionGroups[groupNode.ID] = append(executionGroups[groupNode.ID], idx)
		if len(executionGroups[groupNode.ID]) == 1 {
			executionGroupIDs = append(executionGroupIDs, groupNode.ID)
		}
	}

	// now, we only shuffle all the execution groups if we're randomizing all specs, otherwise
	// we shuffle outermost containers.  so we need to form shufflable groupings of GroupIDs
	shufflableGroupingIDs := []uint{}
	shufflableGroupingIDToGroupIDs := map[uint][]uint{}

	// for each execution group we're going to have to pick a node to represent how the
	// execution group is grouped for shuffling:
	nodeTypesToShuffle := types.NodeTypesForContainerAndIt
	if suiteConfig.RandomizeAllSpecs {
		nodeTypesToShuffle = types.NodeTypeIt
	}

	//so, for each execution group:
	for _, groupID := range executionGroupIDs {
		// pick out a representative spec
		representativeSpec := specs[executionGroups[groupID][0]]

		// and grab the node on the spec that will represent which shufflable group this execution group belongs to
		shufflableGroupingNode := representativeSpec.Nodes.FirstNodeWithType(nodeTypesToShuffle)

		//add the execution group to its shufflable group
		shufflableGroupingIDToGroupIDs[shufflableGroupingNode.ID] = append(shufflableGroupingIDToGroupIDs[shufflableGroupingNode.ID], groupID)

		//and if it's the first one in
		if len(shufflableGroupingIDToGroupIDs[shufflableGroupingNode.ID]) == 1 {
			// record the shuffleable group ID
			shufflableGroupingIDs = append(shufflableGroupingIDs, shufflableGroupingNode.ID)
		}
	}

	// now, for each shuffleable group, we compute the priority
	shufflableGroupingIDPriorities := map[uint]int{}
	for shufflableGroupingID, groupIDs := range shufflableGroupingIDToGroupIDs {
		// the priority of a shufflable grouping is the max priority of any spec in any execution group in the shufflable grouping
		maxPriority := -1 << 31 // min int
		for _, groupID := range groupIDs {
			for _, specIdx := range executionGroups[groupID] {
				specPriority := specs[specIdx].Nodes.GetSpecPriority()
				maxPriority = max(specPriority, maxPriority)
			}
		}
		shufflableGroupingIDPriorities[shufflableGroupingID] = maxPriority
	}

	// now we permute the sorted shufflable grouping IDs and build the ordered Groups
	permutation := r.Perm(len(shufflableGroupingIDs))
	shuffledGroupingIds := make([]uint, len(shufflableGroupingIDs))
	for i, j := range permutation {
		shuffledGroupingIds[i] = shufflableGroupingIDs[j]
	}
	// now, we need to stable sort the shuffledGroupingIds by priority (higher priority first)
	sort.SliceStable(shuffledGroupingIds, func(i, j int) bool {
		return shufflableGroupingIDPriorities[shuffledGroupingIds[i]] > shufflableGroupingIDPriorities[shuffledGroupingIds[j]]
	})

	// we can now take these prioritized, shuffled, groupings and form the final set of ordered spec groups
	orderedGroups := GroupedSpecIndices{}
	for _, id := range shuffledGroupingIds {
		for _, executionGroupID := range shufflableGroupingIDToGroupIDs[id] {
			orderedGroups = append(orderedGroups, executionGroups[executionGroupID])
		}
	}

	// If we're running in series, we're done.
	if suiteConfig.ParallelTotal == 1 {
		return orderedGroups, GroupedSpecIndices{}
	}

	// We're running in parallel so we need to partition the ordered groups into a parallelizable set and a serialized set.
	// The parallelizable groups will run across all Ginkgo processes...
	// ...the serial groups will only run on Process #1 after all other processes have exited.
	parallelizableGroups, serialGroups := GroupedSpecIndices{}, GroupedSpecIndices{}
	for _, specIndices := range orderedGroups {
		if specs[specIndices[0]].Nodes.HasNodeMarkedSerial() {
			serialGroups = append(serialGroups, specIndices)
		} else {
			parallelizableGroups = append(parallelizableGroups, specIndices)
		}
	}

	return parallelizableGroups, serialGroups
}
