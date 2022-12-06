package internal

import (
	"math/rand"
	"sort"

	"github.com/onsi/ginkgo/v2/types"
)

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

	// first break things into execution groups
	// a group represents a single unit of execution and is a collection of SpecIndices
	// usually a group is just a single spec, however ordered containers must be preserved as a single group
	executionGroupIDs := []uint{}
	executionGroups := map[uint]SpecIndices{}
	for idx, spec := range specs {
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
	shufflableGroupingsIDToSortKeys := map[uint]string{}

	// for each execution group we're going to have to pick a node to represent how the
	// execution group is grouped for shuffling:
	nodeTypesToShuffle := types.NodeTypesForContainerAndIt
	if suiteConfig.RandomizeAllSpecs {
		nodeTypesToShuffle = types.NodeTypeIt
	}

	//so, fo reach execution group:
	for _, groupID := range executionGroupIDs {
		// pick out a representative spec
		representativeSpec := specs[executionGroups[groupID][0]]

		// and grab the node on the spec that will represent which shufflable group this execution group belongs tu
		shufflableGroupingNode := representativeSpec.Nodes.FirstNodeWithType(nodeTypesToShuffle)

		//add the execution group to its shufflable group
		shufflableGroupingIDToGroupIDs[shufflableGroupingNode.ID] = append(shufflableGroupingIDToGroupIDs[shufflableGroupingNode.ID], groupID)

		//and if it's the first one in
		if len(shufflableGroupingIDToGroupIDs[shufflableGroupingNode.ID]) == 1 {
			// record the shuffleable group ID
			shufflableGroupingIDs = append(shufflableGroupingIDs, shufflableGroupingNode.ID)
			// and record the sort key to use
			shufflableGroupingsIDToSortKeys[shufflableGroupingNode.ID] = shufflableGroupingNode.CodeLocation.String()
		}
	}

	// now we sort the shufflable groups by the sort key.  We use the shufflable group nodes code location and break ties using its node id
	sort.SliceStable(shufflableGroupingIDs, func(i, j int) bool {
		keyA := shufflableGroupingsIDToSortKeys[shufflableGroupingIDs[i]]
		keyB := shufflableGroupingsIDToSortKeys[shufflableGroupingIDs[j]]
		if keyA == keyB {
			return shufflableGroupingIDs[i] < shufflableGroupingIDs[j]
		} else {
			return keyA < keyB
		}
	})

	// now we permute the sorted shufflable grouping IDs and build the ordered Groups
	orderedGroups := GroupedSpecIndices{}
	permutation := r.Perm(len(shufflableGroupingIDs))
	for _, j := range permutation {
		//let's get the execution group IDs for this shufflable group:
		executionGroupIDsForJ := shufflableGroupingIDToGroupIDs[shufflableGroupingIDs[j]]
		// and we'll add their associated specindices to the orderedGroups slice:
		for _, executionGroupID := range executionGroupIDsForJ {
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
