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
		Ginkgo has sophisticated suport for randomizing specs.  Specs are guaranteed to have the same
		order for a given seed across test runs.

		By default only top-level containers and specs are shuffled - this makes for a more intuitive debugging
		experience - specs within a given container run in the order they appear in the file.

		Developers can set -randomizeAllSpecs to shuffle _all_ specs.

		In addition, spec containers can be marked as Ordered.  Specs within an Ordered container are never shuffled.

		Finally, specs and spec containers can be marked as Serial.  When running in parallel, serial specs run on Process #1 _after_ all other processes have finished.
	*/

	// Seed a new random source based on thee configured random seed.
	r := rand.New(rand.NewSource(suiteConfig.RandomSeed))

	// Decide how to group specs for shuffling.  By default we shuffle top-level containers,
	// but setting --randomize-all-specs causes us to shuffle all specs (excpect for Ordered specs)
	nodeTypesToGroup := types.NodeTypesForContainerAndIt
	if suiteConfig.RandomizeAllSpecs {
		nodeTypesToGroup = types.NodeTypeIt
	}

	// Go through all specs and build the permutable groups.  These are groupings that can be shuffled.
	// Along the way we extract sort keys to ensure a consistent order of specs before we permute them.
	permutableGroups := map[uint]SpecIndices{}
	groupIsMarkedOrdered := map[uint]bool{}
	groupSortKeys := map[uint]string{}
	groupIDs := []uint{}
	for idx, spec := range specs {
		groupingNode := spec.Nodes.FirstNodeMarkedOrdered()
		if groupingNode.IsZero() {
			// If a spec is not in an ordered container...
			// ...we group based on the first node with a nodetype satisfying `nodeTypesToGroup`
			groupingNode = spec.Nodes.FirstNodeWithType(nodeTypesToGroup)
		} else {
			// If a spec is in an ordered container...
			// ...we group based on the outermost ordered container
			groupIsMarkedOrdered[groupingNode.ID] = true
		}
		// we've figured out which group we're in, so we add this specs index to the group.
		permutableGroups[groupingNode.ID] = append(permutableGroups[groupingNode.ID], idx)
		// and, while we're at it, extract the sort key for this group if we haven't already.
		if groupSortKeys[groupingNode.ID] == "" {
			groupSortKeys[groupingNode.ID] = groupingNode.CodeLocation.String()
			groupIDs = append(groupIDs, groupingNode.ID)
		}
	}

	// now sort the groups by the sort key.  We use the grouping node's code location and break ties using group ID
	sort.SliceStable(groupIDs, func(i, j int) bool {
		keyA := groupSortKeys[groupIDs[i]]
		keyB := groupSortKeys[groupIDs[j]]
		if keyA == keyB {
			return groupIDs[i] < groupIDs[j]
		} else {
			return keyA < keyB
		}
	})

	// now permute the sorted group IDs and build the ordered Groups
	orderedGroups := GroupedSpecIndices{}
	permutation := r.Perm(len(groupIDs))
	for _, j := range permutation {
		if groupIsMarkedOrdered[groupIDs[j]] {
			// If the group is marked ordered, we preserve the grouping to ensure ordered specs always run on the same Ginkgo process
			orderedGroups = append(orderedGroups, permutableGroups[groupIDs[j]])
		} else {
			// If the group is _not_ marked ordered, we expand the grouping (it has served its purpose for permutation), in order to allow parallelizing across the specs in the group.
			for _, idx := range permutableGroups[groupIDs[j]] {
				orderedGroups = append(orderedGroups, SpecIndices{idx})
			}
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
