package internal

import (
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2/types"
)

type runOncePair struct {
	//nodeId should only run once...
	nodeID   uint
	nodeType types.NodeType
	//...for specs in a hierarchy that includes this context
	containerID uint
}

func (pair runOncePair) isZero() bool {
	return pair.nodeID == 0
}

func runOncePairForNode(node Node, containerID uint) runOncePair {
	return runOncePair{
		nodeID:      node.ID,
		nodeType:    node.NodeType,
		containerID: containerID,
	}
}

type runOncePairs []runOncePair

func runOncePairsForSpec(spec Spec) runOncePairs {
	pairs := runOncePairs{}

	containers := spec.Nodes.WithType(types.NodeTypeContainer)
	for _, node := range spec.Nodes {
		if node.NodeType.Is(types.NodeTypeBeforeAll | types.NodeTypeAfterAll) {
			pairs = append(pairs, runOncePairForNode(node, containers.FirstWithNestingLevel(node.NestingLevel-1).ID))
		} else if node.NodeType.Is(types.NodeTypeBeforeEach|types.NodeTypeJustBeforeEach|types.NodeTypeAfterEach|types.NodeTypeJustAfterEach) && node.MarkedOncePerOrdered {
			passedIntoAnOrderedContainer := false
			firstOrderedContainerDeeperThanNode := containers.FirstSatisfying(func(container Node) bool {
				passedIntoAnOrderedContainer = passedIntoAnOrderedContainer || container.MarkedOrdered
				return container.NestingLevel >= node.NestingLevel && passedIntoAnOrderedContainer
			})
			if firstOrderedContainerDeeperThanNode.IsZero() {
				continue
			}
			pairs = append(pairs, runOncePairForNode(node, firstOrderedContainerDeeperThanNode.ID))
		}
	}

	return pairs
}

func (pairs runOncePairs) runOncePairFor(nodeID uint) runOncePair {
	for i := range pairs {
		if pairs[i].nodeID == nodeID {
			return pairs[i]
		}
	}
	return runOncePair{}
}

func (pairs runOncePairs) hasRunOncePair(pair runOncePair) bool {
	for i := range pairs {
		if pairs[i] == pair {
			return true
		}
	}
	return false
}

func (pairs runOncePairs) withType(nodeTypes types.NodeType) runOncePairs {
	count := 0
	for i := range pairs {
		if pairs[i].nodeType.Is(nodeTypes) {
			count++
		}
	}

	out, j := make(runOncePairs, count), 0
	for i := range pairs {
		if pairs[i].nodeType.Is(nodeTypes) {
			out[j] = pairs[i]
			j++
		}
	}
	return out
}

type group struct {
	suite          *Suite
	specs          Specs
	runOncePairs   map[uint]runOncePairs
	runOnceTracker map[runOncePair]types.SpecState

	succeeded bool
}

func newGroup(suite *Suite) *group {
	return &group{
		suite:          suite,
		runOncePairs:   map[uint]runOncePairs{},
		runOnceTracker: map[runOncePair]types.SpecState{},
		succeeded:      true,
	}
}

func (g *group) initialReportForSpec(spec Spec) types.SpecReport {
	return types.SpecReport{
		ContainerHierarchyTexts:     spec.Nodes.WithType(types.NodeTypeContainer).Texts(),
		ContainerHierarchyLocations: spec.Nodes.WithType(types.NodeTypeContainer).CodeLocations(),
		ContainerHierarchyLabels:    spec.Nodes.WithType(types.NodeTypeContainer).Labels(),
		LeafNodeLocation:            spec.FirstNodeWithType(types.NodeTypeIt).CodeLocation,
		LeafNodeType:                types.NodeTypeIt,
		LeafNodeText:                spec.FirstNodeWithType(types.NodeTypeIt).Text,
		LeafNodeLabels:              []string(spec.FirstNodeWithType(types.NodeTypeIt).Labels),
		ParallelProcess:             g.suite.config.ParallelProcess,
		IsSerial:                    spec.Nodes.HasNodeMarkedSerial(),
		IsInOrderedContainer:        !spec.Nodes.FirstNodeMarkedOrdered().IsZero(),
	}
}

func (g *group) evaluateSkipStatus(spec Spec) (types.SpecState, types.Failure) {
	if spec.Nodes.HasNodeMarkedPending() {
		return types.SpecStatePending, types.Failure{}
	}
	if spec.Skip {
		return types.SpecStateSkipped, types.Failure{}
	}
	if g.suite.interruptHandler.Status().Interrupted || g.suite.skipAll {
		return types.SpecStateSkipped, types.Failure{}
	}
	if !g.succeeded {
		return types.SpecStateSkipped, g.suite.failureForLeafNodeWithMessage(spec.FirstNodeWithType(types.NodeTypeIt),
			"Spec skipped because an earlier spec in an ordered container failed")
	}
	beforeOncePairs := g.runOncePairs[spec.SubjectID()].withType(types.NodeTypeBeforeAll | types.NodeTypeBeforeEach | types.NodeTypeJustBeforeEach)
	for _, pair := range beforeOncePairs {
		if g.runOnceTracker[pair].Is(types.SpecStateSkipped) {
			return types.SpecStateSkipped, g.suite.failureForLeafNodeWithMessage(spec.FirstNodeWithType(types.NodeTypeIt),
				fmt.Sprintf("Spec skipped because Skip() was called in %s", pair.nodeType))
		}
	}
	if g.suite.config.DryRun {
		return types.SpecStatePassed, types.Failure{}
	}
	return g.suite.currentSpecReport.State, g.suite.currentSpecReport.Failure
}

func (g *group) isLastSpecWithPair(specID uint, pair runOncePair) bool {
	lastSpecID := uint(0)
	for idx := range g.specs {
		if g.specs[idx].Skip {
			continue
		}
		sID := g.specs[idx].SubjectID()
		if g.runOncePairs[sID].hasRunOncePair(pair) {
			lastSpecID = sID
		}
	}
	return lastSpecID == specID
}

func (g *group) attemptSpec(isFinalAttempt bool, spec Spec) {
	interruptStatus := g.suite.interruptHandler.Status()

	pairs := g.runOncePairs[spec.SubjectID()]

	nodes := spec.Nodes.WithType(types.NodeTypeBeforeAll)
	nodes = append(nodes, spec.Nodes.WithType(types.NodeTypeBeforeEach)...).SortedByAscendingNestingLevel()
	nodes = append(nodes, spec.Nodes.WithType(types.NodeTypeJustBeforeEach).SortedByAscendingNestingLevel()...)
	nodes = append(nodes, spec.Nodes.FirstNodeWithType(types.NodeTypeIt))
	terminatingNode, terminatingPair := Node{}, runOncePair{}

	for _, node := range nodes {
		oncePair := pairs.runOncePairFor(node.ID)
		if !oncePair.isZero() && g.runOnceTracker[oncePair].Is(types.SpecStatePassed) {
			continue
		}
		g.suite.currentSpecReport.State, g.suite.currentSpecReport.Failure = g.suite.runNode(node, interruptStatus.Channel, spec.Nodes.BestTextFor(node))
		g.suite.currentSpecReport.RunTime = time.Since(g.suite.currentSpecReport.StartTime)
		if !oncePair.isZero() {
			g.runOnceTracker[oncePair] = g.suite.currentSpecReport.State
		}
		if g.suite.currentSpecReport.State != types.SpecStatePassed {
			terminatingNode, terminatingPair = node, oncePair
			break
		}
	}

	afterNodeWasRun := map[uint]bool{}
	includeDeferCleanups := false
	for {
		nodes := spec.Nodes.WithType(types.NodeTypeAfterEach)
		nodes = append(nodes, spec.Nodes.WithType(types.NodeTypeAfterAll)...).SortedByDescendingNestingLevel()
		nodes = append(spec.Nodes.WithType(types.NodeTypeJustAfterEach).SortedByDescendingNestingLevel(), nodes...)
		if !terminatingNode.IsZero() {
			nodes = nodes.WithinNestingLevel(terminatingNode.NestingLevel)
		}
		if includeDeferCleanups {
			nodes = append(nodes, g.suite.cleanupNodes.WithType(types.NodeTypeCleanupAfterEach).Reverse()...)
			nodes = append(nodes, g.suite.cleanupNodes.WithType(types.NodeTypeCleanupAfterAll).Reverse()...)
		}
		nodes = nodes.Filter(func(node Node) bool {
			if afterNodeWasRun[node.ID] {
				//this node has already been run on this attempt, don't rerun it
				return false
			}
			pair := runOncePair{}
			switch node.NodeType {
			case types.NodeTypeCleanupAfterEach, types.NodeTypeCleanupAfterAll:
				// check if we were generated in an AfterNode that has already run
				if afterNodeWasRun[node.NodeIDWhereCleanupWasGenerated] {
					return true // we were, so we should definitely run this cleanup now
				}
				// looks like this cleanup nodes was generated by a before node or it.
				// the run-once status of a cleanup node is governed by the run-once status of its generator
				pair = pairs.runOncePairFor(node.NodeIDWhereCleanupWasGenerated)
			default:
				pair = pairs.runOncePairFor(node.ID)
			}
			if pair.isZero() {
				// this node is not governed by any run-once policy, we should run it
				return true
			}
			// it's our last chance to run if we're the last spec for our oncePair
			isLastSpecWithPair := g.isLastSpecWithPair(spec.SubjectID(), pair)

			switch g.suite.currentSpecReport.State {
			case types.SpecStatePassed: //this attempt is passing...
				return isLastSpecWithPair //...we should run-once if we'this is our last chance
			case types.SpecStateSkipped: //the spec was skipped by the user...
				if isLastSpecWithPair {
					return true //...we're the last spec, so we should run the AfterNode
				}
				if !terminatingPair.isZero() && terminatingNode.NestingLevel == node.NestingLevel {
					return true //...or, a run-once node at our nesting level was skipped which means this is our last chance to run
				}
			case types.SpecStateFailed, types.SpecStatePanicked: // the spec has failed...
				if isFinalAttempt {
					return true //...if this was the last attempt then we're the last spec to run and so the AfterNode should run
				}
				if !terminatingPair.isZero() { // ...and it failed in a run-once.  which will be running again
					if node.NodeType.Is(types.NodeTypeCleanupAfterEach | types.NodeTypeCleanupAfterAll) {
						return terminatingNode.ID == node.NodeIDWhereCleanupWasGenerated // we should run this node if we're a clean-up generated by it
					} else {
						return terminatingNode.NestingLevel == node.NestingLevel // ...or if we're at the same nesting level
					}
				}
			case types.SpecStateInterrupted, types.SpecStateAborted: // ...we've been interrupted and/or aborted
				return true //...that means the test run is over and we should clean up the stack.  Run the AfterNode
			}
			return false
		})

		if len(nodes) == 0 && includeDeferCleanups {
			break
		}

		for _, node := range nodes {
			afterNodeWasRun[node.ID] = true
			state, failure := g.suite.runNode(node, g.suite.interruptHandler.Status().Channel, spec.Nodes.BestTextFor(node))
			g.suite.currentSpecReport.RunTime = time.Since(g.suite.currentSpecReport.StartTime)
			if g.suite.currentSpecReport.State == types.SpecStatePassed || state == types.SpecStateAborted {
				g.suite.currentSpecReport.State = state
				g.suite.currentSpecReport.Failure = failure
			}
		}
		includeDeferCleanups = true
	}

}

func (g *group) run(specs Specs) {
	g.specs = specs
	for _, spec := range g.specs {
		g.runOncePairs[spec.SubjectID()] = runOncePairsForSpec(spec)
	}

	for _, spec := range g.specs {
		g.suite.currentSpecReport = g.initialReportForSpec(spec)
		g.suite.currentSpecReport.State, g.suite.currentSpecReport.Failure = g.evaluateSkipStatus(spec)
		g.suite.reporter.WillRun(g.suite.currentSpecReport)
		g.suite.reportEach(spec, types.NodeTypeReportBeforeEach)

		skip := g.suite.config.DryRun || g.suite.currentSpecReport.State.Is(types.SpecStateFailureStates|types.SpecStateSkipped|types.SpecStatePending)

		g.suite.currentSpecReport.StartTime = time.Now()
		if !skip {
			maxAttempts := max(1, spec.FlakeAttempts())
			if g.suite.config.FlakeAttempts > 0 {
				maxAttempts = g.suite.config.FlakeAttempts
			}
			for attempt := 0; attempt < maxAttempts; attempt++ {
				g.suite.currentSpecReport.NumAttempts = attempt + 1
				g.suite.writer.Truncate()
				g.suite.outputInterceptor.StartInterceptingOutput()
				if attempt > 0 {
					fmt.Fprintf(g.suite.writer, "\nGinkgo: Attempt #%d Failed.  Retrying...\n", attempt)
				}

				g.attemptSpec(attempt == maxAttempts-1, spec)

				g.suite.currentSpecReport.EndTime = time.Now()
				g.suite.currentSpecReport.RunTime = g.suite.currentSpecReport.EndTime.Sub(g.suite.currentSpecReport.StartTime)
				g.suite.currentSpecReport.CapturedGinkgoWriterOutput += string(g.suite.writer.Bytes())
				g.suite.currentSpecReport.CapturedStdOutErr += g.suite.outputInterceptor.StopInterceptingAndReturnOutput()

				if g.suite.currentSpecReport.State.Is(types.SpecStatePassed | types.SpecStateSkipped | types.SpecStateAborted | types.SpecStateInterrupted) {
					break
				}
			}
		}

		g.suite.reportEach(spec, types.NodeTypeReportAfterEach)
		g.suite.processCurrentSpecReport()
		if g.suite.currentSpecReport.State.Is(types.SpecStateFailureStates) {
			g.succeeded = false
		}
		g.suite.currentSpecReport = types.SpecReport{}
	}
}

func (g *group) oldRun(specs Specs) {
	var suite = g.suite
	nodeState := map[uint]types.SpecState{}
	groupSucceeded := true

	indexOfLastSpecContainingNodeID := func(id uint) int {
		lastIdx := -1
		for idx := range specs {
			if specs[idx].Nodes.ContainsNodeID(id) && !specs[idx].Skip {
				lastIdx = idx
			}
		}
		return lastIdx
	}

	for i, spec := range specs {
		suite.currentSpecReport = types.SpecReport{
			ContainerHierarchyTexts:     spec.Nodes.WithType(types.NodeTypeContainer).Texts(),
			ContainerHierarchyLocations: spec.Nodes.WithType(types.NodeTypeContainer).CodeLocations(),
			ContainerHierarchyLabels:    spec.Nodes.WithType(types.NodeTypeContainer).Labels(),
			LeafNodeLocation:            spec.FirstNodeWithType(types.NodeTypeIt).CodeLocation,
			LeafNodeType:                types.NodeTypeIt,
			LeafNodeText:                spec.FirstNodeWithType(types.NodeTypeIt).Text,
			LeafNodeLabels:              []string(spec.FirstNodeWithType(types.NodeTypeIt).Labels),
			ParallelProcess:             suite.config.ParallelProcess,
			IsSerial:                    spec.Nodes.HasNodeMarkedSerial(),
			IsInOrderedContainer:        !spec.Nodes.FirstNodeMarkedOrdered().IsZero(),
		}

		skip := spec.Skip
		if spec.Nodes.HasNodeMarkedPending() {
			skip = true
			suite.currentSpecReport.State = types.SpecStatePending
		} else {
			if suite.interruptHandler.Status().Interrupted || suite.skipAll {
				skip = true
			}
			if !groupSucceeded {
				skip = true
				suite.currentSpecReport.Failure = suite.failureForLeafNodeWithMessage(spec.FirstNodeWithType(types.NodeTypeIt),
					"Spec skipped because an earlier spec in an ordered container failed")
			}
			for _, node := range spec.Nodes.WithType(types.NodeTypeBeforeAll) {
				if nodeState[node.ID] == types.SpecStateSkipped {
					skip = true
					suite.currentSpecReport.Failure = suite.failureForLeafNodeWithMessage(spec.FirstNodeWithType(types.NodeTypeIt),
						"Spec skipped because Skip() was called in BeforeAll")
					break
				}
			}
			if skip {
				suite.currentSpecReport.State = types.SpecStateSkipped
			}
		}

		if suite.config.DryRun && !skip {
			skip = true
			suite.currentSpecReport.State = types.SpecStatePassed
		}

		suite.reporter.WillRun(suite.currentSpecReport)
		//send the spec report to any attached ReportBeforeEach blocks - this will update suite.currentSpecReport if failures occur in these blocks
		suite.reportEach(spec, types.NodeTypeReportBeforeEach)
		if suite.currentSpecReport.State.Is(types.SpecStateFailureStates) {
			//the reportEach failed, skip this spec
			skip = true
		}

		suite.currentSpecReport.StartTime = time.Now()
		maxAttempts := max(1, spec.FlakeAttempts())
		if suite.config.FlakeAttempts > 0 {
			maxAttempts = suite.config.FlakeAttempts
		}

		for attempt := 0; !skip && (attempt < maxAttempts); attempt++ {
			suite.currentSpecReport.NumAttempts = attempt + 1
			suite.writer.Truncate()
			suite.outputInterceptor.StartInterceptingOutput()
			if attempt > 0 {
				fmt.Fprintf(suite.writer, "\nGinkgo: Attempt #%d Failed.  Retrying...\n", attempt)
			}
			isFinalAttempt := (attempt == maxAttempts-1)

			interruptStatus := suite.interruptHandler.Status()
			deepestNestingLevelAttained := -1
			var nodes = spec.Nodes.WithType(types.NodeTypeBeforeAll).Filter(func(n Node) bool {
				return nodeState[n.ID] != types.SpecStatePassed
			})
			nodes = nodes.CopyAppend(spec.Nodes.WithType(types.NodeTypeBeforeEach)...).SortedByAscendingNestingLevel()
			nodes = nodes.CopyAppend(spec.Nodes.WithType(types.NodeTypeJustBeforeEach).SortedByAscendingNestingLevel()...)
			nodes = nodes.CopyAppend(spec.Nodes.WithType(types.NodeTypeIt)...)

			var terminatingNode Node
			for j := range nodes {
				deepestNestingLevelAttained = max(deepestNestingLevelAttained, nodes[j].NestingLevel)
				suite.currentSpecReport.State, suite.currentSpecReport.Failure = suite.runNode(nodes[j], interruptStatus.Channel, spec.Nodes.BestTextFor(nodes[j]))
				suite.currentSpecReport.RunTime = time.Since(suite.currentSpecReport.StartTime)
				nodeState[nodes[j].ID] = suite.currentSpecReport.State
				if suite.currentSpecReport.State != types.SpecStatePassed {
					terminatingNode = nodes[j]
					break
				}
			}

			afterAllNodesThatRan := map[uint]bool{}
			// pull out some shared code so we aren't repeating ourselves down below. this just runs after and cleanup nodes
			runAfterAndCleanupNodes := func(nodes Nodes) {
				for j := range nodes {
					state, failure := suite.runNode(nodes[j], suite.interruptHandler.Status().Channel, spec.Nodes.BestTextFor(nodes[j]))
					suite.currentSpecReport.RunTime = time.Since(suite.currentSpecReport.StartTime)
					nodeState[nodes[j].ID] = state
					if suite.currentSpecReport.State == types.SpecStatePassed || state == types.SpecStateAborted {
						suite.currentSpecReport.State = state
						suite.currentSpecReport.Failure = failure
						if state != types.SpecStatePassed {
							terminatingNode = nodes[j]
						}
					}
					if nodes[j].NodeType.Is(types.NodeTypeAfterAll) {
						afterAllNodesThatRan[nodes[j].ID] = true
					}
				}
			}

			// pull out a helper that captures the logic of whether or not we should run a given After node.
			// there is complexity here stemming from the fact that we allow nested ordered contexts and flakey retries
			shouldRunAfterNode := func(n Node) bool {
				if n.NodeType.Is(types.NodeTypeAfterEach | types.NodeTypeJustAfterEach) {
					return true
				}
				var id uint
				if n.NodeType.Is(types.NodeTypeAfterAll) {
					id = n.ID
					if afterAllNodesThatRan[id] { //we've already run on this attempt. don't run again.
						return false
					}
				}
				if n.NodeType.Is(types.NodeTypeCleanupAfterAll) {
					id = n.NodeIDWhereCleanupWasGenerated
				}
				isLastSpecWithNode := indexOfLastSpecContainingNodeID(id) == i

				switch suite.currentSpecReport.State {
				case types.SpecStatePassed: //we've passed so far...
					return isLastSpecWithNode //... and we're the last spec with this AfterNode, so we should run it
				case types.SpecStateSkipped: //the spec was skipped by the user...
					if isLastSpecWithNode {
						return true //...we're the last spec, so we should run the AfterNode
					}
					if terminatingNode.NodeType.Is(types.NodeTypeBeforeAll) && terminatingNode.NestingLevel == n.NestingLevel {
						return true //...or, a BeforeAll was skipped and it's at our nesting level, so our subgroup is going to skip
					}
				case types.SpecStateFailed, types.SpecStatePanicked: // the spec has failed...
					if isFinalAttempt {
						return true //...if this was the last attempt then we're the last spec to run and so the AfterNode should run
					}
					if terminatingNode.NodeType.Is(types.NodeTypeBeforeAll) {
						//...we'll be rerunning a BeforeAll so we should cleanup after it if...
						if n.NodeType.Is(types.NodeTypeAfterAll) && terminatingNode.NestingLevel == n.NestingLevel {
							return true //we're at the same nesting level
						}
						if n.NodeType.Is(types.NodeTypeCleanupAfterAll) && terminatingNode.ID == n.NodeIDWhereCleanupWasGenerated {
							return true //we're a DeferCleanup generated by it
						}
					}
					if terminatingNode.NodeType.Is(types.NodeTypeAfterAll) {
						//...we'll be rerunning an AfterAll so we should cleanup after it if...
						if n.NodeType.Is(types.NodeTypeCleanupAfterAll) && terminatingNode.ID == n.NodeIDWhereCleanupWasGenerated {
							return true //we're a DeferCleanup generated by it
						}
					}
				case types.SpecStateInterrupted, types.SpecStateAborted: // ...we've been interrupted and/or aborted
					return true //...that means the test run is over and we should clean up the stack.  Run the AfterNode
				}
				return false
			}

			// first pass - run all the JustAfterEach, Aftereach, and AfterAlls.  Our shoudlRunAfterNode filter function will clean up the AfterAlls for us.
			afterNodes := spec.Nodes.WithType(types.NodeTypeJustAfterEach).SortedByDescendingNestingLevel()
			afterNodes = afterNodes.CopyAppend(spec.Nodes.WithType(types.NodeTypeAfterEach).CopyAppend(spec.Nodes.WithType(types.NodeTypeAfterAll)...).SortedByDescendingNestingLevel()...)
			afterNodes = afterNodes.WithinNestingLevel(deepestNestingLevelAttained)
			afterNodes = afterNodes.Filter(shouldRunAfterNode)
			runAfterAndCleanupNodes(afterNodes)

			// second-pass perhaps we didn't run the AfterAlls but a state change due to an AfterEach now requires us to run the AfterAlls:
			afterNodes = spec.Nodes.WithType(types.NodeTypeAfterAll).WithinNestingLevel(deepestNestingLevelAttained).Filter(shouldRunAfterNode)
			runAfterAndCleanupNodes(afterNodes)

			// now we run any DeferCleanups
			afterNodes = suite.cleanupNodes.WithType(types.NodeTypeCleanupAfterEach).Reverse()
			afterNodes = append(afterNodes, suite.cleanupNodes.WithType(types.NodeTypeCleanupAfterAll).Filter(shouldRunAfterNode).Reverse()...)
			runAfterAndCleanupNodes(afterNodes)

			// third-pass, perhaps a DeferCleanup failed and now we need to run the AfterAlls.
			afterNodes = spec.Nodes.WithType(types.NodeTypeAfterAll).WithinNestingLevel(deepestNestingLevelAttained).Filter(shouldRunAfterNode)
			runAfterAndCleanupNodes(afterNodes)

			// and finally - running AfterAlls may have generated some new DeferCleanup nodes, let's run them to finish up
			afterNodes = suite.cleanupNodes.WithType(types.NodeTypeCleanupAfterAll).Reverse().Filter(shouldRunAfterNode)
			runAfterAndCleanupNodes(afterNodes)

			suite.currentSpecReport.EndTime = time.Now()
			suite.currentSpecReport.RunTime = suite.currentSpecReport.EndTime.Sub(suite.currentSpecReport.StartTime)
			suite.currentSpecReport.CapturedGinkgoWriterOutput += string(suite.writer.Bytes())
			suite.currentSpecReport.CapturedStdOutErr += suite.outputInterceptor.StopInterceptingAndReturnOutput()

			if suite.currentSpecReport.State.Is(types.SpecStatePassed | types.SpecStateSkipped | types.SpecStateAborted | types.SpecStateInterrupted) {
				break
			}
		}

		//send the spec report to any attached ReportAfterEach blocks - this will update suite.currentSpecReport if failures occur in these blocks
		suite.reportEach(spec, types.NodeTypeReportAfterEach)
		suite.processCurrentSpecReport()
		if suite.currentSpecReport.State.Is(types.SpecStateFailureStates) {
			groupSucceeded = false
		}
		suite.currentSpecReport = types.SpecReport{}
	}
}
