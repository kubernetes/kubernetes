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

	succeeded              bool
	failedInARunOnceBefore bool
	continueOnFailure      bool
}

func newGroup(suite *Suite) *group {
	return &group{
		suite:                  suite,
		runOncePairs:           map[uint]runOncePairs{},
		runOnceTracker:         map[runOncePair]types.SpecState{},
		succeeded:              true,
		failedInARunOnceBefore: false,
		continueOnFailure:      false,
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
		RunningInParallel:           g.suite.isRunningInParallel(),
		IsSerial:                    spec.Nodes.HasNodeMarkedSerial(),
		IsInOrderedContainer:        !spec.Nodes.FirstNodeMarkedOrdered().IsZero(),
		MaxFlakeAttempts:            spec.Nodes.GetMaxFlakeAttempts(),
		MaxMustPassRepeatedly:       spec.Nodes.GetMaxMustPassRepeatedly(),
	}
}

func (g *group) evaluateSkipStatus(spec Spec) (types.SpecState, types.Failure) {
	if spec.Nodes.HasNodeMarkedPending() {
		return types.SpecStatePending, types.Failure{}
	}
	if spec.Skip {
		return types.SpecStateSkipped, types.Failure{}
	}
	if g.suite.interruptHandler.Status().Interrupted() || g.suite.skipAll {
		return types.SpecStateSkipped, types.Failure{}
	}
	if !g.suite.deadline.IsZero() && g.suite.deadline.Before(time.Now()) {
		return types.SpecStateSkipped, types.Failure{}
	}
	if !g.succeeded && !g.continueOnFailure {
		return types.SpecStateSkipped, g.suite.failureForLeafNodeWithMessage(spec.FirstNodeWithType(types.NodeTypeIt),
			"Spec skipped because an earlier spec in an ordered container failed")
	}
	if g.failedInARunOnceBefore && g.continueOnFailure {
		return types.SpecStateSkipped, g.suite.failureForLeafNodeWithMessage(spec.FirstNodeWithType(types.NodeTypeIt),
			"Spec skipped because a BeforeAll node failed")
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

func (g *group) attemptSpec(isFinalAttempt bool, spec Spec) bool {
	failedInARunOnceBefore := false
	pairs := g.runOncePairs[spec.SubjectID()]

	nodes := spec.Nodes.WithType(types.NodeTypeBeforeAll)
	nodes = append(nodes, spec.Nodes.WithType(types.NodeTypeBeforeEach)...).SortedByAscendingNestingLevel()
	nodes = append(nodes, spec.Nodes.WithType(types.NodeTypeJustBeforeEach).SortedByAscendingNestingLevel()...)
	nodes = append(nodes, spec.Nodes.FirstNodeWithType(types.NodeTypeIt))
	terminatingNode, terminatingPair := Node{}, runOncePair{}

	deadline := time.Time{}
	if spec.SpecTimeout() > 0 {
		deadline = time.Now().Add(spec.SpecTimeout())
	}

	for _, node := range nodes {
		oncePair := pairs.runOncePairFor(node.ID)
		if !oncePair.isZero() && g.runOnceTracker[oncePair].Is(types.SpecStatePassed) {
			continue
		}
		g.suite.currentSpecReport.State, g.suite.currentSpecReport.Failure = g.suite.runNode(node, deadline, spec.Nodes.BestTextFor(node))
		g.suite.currentSpecReport.RunTime = time.Since(g.suite.currentSpecReport.StartTime)
		if !oncePair.isZero() {
			g.runOnceTracker[oncePair] = g.suite.currentSpecReport.State
		}
		if g.suite.currentSpecReport.State != types.SpecStatePassed {
			terminatingNode, terminatingPair = node, oncePair
			failedInARunOnceBefore = !terminatingPair.isZero()
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
			var pair runOncePair
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
			case types.SpecStateFailed, types.SpecStatePanicked, types.SpecStateTimedout: // the spec has failed...
				if isFinalAttempt {
					if g.continueOnFailure {
						return isLastSpecWithPair || failedInARunOnceBefore //...we're configured to continue on failures - so we should only run if we're the last spec for this pair or if we failed in a runOnceBefore (which means we _are_ the last spec to run)
					} else {
						return true //...this was the last attempt and continueOnFailure is false therefore we are the last spec to run and so the AfterNode should run
					}
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
			state, failure := g.suite.runNode(node, deadline, spec.Nodes.BestTextFor(node))
			g.suite.currentSpecReport.RunTime = time.Since(g.suite.currentSpecReport.StartTime)
			if g.suite.currentSpecReport.State == types.SpecStatePassed || state == types.SpecStateAborted {
				g.suite.currentSpecReport.State = state
				g.suite.currentSpecReport.Failure = failure
			} else if state.Is(types.SpecStateFailureStates) {
				g.suite.currentSpecReport.AdditionalFailures = append(g.suite.currentSpecReport.AdditionalFailures, types.AdditionalFailure{State: state, Failure: failure})
			}
		}
		includeDeferCleanups = true
	}

	return failedInARunOnceBefore
}

func (g *group) run(specs Specs) {
	g.specs = specs
	g.continueOnFailure = specs[0].Nodes.FirstNodeMarkedOrdered().MarkedContinueOnFailure
	for _, spec := range g.specs {
		g.runOncePairs[spec.SubjectID()] = runOncePairsForSpec(spec)
	}

	for _, spec := range g.specs {
		g.suite.selectiveLock.Lock()
		g.suite.currentSpecReport = g.initialReportForSpec(spec)
		g.suite.selectiveLock.Unlock()

		g.suite.currentSpecReport.State, g.suite.currentSpecReport.Failure = g.evaluateSkipStatus(spec)
		g.suite.reporter.WillRun(g.suite.currentSpecReport)
		g.suite.reportEach(spec, types.NodeTypeReportBeforeEach)

		skip := g.suite.config.DryRun || g.suite.currentSpecReport.State.Is(types.SpecStateFailureStates|types.SpecStateSkipped|types.SpecStatePending)

		g.suite.currentSpecReport.StartTime = time.Now()
		failedInARunOnceBefore := false
		if !skip {
			var maxAttempts = 1

			if g.suite.config.MustPassRepeatedly > 0 {
				maxAttempts = g.suite.config.MustPassRepeatedly
				g.suite.currentSpecReport.MaxMustPassRepeatedly = maxAttempts
			} else if g.suite.currentSpecReport.MaxMustPassRepeatedly > 0 {
				maxAttempts = max(1, spec.MustPassRepeatedly())
			} else if g.suite.config.FlakeAttempts > 0 {
				maxAttempts = g.suite.config.FlakeAttempts
				g.suite.currentSpecReport.MaxFlakeAttempts = maxAttempts
			} else if g.suite.currentSpecReport.MaxFlakeAttempts > 0 {
				maxAttempts = max(1, spec.FlakeAttempts())
			}

			for attempt := 0; attempt < maxAttempts; attempt++ {
				g.suite.currentSpecReport.NumAttempts = attempt + 1
				g.suite.writer.Truncate()
				g.suite.outputInterceptor.StartInterceptingOutput()
				if attempt > 0 {
					if g.suite.currentSpecReport.MaxMustPassRepeatedly > 0 {
						g.suite.handleSpecEvent(types.SpecEvent{SpecEventType: types.SpecEventSpecRepeat, Attempt: attempt})
					}
					if g.suite.currentSpecReport.MaxFlakeAttempts > 0 {
						g.suite.handleSpecEvent(types.SpecEvent{SpecEventType: types.SpecEventSpecRetry, Attempt: attempt})
					}
				}

				failedInARunOnceBefore = g.attemptSpec(attempt == maxAttempts-1, spec)

				g.suite.currentSpecReport.EndTime = time.Now()
				g.suite.currentSpecReport.RunTime = g.suite.currentSpecReport.EndTime.Sub(g.suite.currentSpecReport.StartTime)
				g.suite.currentSpecReport.CapturedGinkgoWriterOutput += string(g.suite.writer.Bytes())
				g.suite.currentSpecReport.CapturedStdOutErr += g.suite.outputInterceptor.StopInterceptingAndReturnOutput()

				if g.suite.currentSpecReport.MaxMustPassRepeatedly > 0 {
					if g.suite.currentSpecReport.State.Is(types.SpecStateFailureStates | types.SpecStateSkipped) {
						break
					}
				}
				if g.suite.currentSpecReport.MaxFlakeAttempts > 0 {
					if g.suite.currentSpecReport.State.Is(types.SpecStatePassed | types.SpecStateSkipped | types.SpecStateAborted | types.SpecStateInterrupted) {
						break
					} else if attempt < maxAttempts-1 {
						af := types.AdditionalFailure{State: g.suite.currentSpecReport.State, Failure: g.suite.currentSpecReport.Failure}
						af.Failure.Message = fmt.Sprintf("Failure recorded during attempt %d:\n%s", attempt+1, af.Failure.Message)
						g.suite.currentSpecReport.AdditionalFailures = append(g.suite.currentSpecReport.AdditionalFailures, af)
					}
				}
			}
		}

		g.suite.reportEach(spec, types.NodeTypeReportAfterEach)
		g.suite.processCurrentSpecReport()
		if g.suite.currentSpecReport.State.Is(types.SpecStateFailureStates) {
			g.succeeded = false
			g.failedInARunOnceBefore = g.failedInARunOnceBefore || failedInARunOnceBefore
		}
		g.suite.selectiveLock.Lock()
		g.suite.currentSpecReport = types.SpecReport{}
		g.suite.selectiveLock.Unlock()
	}
}
