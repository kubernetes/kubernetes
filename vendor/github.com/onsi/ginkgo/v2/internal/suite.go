package internal

import (
	"fmt"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2/internal/interrupt_handler"
	"github.com/onsi/ginkgo/v2/internal/parallel_support"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
	"golang.org/x/net/context"
)

type Phase uint

const (
	PhaseBuildTopLevel Phase = iota
	PhaseBuildTree
	PhaseRun
)

var PROGRESS_REPORTER_DEADLING = 5 * time.Second

type Suite struct {
	tree               *TreeNode
	topLevelContainers Nodes

	*ProgressReporterManager

	phase Phase

	suiteNodes   Nodes
	cleanupNodes Nodes

	failer            *Failer
	reporter          reporters.Reporter
	writer            WriterInterface
	outputInterceptor OutputInterceptor
	interruptHandler  interrupt_handler.InterruptHandlerInterface
	config            types.SuiteConfig
	deadline          time.Time

	skipAll              bool
	report               types.Report
	currentSpecReport    types.SpecReport
	currentNode          Node
	currentNodeStartTime time.Time

	currentSpecContext *specContext

	currentByStep types.SpecEvent
	timelineOrder int

	/*
		We don't need to lock around all operations.  Just those that *could* happen concurrently.

		Suite, generally, only runs one node at a time - and so the possibiity for races is small.  In fact, the presence of a race usually indicates the user has launched a goroutine that has leaked past the node it was launched in.

		However, there are some operations that can happen concurrently:

		- AddReportEntry and CurrentSpecReport can be accessed at any point by the user - including in goroutines that outlive the node intentionally (see, e.g. #1020).  They both form a self-contained read-write pair and so a lock in them is sufficent.
		- generateProgressReport can be invoked at any point in time by an interrupt or a progres poll.  Moreover, it requires access to currentSpecReport, currentNode, currentNodeStartTime, and progressStepCursor.  To make it threadsafe we need to lock around generateProgressReport when we read those variables _and_ everywhere those variables are *written*.  In general we don't need to worry about all possible field writes to these variables as what `generateProgressReport` does with these variables is fairly selective (hence the name of the lock).  Specifically, we dont' need to lock around state and failure message changes on `currentSpecReport` - just the setting of the variable itself.
	*/
	selectiveLock *sync.Mutex

	client parallel_support.Client

	annotateFn AnnotateFunc
}

func NewSuite() *Suite {
	return &Suite{
		tree:                    &TreeNode{},
		phase:                   PhaseBuildTopLevel,
		ProgressReporterManager: NewProgressReporterManager(),

		selectiveLock: &sync.Mutex{},
	}
}

func (suite *Suite) Clone() (*Suite, error) {
	if suite.phase != PhaseBuildTopLevel {
		return nil, fmt.Errorf("cannot clone suite after tree has been built")
	}
	return &Suite{
		tree:                    &TreeNode{},
		phase:                   PhaseBuildTopLevel,
		ProgressReporterManager: NewProgressReporterManager(),
		topLevelContainers:      suite.topLevelContainers.Clone(),
		suiteNodes:              suite.suiteNodes.Clone(),
		selectiveLock:           &sync.Mutex{},
	}, nil
}

func (suite *Suite) BuildTree() error {
	// During PhaseBuildTopLevel, the top level containers are stored in suite.topLevelCotainers and entered
	// We now enter PhaseBuildTree where these top level containers are entered and added to the spec tree
	suite.phase = PhaseBuildTree
	for _, topLevelContainer := range suite.topLevelContainers {
		err := suite.PushNode(topLevelContainer)
		if err != nil {
			return err
		}
	}
	return nil
}

func (suite *Suite) Run(description string, suiteLabels Labels, suitePath string, failer *Failer, reporter reporters.Reporter, writer WriterInterface, outputInterceptor OutputInterceptor, interruptHandler interrupt_handler.InterruptHandlerInterface, client parallel_support.Client, progressSignalRegistrar ProgressSignalRegistrar, suiteConfig types.SuiteConfig) (bool, bool) {
	if suite.phase != PhaseBuildTree {
		panic("cannot run before building the tree = call suite.BuildTree() first")
	}
	ApplyNestedFocusPolicyToTree(suite.tree)
	specs := GenerateSpecsFromTreeRoot(suite.tree)
	if suite.annotateFn != nil {
		for _, spec := range specs {
			suite.annotateFn(spec.Text(), spec)
		}
	}
	specs, hasProgrammaticFocus := ApplyFocusToSpecs(specs, description, suiteLabels, suiteConfig)

	suite.phase = PhaseRun
	suite.client = client
	suite.failer = failer
	suite.reporter = reporter
	suite.writer = writer
	suite.outputInterceptor = outputInterceptor
	suite.interruptHandler = interruptHandler
	suite.config = suiteConfig

	if suite.config.Timeout > 0 {
		suite.deadline = time.Now().Add(suite.config.Timeout)
	}

	cancelProgressHandler := progressSignalRegistrar(suite.handleProgressSignal)

	success := suite.runSpecs(description, suiteLabels, suitePath, hasProgrammaticFocus, specs)

	cancelProgressHandler()

	return success, hasProgrammaticFocus
}

func (suite *Suite) InRunPhase() bool {
	return suite.phase == PhaseRun
}

/*
  Tree Construction methods

  PushNode is used during PhaseBuildTopLevel and PhaseBuildTree
*/

func (suite *Suite) PushNode(node Node) error {
	if node.NodeType.Is(types.NodeTypeCleanupInvalid | types.NodeTypeCleanupAfterEach | types.NodeTypeCleanupAfterAll | types.NodeTypeCleanupAfterSuite) {
		return suite.pushCleanupNode(node)
	}

	if node.NodeType.Is(types.NodeTypeBeforeSuite | types.NodeTypeAfterSuite | types.NodeTypeSynchronizedBeforeSuite | types.NodeTypeSynchronizedAfterSuite | types.NodeTypeBeforeSuite | types.NodeTypeReportBeforeSuite | types.NodeTypeReportAfterSuite) {
		return suite.pushSuiteNode(node)
	}

	if suite.phase == PhaseRun {
		return types.GinkgoErrors.PushingNodeInRunPhase(node.NodeType, node.CodeLocation)
	}

	if node.MarkedSerial {
		firstOrderedNode := suite.tree.AncestorNodeChain().FirstNodeMarkedOrdered()
		if !firstOrderedNode.IsZero() && !firstOrderedNode.MarkedSerial {
			return types.GinkgoErrors.InvalidSerialNodeInNonSerialOrderedContainer(node.CodeLocation, node.NodeType)
		}
	}

	if node.NodeType.Is(types.NodeTypeBeforeAll | types.NodeTypeAfterAll) {
		firstOrderedNode := suite.tree.AncestorNodeChain().FirstNodeMarkedOrdered()
		if firstOrderedNode.IsZero() {
			return types.GinkgoErrors.SetupNodeNotInOrderedContainer(node.CodeLocation, node.NodeType)
		}
	}

	if node.MarkedContinueOnFailure {
		firstOrderedNode := suite.tree.AncestorNodeChain().FirstNodeMarkedOrdered()
		if !firstOrderedNode.IsZero() {
			return types.GinkgoErrors.InvalidContinueOnFailureDecoration(node.CodeLocation)
		}
	}

	if node.NodeType == types.NodeTypeContainer {
		// During PhaseBuildTopLevel we only track the top level containers without entering them
		// We only enter the top level container nodes during PhaseBuildTree
		//
		// This ensures the tree is only constructed after `go spec` has called `flag.Parse()` and gives
		// the user an opportunity to load suiteConfiguration information in the `TestX` go spec hook just before `RunSpecs`
		// is invoked.  This makes the lifecycle easier to reason about and solves issues like #693.
		if suite.phase == PhaseBuildTopLevel {
			suite.topLevelContainers = append(suite.topLevelContainers, node)
			return nil
		}
		if suite.phase == PhaseBuildTree {
			parentTree := suite.tree
			suite.tree = &TreeNode{Node: node}
			parentTree.AppendChild(suite.tree)
			err := func() (err error) {
				defer func() {
					if e := recover(); e != nil {
						err = types.GinkgoErrors.CaughtPanicDuringABuildPhase(e, node.CodeLocation)
					}
				}()
				node.Body(nil)
				return err
			}()
			suite.tree = parentTree
			return err
		}
	} else {
		suite.tree.AppendChild(&TreeNode{Node: node})
		return nil
	}

	return nil
}

func (suite *Suite) pushSuiteNode(node Node) error {
	if suite.phase == PhaseBuildTree {
		return types.GinkgoErrors.SuiteNodeInNestedContext(node.NodeType, node.CodeLocation)
	}

	if suite.phase == PhaseRun {
		return types.GinkgoErrors.SuiteNodeDuringRunPhase(node.NodeType, node.CodeLocation)
	}

	switch node.NodeType {
	case types.NodeTypeBeforeSuite, types.NodeTypeSynchronizedBeforeSuite:
		existingBefores := suite.suiteNodes.WithType(types.NodeTypeBeforeSuite | types.NodeTypeSynchronizedBeforeSuite)
		if len(existingBefores) > 0 {
			return types.GinkgoErrors.MultipleBeforeSuiteNodes(node.NodeType, node.CodeLocation, existingBefores[0].NodeType, existingBefores[0].CodeLocation)
		}
	case types.NodeTypeAfterSuite, types.NodeTypeSynchronizedAfterSuite:
		existingAfters := suite.suiteNodes.WithType(types.NodeTypeAfterSuite | types.NodeTypeSynchronizedAfterSuite)
		if len(existingAfters) > 0 {
			return types.GinkgoErrors.MultipleAfterSuiteNodes(node.NodeType, node.CodeLocation, existingAfters[0].NodeType, existingAfters[0].CodeLocation)
		}
	}

	suite.suiteNodes = append(suite.suiteNodes, node)
	return nil
}

func (suite *Suite) pushCleanupNode(node Node) error {
	if suite.phase != PhaseRun || suite.currentNode.IsZero() {
		return types.GinkgoErrors.PushingCleanupNodeDuringTreeConstruction(node.CodeLocation)
	}

	switch suite.currentNode.NodeType {
	case types.NodeTypeBeforeSuite, types.NodeTypeSynchronizedBeforeSuite, types.NodeTypeAfterSuite, types.NodeTypeSynchronizedAfterSuite:
		node.NodeType = types.NodeTypeCleanupAfterSuite
	case types.NodeTypeBeforeAll, types.NodeTypeAfterAll:
		node.NodeType = types.NodeTypeCleanupAfterAll
	case types.NodeTypeReportBeforeEach, types.NodeTypeReportAfterEach, types.NodeTypeReportBeforeSuite, types.NodeTypeReportAfterSuite:
		return types.GinkgoErrors.PushingCleanupInReportingNode(node.CodeLocation, suite.currentNode.NodeType)
	case types.NodeTypeCleanupInvalid, types.NodeTypeCleanupAfterEach, types.NodeTypeCleanupAfterAll, types.NodeTypeCleanupAfterSuite:
		return types.GinkgoErrors.PushingCleanupInCleanupNode(node.CodeLocation)
	default:
		node.NodeType = types.NodeTypeCleanupAfterEach
	}

	node.NodeIDWhereCleanupWasGenerated = suite.currentNode.ID
	node.NestingLevel = suite.currentNode.NestingLevel
	suite.selectiveLock.Lock()
	suite.cleanupNodes = append(suite.cleanupNodes, node)
	suite.selectiveLock.Unlock()

	return nil
}

func (suite *Suite) generateTimelineLocation() types.TimelineLocation {
	suite.selectiveLock.Lock()
	defer suite.selectiveLock.Unlock()

	suite.timelineOrder += 1
	return types.TimelineLocation{
		Offset: len(suite.currentSpecReport.CapturedGinkgoWriterOutput) + suite.writer.Len(),
		Order:  suite.timelineOrder,
		Time:   time.Now(),
	}
}

func (suite *Suite) handleSpecEvent(event types.SpecEvent) types.SpecEvent {
	event.TimelineLocation = suite.generateTimelineLocation()
	suite.selectiveLock.Lock()
	suite.currentSpecReport.SpecEvents = append(suite.currentSpecReport.SpecEvents, event)
	suite.selectiveLock.Unlock()
	suite.reporter.EmitSpecEvent(event)
	return event
}

func (suite *Suite) handleSpecEventEnd(eventType types.SpecEventType, startEvent types.SpecEvent) {
	event := startEvent
	event.SpecEventType = eventType
	event.TimelineLocation = suite.generateTimelineLocation()
	event.Duration = event.TimelineLocation.Time.Sub(startEvent.TimelineLocation.Time)
	suite.selectiveLock.Lock()
	suite.currentSpecReport.SpecEvents = append(suite.currentSpecReport.SpecEvents, event)
	suite.selectiveLock.Unlock()
	suite.reporter.EmitSpecEvent(event)
}

func (suite *Suite) By(text string, callback ...func()) error {
	cl := types.NewCodeLocation(2)
	if suite.phase != PhaseRun {
		return types.GinkgoErrors.ByNotDuringRunPhase(cl)
	}

	event := suite.handleSpecEvent(types.SpecEvent{
		SpecEventType: types.SpecEventByStart,
		CodeLocation:  cl,
		Message:       text,
	})
	suite.selectiveLock.Lock()
	suite.currentByStep = event
	suite.selectiveLock.Unlock()

	if len(callback) == 1 {
		defer func() {
			suite.selectiveLock.Lock()
			suite.currentByStep = types.SpecEvent{}
			suite.selectiveLock.Unlock()
			suite.handleSpecEventEnd(types.SpecEventByEnd, event)
		}()
		callback[0]()
	} else if len(callback) > 1 {
		panic("just one callback per By, please")
	}
	return nil
}

/*
Spec Running methods - used during PhaseRun
*/
func (suite *Suite) CurrentSpecReport() types.SpecReport {
	suite.selectiveLock.Lock()
	defer suite.selectiveLock.Unlock()
	report := suite.currentSpecReport
	if suite.writer != nil {
		report.CapturedGinkgoWriterOutput = string(suite.writer.Bytes())
	}
	report.ReportEntries = make([]ReportEntry, len(report.ReportEntries))
	copy(report.ReportEntries, suite.currentSpecReport.ReportEntries)
	return report
}

// Only valid in the preview context.  In general suite.report only includes
// the specs run by _this_ node - it is only at the end of the suite that
// the parallel reports are aggregated.  However in the preview context we run
// in series and
func (suite *Suite) GetPreviewReport() types.Report {
	suite.selectiveLock.Lock()
	defer suite.selectiveLock.Unlock()
	return suite.report
}

func (suite *Suite) AddReportEntry(entry ReportEntry) error {
	if suite.phase != PhaseRun {
		return types.GinkgoErrors.AddReportEntryNotDuringRunPhase(entry.Location)
	}
	entry.TimelineLocation = suite.generateTimelineLocation()
	entry.Time = entry.TimelineLocation.Time
	suite.selectiveLock.Lock()
	suite.currentSpecReport.ReportEntries = append(suite.currentSpecReport.ReportEntries, entry)
	suite.selectiveLock.Unlock()
	suite.reporter.EmitReportEntry(entry)
	return nil
}

func (suite *Suite) generateProgressReport(fullReport bool) types.ProgressReport {
	timelineLocation := suite.generateTimelineLocation()
	suite.selectiveLock.Lock()
	defer suite.selectiveLock.Unlock()

	deadline, cancel := context.WithTimeout(context.Background(), PROGRESS_REPORTER_DEADLING)
	defer cancel()
	var additionalReports []string
	if suite.currentSpecContext != nil {
		additionalReports = append(additionalReports, suite.currentSpecContext.QueryProgressReporters(deadline, suite.failer)...)
	}
	additionalReports = append(additionalReports, suite.QueryProgressReporters(deadline, suite.failer)...)
	gwOutput := suite.currentSpecReport.CapturedGinkgoWriterOutput + string(suite.writer.Bytes())
	pr, err := NewProgressReport(suite.isRunningInParallel(), suite.currentSpecReport, suite.currentNode, suite.currentNodeStartTime, suite.currentByStep, gwOutput, timelineLocation, additionalReports, suite.config.SourceRoots, fullReport)

	if err != nil {
		fmt.Printf("{{red}}Failed to generate progress report:{{/}}\n%s\n", err.Error())
	}
	return pr
}

func (suite *Suite) handleProgressSignal() {
	report := suite.generateProgressReport(false)
	report.Message = "{{bold}}You've requested a progress report:{{/}}"
	suite.emitProgressReport(report)
}

func (suite *Suite) emitProgressReport(report types.ProgressReport) {
	suite.selectiveLock.Lock()
	suite.currentSpecReport.ProgressReports = append(suite.currentSpecReport.ProgressReports, report.WithoutCapturedGinkgoWriterOutput())
	suite.selectiveLock.Unlock()

	suite.reporter.EmitProgressReport(report)
	if suite.isRunningInParallel() {
		err := suite.client.PostEmitProgressReport(report)
		if err != nil {
			fmt.Println(err.Error())
		}
	}
}

func (suite *Suite) isRunningInParallel() bool {
	return suite.config.ParallelTotal > 1
}

func (suite *Suite) processCurrentSpecReport() {
	suite.reporter.DidRun(suite.currentSpecReport)
	if suite.isRunningInParallel() {
		suite.client.PostDidRun(suite.currentSpecReport)
	}
	suite.report.SpecReports = append(suite.report.SpecReports, suite.currentSpecReport)

	if suite.currentSpecReport.State.Is(types.SpecStateFailureStates) {
		suite.report.SuiteSucceeded = false
		if suite.config.FailFast || suite.currentSpecReport.State.Is(types.SpecStateAborted) {
			suite.skipAll = true
			if suite.isRunningInParallel() {
				suite.client.PostAbort()
			}
		}
	}
}

func (suite *Suite) runSpecs(description string, suiteLabels Labels, suitePath string, hasProgrammaticFocus bool, specs Specs) bool {
	numSpecsThatWillBeRun := specs.CountWithoutSkip()

	suite.report = types.Report{
		SuitePath:                 suitePath,
		SuiteDescription:          description,
		SuiteLabels:               suiteLabels,
		SuiteConfig:               suite.config,
		SuiteHasProgrammaticFocus: hasProgrammaticFocus,
		PreRunStats: types.PreRunStats{
			TotalSpecs:       len(specs),
			SpecsThatWillRun: numSpecsThatWillBeRun,
		},
		StartTime: time.Now(),
	}

	suite.reporter.SuiteWillBegin(suite.report)
	if suite.isRunningInParallel() {
		suite.client.PostSuiteWillBegin(suite.report)
	}

	suite.report.SuiteSucceeded = true

	suite.runReportSuiteNodesIfNeedBe(types.NodeTypeReportBeforeSuite)

	ranBeforeSuite := suite.report.SuiteSucceeded
	if suite.report.SuiteSucceeded {
		suite.runBeforeSuite(numSpecsThatWillBeRun)
	}

	if suite.report.SuiteSucceeded {
		groupedSpecIndices, serialGroupedSpecIndices := OrderSpecs(specs, suite.config)
		nextIndex := MakeIncrementingIndexCounter()
		if suite.isRunningInParallel() {
			nextIndex = suite.client.FetchNextCounter
		}

		for {
			groupedSpecIdx, err := nextIndex()
			if err != nil {
				suite.report.SpecialSuiteFailureReasons = append(suite.report.SpecialSuiteFailureReasons, fmt.Sprintf("Failed to iterate over specs:\n%s", err.Error()))
				suite.report.SuiteSucceeded = false
				break
			}

			if groupedSpecIdx >= len(groupedSpecIndices) {
				if suite.config.ParallelProcess == 1 && len(serialGroupedSpecIndices) > 0 {
					groupedSpecIndices, serialGroupedSpecIndices, nextIndex = serialGroupedSpecIndices, GroupedSpecIndices{}, MakeIncrementingIndexCounter()
					suite.client.BlockUntilNonprimaryProcsHaveFinished()
					continue
				}
				break
			}

			// the complexity for running groups of specs is very high because of Ordered containers and FlakeAttempts
			// we encapsulate that complexity in the notion of a Group that can run
			// Group is really just an extension of suite so it gets passed a suite and has access to all its internals
			// Note that group is stateful and intended for single use!
			newGroup(suite).run(specs.AtIndices(groupedSpecIndices[groupedSpecIdx]))
		}

		if specs.HasAnySpecsMarkedPending() && suite.config.FailOnPending {
			suite.report.SpecialSuiteFailureReasons = append(suite.report.SpecialSuiteFailureReasons, "Detected pending specs and --fail-on-pending is set")
			suite.report.SuiteSucceeded = false
		}
	}

	if ranBeforeSuite {
		suite.runAfterSuiteCleanup(numSpecsThatWillBeRun)
	}

	interruptStatus := suite.interruptHandler.Status()
	if interruptStatus.Interrupted() {
		suite.report.SpecialSuiteFailureReasons = append(suite.report.SpecialSuiteFailureReasons, interruptStatus.Cause.String())
		suite.report.SuiteSucceeded = false
	}
	suite.report.EndTime = time.Now()
	suite.report.RunTime = suite.report.EndTime.Sub(suite.report.StartTime)
	if !suite.deadline.IsZero() && suite.report.EndTime.After(suite.deadline) {
		suite.report.SpecialSuiteFailureReasons = append(suite.report.SpecialSuiteFailureReasons, "Suite Timeout Elapsed")
		suite.report.SuiteSucceeded = false
	}

	suite.runReportSuiteNodesIfNeedBe(types.NodeTypeReportAfterSuite)
	suite.reporter.SuiteDidEnd(suite.report)
	if suite.isRunningInParallel() {
		suite.client.PostSuiteDidEnd(suite.report)
	}

	return suite.report.SuiteSucceeded
}

func (suite *Suite) runBeforeSuite(numSpecsThatWillBeRun int) {
	beforeSuiteNode := suite.suiteNodes.FirstNodeWithType(types.NodeTypeBeforeSuite | types.NodeTypeSynchronizedBeforeSuite)
	if !beforeSuiteNode.IsZero() && numSpecsThatWillBeRun > 0 {
		suite.selectiveLock.Lock()
		suite.currentSpecReport = types.SpecReport{
			LeafNodeType:      beforeSuiteNode.NodeType,
			LeafNodeLocation:  beforeSuiteNode.CodeLocation,
			ParallelProcess:   suite.config.ParallelProcess,
			RunningInParallel: suite.isRunningInParallel(),
		}
		suite.selectiveLock.Unlock()

		suite.reporter.WillRun(suite.currentSpecReport)
		suite.runSuiteNode(beforeSuiteNode)
		if suite.currentSpecReport.State.Is(types.SpecStateSkipped) {
			suite.report.SpecialSuiteFailureReasons = append(suite.report.SpecialSuiteFailureReasons, "Suite skipped in BeforeSuite")
			suite.skipAll = true
		}
		suite.processCurrentSpecReport()
	}
}

func (suite *Suite) runAfterSuiteCleanup(numSpecsThatWillBeRun int) {
	afterSuiteNode := suite.suiteNodes.FirstNodeWithType(types.NodeTypeAfterSuite | types.NodeTypeSynchronizedAfterSuite)
	if !afterSuiteNode.IsZero() && numSpecsThatWillBeRun > 0 {
		suite.selectiveLock.Lock()
		suite.currentSpecReport = types.SpecReport{
			LeafNodeType:      afterSuiteNode.NodeType,
			LeafNodeLocation:  afterSuiteNode.CodeLocation,
			ParallelProcess:   suite.config.ParallelProcess,
			RunningInParallel: suite.isRunningInParallel(),
		}
		suite.selectiveLock.Unlock()

		suite.reporter.WillRun(suite.currentSpecReport)
		suite.runSuiteNode(afterSuiteNode)
		suite.processCurrentSpecReport()
	}

	afterSuiteCleanup := suite.cleanupNodes.WithType(types.NodeTypeCleanupAfterSuite).Reverse()
	if len(afterSuiteCleanup) > 0 {
		for _, cleanupNode := range afterSuiteCleanup {
			suite.selectiveLock.Lock()
			suite.currentSpecReport = types.SpecReport{
				LeafNodeType:      cleanupNode.NodeType,
				LeafNodeLocation:  cleanupNode.CodeLocation,
				ParallelProcess:   suite.config.ParallelProcess,
				RunningInParallel: suite.isRunningInParallel(),
			}
			suite.selectiveLock.Unlock()

			suite.reporter.WillRun(suite.currentSpecReport)
			suite.runSuiteNode(cleanupNode)
			suite.processCurrentSpecReport()
		}
	}
}

func (suite *Suite) reportEach(spec Spec, nodeType types.NodeType) {
	nodes := spec.Nodes.WithType(nodeType)
	if nodeType == types.NodeTypeReportAfterEach {
		nodes = nodes.SortedByDescendingNestingLevel()
	}
	if nodeType == types.NodeTypeReportBeforeEach {
		nodes = nodes.SortedByAscendingNestingLevel()
	}
	if len(nodes) == 0 {
		return
	}

	for i := range nodes {
		suite.writer.Truncate()
		suite.outputInterceptor.StartInterceptingOutput()
		report := suite.currentSpecReport
		nodes[i].Body = func(SpecContext) {
			nodes[i].ReportEachBody(report)
		}
		state, failure := suite.runNode(nodes[i], time.Time{}, spec.Nodes.BestTextFor(nodes[i]))

		// If the spec is not in a failure state (i.e. it's Passed/Skipped/Pending) and the reporter has failed, override the state.
		// Also, if the reporter is every aborted - always override the state to propagate the abort
		if (!suite.currentSpecReport.State.Is(types.SpecStateFailureStates) && state.Is(types.SpecStateFailureStates)) || state.Is(types.SpecStateAborted) {
			suite.currentSpecReport.State = state
			suite.currentSpecReport.Failure = failure
		}
		suite.currentSpecReport.CapturedGinkgoWriterOutput += string(suite.writer.Bytes())
		suite.currentSpecReport.CapturedStdOutErr += suite.outputInterceptor.StopInterceptingAndReturnOutput()
	}
}

func (suite *Suite) runSuiteNode(node Node) {
	if suite.config.DryRun {
		suite.currentSpecReport.State = types.SpecStatePassed
		return
	}

	suite.writer.Truncate()
	suite.outputInterceptor.StartInterceptingOutput()
	suite.currentSpecReport.StartTime = time.Now()

	var err error
	switch node.NodeType {
	case types.NodeTypeBeforeSuite, types.NodeTypeAfterSuite:
		suite.currentSpecReport.State, suite.currentSpecReport.Failure = suite.runNode(node, time.Time{}, "")
	case types.NodeTypeCleanupAfterSuite:
		if suite.config.ParallelTotal > 1 && suite.config.ParallelProcess == 1 {
			err = suite.client.BlockUntilNonprimaryProcsHaveFinished()
		}
		if err == nil {
			suite.currentSpecReport.State, suite.currentSpecReport.Failure = suite.runNode(node, time.Time{}, "")
		}
	case types.NodeTypeSynchronizedBeforeSuite:
		var data []byte
		var runAllProcs bool
		if suite.config.ParallelProcess == 1 {
			if suite.config.ParallelTotal > 1 {
				suite.outputInterceptor.StopInterceptingAndReturnOutput()
				suite.outputInterceptor.StartInterceptingOutputAndForwardTo(suite.client)
			}
			node.Body = func(c SpecContext) { data = node.SynchronizedBeforeSuiteProc1Body(c) }
			node.HasContext = node.SynchronizedBeforeSuiteProc1BodyHasContext
			suite.currentSpecReport.State, suite.currentSpecReport.Failure = suite.runNode(node, time.Time{}, "")
			if suite.config.ParallelTotal > 1 {
				suite.currentSpecReport.CapturedStdOutErr += suite.outputInterceptor.StopInterceptingAndReturnOutput()
				suite.outputInterceptor.StartInterceptingOutput()
				if suite.currentSpecReport.State.Is(types.SpecStatePassed) {
					err = suite.client.PostSynchronizedBeforeSuiteCompleted(types.SpecStatePassed, data)
				} else {
					err = suite.client.PostSynchronizedBeforeSuiteCompleted(suite.currentSpecReport.State, nil)
				}
			}
			runAllProcs = suite.currentSpecReport.State.Is(types.SpecStatePassed) && err == nil
		} else {
			var proc1State types.SpecState
			proc1State, data, err = suite.client.BlockUntilSynchronizedBeforeSuiteData()
			switch proc1State {
			case types.SpecStatePassed:
				runAllProcs = true
			case types.SpecStateFailed, types.SpecStatePanicked, types.SpecStateTimedout:
				err = types.GinkgoErrors.SynchronizedBeforeSuiteFailedOnProc1()
			case types.SpecStateInterrupted, types.SpecStateAborted, types.SpecStateSkipped:
				suite.currentSpecReport.State = proc1State
			}
		}
		if runAllProcs {
			node.Body = func(c SpecContext) { node.SynchronizedBeforeSuiteAllProcsBody(c, data) }
			node.HasContext = node.SynchronizedBeforeSuiteAllProcsBodyHasContext
			suite.currentSpecReport.State, suite.currentSpecReport.Failure = suite.runNode(node, time.Time{}, "")
		}
	case types.NodeTypeSynchronizedAfterSuite:
		node.Body = node.SynchronizedAfterSuiteAllProcsBody
		node.HasContext = node.SynchronizedAfterSuiteAllProcsBodyHasContext
		suite.currentSpecReport.State, suite.currentSpecReport.Failure = suite.runNode(node, time.Time{}, "")
		if suite.config.ParallelProcess == 1 {
			if suite.config.ParallelTotal > 1 {
				err = suite.client.BlockUntilNonprimaryProcsHaveFinished()
			}
			if err == nil {
				if suite.config.ParallelTotal > 1 {
					suite.currentSpecReport.CapturedStdOutErr += suite.outputInterceptor.StopInterceptingAndReturnOutput()
					suite.outputInterceptor.StartInterceptingOutputAndForwardTo(suite.client)
				}

				node.Body = node.SynchronizedAfterSuiteProc1Body
				node.HasContext = node.SynchronizedAfterSuiteProc1BodyHasContext
				state, failure := suite.runNode(node, time.Time{}, "")
				if suite.currentSpecReport.State.Is(types.SpecStatePassed) {
					suite.currentSpecReport.State, suite.currentSpecReport.Failure = state, failure
				}
			}
		}
	}

	if err != nil && !suite.currentSpecReport.State.Is(types.SpecStateFailureStates) {
		suite.currentSpecReport.State, suite.currentSpecReport.Failure = types.SpecStateFailed, suite.failureForLeafNodeWithMessage(node, err.Error())
		suite.reporter.EmitFailure(suite.currentSpecReport.State, suite.currentSpecReport.Failure)
	}

	suite.currentSpecReport.EndTime = time.Now()
	suite.currentSpecReport.RunTime = suite.currentSpecReport.EndTime.Sub(suite.currentSpecReport.StartTime)
	suite.currentSpecReport.CapturedGinkgoWriterOutput = string(suite.writer.Bytes())
	suite.currentSpecReport.CapturedStdOutErr += suite.outputInterceptor.StopInterceptingAndReturnOutput()
}

func (suite *Suite) runReportSuiteNodesIfNeedBe(nodeType types.NodeType) {
	nodes := suite.suiteNodes.WithType(nodeType)
	// only run ReportAfterSuite on proc 1
	if nodeType.Is(types.NodeTypeReportAfterSuite) && suite.config.ParallelProcess != 1 {
		return
	}
	// if we're running ReportBeforeSuite on proc > 1 - we should wait until proc 1 has completed
	if nodeType.Is(types.NodeTypeReportBeforeSuite) && suite.config.ParallelProcess != 1 && len(nodes) > 0 {
		state, err := suite.client.BlockUntilReportBeforeSuiteCompleted()
		if err != nil || state.Is(types.SpecStateFailed) {
			suite.report.SuiteSucceeded = false
		}
		return
	}

	for _, node := range nodes {
		suite.selectiveLock.Lock()
		suite.currentSpecReport = types.SpecReport{
			LeafNodeType:      node.NodeType,
			LeafNodeLocation:  node.CodeLocation,
			LeafNodeText:      node.Text,
			ParallelProcess:   suite.config.ParallelProcess,
			RunningInParallel: suite.isRunningInParallel(),
		}
		suite.selectiveLock.Unlock()

		suite.reporter.WillRun(suite.currentSpecReport)
		suite.runReportSuiteNode(node, suite.report)
		suite.processCurrentSpecReport()
	}

	// if we're running ReportBeforeSuite and we're running in parallel - we shuld tell the other procs that we're done
	if nodeType.Is(types.NodeTypeReportBeforeSuite) && suite.isRunningInParallel() && len(nodes) > 0 {
		if suite.report.SuiteSucceeded {
			suite.client.PostReportBeforeSuiteCompleted(types.SpecStatePassed)
		} else {
			suite.client.PostReportBeforeSuiteCompleted(types.SpecStateFailed)
		}
	}
}

func (suite *Suite) runReportSuiteNode(node Node, report types.Report) {
	suite.writer.Truncate()
	suite.outputInterceptor.StartInterceptingOutput()
	suite.currentSpecReport.StartTime = time.Now()

	// if we're running a ReportAfterSuite in parallel (on proc 1) we (a) wait until other procs have exited and
	// (b) always fetch the latest report as prior ReportAfterSuites will contribute to it
	if node.NodeType.Is(types.NodeTypeReportAfterSuite) && suite.isRunningInParallel() {
		aggregatedReport, err := suite.client.BlockUntilAggregatedNonprimaryProcsReport()
		if err != nil {
			suite.currentSpecReport.State, suite.currentSpecReport.Failure = types.SpecStateFailed, suite.failureForLeafNodeWithMessage(node, err.Error())
			suite.reporter.EmitFailure(suite.currentSpecReport.State, suite.currentSpecReport.Failure)
			return
		}
		report = report.Add(aggregatedReport)
	}

	node.Body = func(SpecContext) { node.ReportSuiteBody(report) }
	suite.currentSpecReport.State, suite.currentSpecReport.Failure = suite.runNode(node, time.Time{}, "")

	suite.currentSpecReport.EndTime = time.Now()
	suite.currentSpecReport.RunTime = suite.currentSpecReport.EndTime.Sub(suite.currentSpecReport.StartTime)
	suite.currentSpecReport.CapturedGinkgoWriterOutput = string(suite.writer.Bytes())
	suite.currentSpecReport.CapturedStdOutErr = suite.outputInterceptor.StopInterceptingAndReturnOutput()
}

func (suite *Suite) runNode(node Node, specDeadline time.Time, text string) (types.SpecState, types.Failure) {
	if node.NodeType.Is(types.NodeTypeCleanupAfterEach | types.NodeTypeCleanupAfterAll | types.NodeTypeCleanupAfterSuite) {
		suite.cleanupNodes = suite.cleanupNodes.WithoutNode(node)
	}

	interruptStatus := suite.interruptHandler.Status()
	if interruptStatus.Level == interrupt_handler.InterruptLevelBailOut {
		return types.SpecStateSkipped, types.Failure{}
	}
	if interruptStatus.Level == interrupt_handler.InterruptLevelReportOnly && !node.NodeType.Is(types.NodeTypesAllowedDuringReportInterrupt) {
		return types.SpecStateSkipped, types.Failure{}
	}
	if interruptStatus.Level == interrupt_handler.InterruptLevelCleanupAndReport && !node.NodeType.Is(types.NodeTypesAllowedDuringReportInterrupt|types.NodeTypesAllowedDuringCleanupInterrupt) {
		return types.SpecStateSkipped, types.Failure{}
	}

	suite.selectiveLock.Lock()
	suite.currentNode = node
	suite.currentNodeStartTime = time.Now()
	suite.currentByStep = types.SpecEvent{}
	suite.selectiveLock.Unlock()
	defer func() {
		suite.selectiveLock.Lock()
		suite.currentNode = Node{}
		suite.currentNodeStartTime = time.Time{}
		suite.selectiveLock.Unlock()
	}()

	if text == "" {
		text = "TOP-LEVEL"
	}
	event := suite.handleSpecEvent(types.SpecEvent{
		SpecEventType: types.SpecEventNodeStart,
		NodeType:      node.NodeType,
		Message:       text,
		CodeLocation:  node.CodeLocation,
	})
	defer func() {
		suite.handleSpecEventEnd(types.SpecEventNodeEnd, event)
	}()

	var failure types.Failure
	failure.FailureNodeType, failure.FailureNodeLocation = node.NodeType, node.CodeLocation
	if node.NodeType.Is(types.NodeTypeIt) || node.NodeType.Is(types.NodeTypesForSuiteLevelNodes) {
		failure.FailureNodeContext = types.FailureNodeIsLeafNode
	} else if node.NestingLevel <= 0 {
		failure.FailureNodeContext = types.FailureNodeAtTopLevel
	} else {
		failure.FailureNodeContext, failure.FailureNodeContainerIndex = types.FailureNodeInContainer, node.NestingLevel-1
	}
	var outcome types.SpecState

	gracePeriod := suite.config.GracePeriod
	if node.GracePeriod >= 0 {
		gracePeriod = node.GracePeriod
	}

	now := time.Now()
	deadline := suite.deadline
	timeoutInPlay := "suite"
	if deadline.IsZero() || (!specDeadline.IsZero() && specDeadline.Before(deadline)) {
		deadline = specDeadline
		timeoutInPlay = "spec"
	}
	if node.NodeTimeout > 0 && (deadline.IsZero() || deadline.Sub(now) > node.NodeTimeout) {
		deadline = now.Add(node.NodeTimeout)
		timeoutInPlay = "node"
	}
	if (!deadline.IsZero() && deadline.Before(now)) || interruptStatus.Interrupted() {
		//we're out of time already.  let's wait for a NodeTimeout if we have it, or GracePeriod if we don't
		if node.NodeTimeout > 0 {
			deadline = now.Add(node.NodeTimeout)
			timeoutInPlay = "node"
		} else {
			deadline = now.Add(gracePeriod)
			timeoutInPlay = "grace period"
		}
	}

	if !node.HasContext {
		// this maps onto the pre-context behavior:
		// - an interrupted node exits immediately.  with this, context-less nodes that are in a spec with a SpecTimeout and/or are interrupted by other means will simply exit immediately after the timeout/interrupt
		// - clean up nodes have up to GracePeriod (formerly hard-coded at 30s) to complete before they are interrupted
		gracePeriod = 0
	}

	sc := NewSpecContext(suite)
	defer sc.cancel(fmt.Errorf("spec has finished"))

	suite.selectiveLock.Lock()
	suite.currentSpecContext = sc
	suite.selectiveLock.Unlock()

	var deadlineChannel <-chan time.Time
	if !deadline.IsZero() {
		deadlineChannel = time.After(deadline.Sub(now))
	}
	var gracePeriodChannel <-chan time.Time

	outcomeC := make(chan types.SpecState)
	failureC := make(chan types.Failure)

	go func() {
		finished := false
		defer func() {
			if e := recover(); e != nil || !finished {
				suite.failer.Panic(types.NewCodeLocationWithStackTrace(2), e)
			}

			outcomeFromRun, failureFromRun := suite.failer.Drain()
			failureFromRun.TimelineLocation = suite.generateTimelineLocation()
			outcomeC <- outcomeFromRun
			failureC <- failureFromRun
		}()

		node.Body(sc)
		finished = true
	}()

	// progress polling timer and channel
	var emitProgressNow <-chan time.Time
	var progressPoller *time.Timer
	var pollProgressAfter, pollProgressInterval = suite.config.PollProgressAfter, suite.config.PollProgressInterval
	if node.PollProgressAfter >= 0 {
		pollProgressAfter = node.PollProgressAfter
	}
	if node.PollProgressInterval >= 0 {
		pollProgressInterval = node.PollProgressInterval
	}
	if pollProgressAfter > 0 {
		progressPoller = time.NewTimer(pollProgressAfter)
		emitProgressNow = progressPoller.C
		defer progressPoller.Stop()
	}

	// now we wait for an outcome, an interrupt, a timeout, or a progress poll
	for {
		select {
		case outcomeFromRun := <-outcomeC:
			failureFromRun := <-failureC
			if outcome.Is(types.SpecStateInterrupted | types.SpecStateTimedout) {
				// we've already been interrupted/timed out.  we just managed to actually exit
				// before the grace period elapsed
				// if we have a failure message we attach it as an additional failure
				if outcomeFromRun != types.SpecStatePassed {
					additionalFailure := types.AdditionalFailure{
						State:   outcomeFromRun,
						Failure: failure, //we make a copy - this will include all the configuration set up above...
					}
					//...and then we update the failure with the details from failureFromRun
					additionalFailure.Failure.Location, additionalFailure.Failure.ForwardedPanic, additionalFailure.Failure.TimelineLocation = failureFromRun.Location, failureFromRun.ForwardedPanic, failureFromRun.TimelineLocation
					additionalFailure.Failure.ProgressReport = types.ProgressReport{}
					if outcome == types.SpecStateTimedout {
						additionalFailure.Failure.Message = fmt.Sprintf("A %s timeout occurred and then the following failure was recorded in the timedout node before it exited:\n%s", timeoutInPlay, failureFromRun.Message)
					} else {
						additionalFailure.Failure.Message = fmt.Sprintf("An interrupt occurred and then the following failure was recorded in the interrupted node before it exited:\n%s", failureFromRun.Message)
					}
					suite.reporter.EmitFailure(additionalFailure.State, additionalFailure.Failure)
					failure.AdditionalFailure = &additionalFailure
				}
				return outcome, failure
			}
			if outcomeFromRun.Is(types.SpecStatePassed) {
				return outcomeFromRun, types.Failure{}
			} else {
				failure.Message, failure.Location, failure.ForwardedPanic, failure.TimelineLocation = failureFromRun.Message, failureFromRun.Location, failureFromRun.ForwardedPanic, failureFromRun.TimelineLocation
				suite.reporter.EmitFailure(outcomeFromRun, failure)
				return outcomeFromRun, failure
			}
		case <-gracePeriodChannel:
			if node.HasContext && outcome.Is(types.SpecStateTimedout) {
				report := suite.generateProgressReport(false)
				report.Message = "{{bold}}{{orange}}A running node failed to exit in time{{/}}\nGinkgo is moving on but a node has timed out and failed to exit before its grace period elapsed.  The node has now leaked and is running in the background.\nHere's a current progress report:"
				suite.emitProgressReport(report)
			}
			return outcome, failure
		case <-deadlineChannel:
			// we're out of time - the outcome is a timeout and we capture the failure and progress report
			outcome = types.SpecStateTimedout
			failure.Message, failure.Location, failure.TimelineLocation = fmt.Sprintf("A %s timeout occurred", timeoutInPlay), node.CodeLocation, suite.generateTimelineLocation()
			failure.ProgressReport = suite.generateProgressReport(false).WithoutCapturedGinkgoWriterOutput()
			failure.ProgressReport.Message = fmt.Sprintf("{{bold}}This is the Progress Report generated when the %s timeout occurred:{{/}}", timeoutInPlay)
			deadlineChannel = nil
			suite.reporter.EmitFailure(outcome, failure)

			// tell the spec to stop.  it's important we generate the progress report first to make sure we capture where
			// the spec is actually stuck
			sc.cancel(fmt.Errorf("%s timeout occurred", timeoutInPlay))
			//and now we wait for the grace period
			gracePeriodChannel = time.After(gracePeriod)
		case <-interruptStatus.Channel:
			interruptStatus = suite.interruptHandler.Status()
			// ignore interruption from other process if we are cleaning up or reporting
			if interruptStatus.Cause == interrupt_handler.InterruptCauseAbortByOtherProcess &&
				node.NodeType.Is(types.NodeTypesAllowedDuringReportInterrupt|types.NodeTypesAllowedDuringCleanupInterrupt) {
				continue
			}

			deadlineChannel = nil // don't worry about deadlines, time's up now

			failureTimelineLocation := suite.generateTimelineLocation()
			progressReport := suite.generateProgressReport(true)

			if outcome == types.SpecStateInvalid {
				outcome = types.SpecStateInterrupted
				failure.Message, failure.Location, failure.TimelineLocation = interruptStatus.Message(), node.CodeLocation, failureTimelineLocation
				if interruptStatus.ShouldIncludeProgressReport() {
					failure.ProgressReport = progressReport.WithoutCapturedGinkgoWriterOutput()
					failure.ProgressReport.Message = "{{bold}}This is the Progress Report generated when the interrupt was received:{{/}}"
				}
				suite.reporter.EmitFailure(outcome, failure)
			}

			progressReport = progressReport.WithoutOtherGoroutines()
			sc.cancel(fmt.Errorf(interruptStatus.Message()))

			if interruptStatus.Level == interrupt_handler.InterruptLevelBailOut {
				if interruptStatus.ShouldIncludeProgressReport() {
					progressReport.Message = fmt.Sprintf("{{bold}}{{orange}}%s{{/}}\n{{bold}}{{red}}Final interrupt received{{/}}; Ginkgo will not run any cleanup or reporting nodes and will terminate as soon as possible.\nHere's a current progress report:", interruptStatus.Message())
					suite.emitProgressReport(progressReport)
				}
				return outcome, failure
			}
			if interruptStatus.ShouldIncludeProgressReport() {
				if interruptStatus.Level == interrupt_handler.InterruptLevelCleanupAndReport {
					progressReport.Message = fmt.Sprintf("{{bold}}{{orange}}%s{{/}}\nFirst interrupt received; Ginkgo will run any cleanup and reporting nodes but will skip all remaining specs.  {{bold}}Interrupt again to skip cleanup{{/}}.\nHere's a current progress report:", interruptStatus.Message())
				} else if interruptStatus.Level == interrupt_handler.InterruptLevelReportOnly {
					progressReport.Message = fmt.Sprintf("{{bold}}{{orange}}%s{{/}}\nSecond interrupt received; Ginkgo will run any reporting nodes but will skip all remaining specs and cleanup nodes.  {{bold}}Interrupt again to bail immediately{{/}}.\nHere's a current progress report:", interruptStatus.Message())
				}
				suite.emitProgressReport(progressReport)
			}

			if gracePeriodChannel == nil {
				// we haven't given grace yet... so let's
				gracePeriodChannel = time.After(gracePeriod)
			} else {
				// we've already given grace.  time's up.  now.
				return outcome, failure
			}
		case <-emitProgressNow:
			report := suite.generateProgressReport(false)
			report.Message = "{{bold}}Automatically polling progress:{{/}}"
			suite.emitProgressReport(report)
			if pollProgressInterval > 0 {
				progressPoller.Reset(pollProgressInterval)
			}
		}
	}
}

// TODO: search for usages and consider if reporter.EmitFailure() is necessary
func (suite *Suite) failureForLeafNodeWithMessage(node Node, message string) types.Failure {
	return types.Failure{
		Message:             message,
		Location:            node.CodeLocation,
		TimelineLocation:    suite.generateTimelineLocation(),
		FailureNodeContext:  types.FailureNodeIsLeafNode,
		FailureNodeType:     node.NodeType,
		FailureNodeLocation: node.CodeLocation,
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
