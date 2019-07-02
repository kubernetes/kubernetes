package spec

import (
	"fmt"
	"io"
	"time"

	"sync"

	"github.com/onsi/ginkgo/internal/containernode"
	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/types"
)

type Spec struct {
	subject          leafnodes.SubjectNode
	focused          bool
	announceProgress bool

	containers []*containernode.ContainerNode

	state            types.SpecState
	runTime          time.Duration
	startTime        time.Time
	failure          types.SpecFailure
	previousFailures bool

	stateMutex *sync.Mutex
}

func New(subject leafnodes.SubjectNode, containers []*containernode.ContainerNode, announceProgress bool) *Spec {
	spec := &Spec{
		subject:          subject,
		containers:       containers,
		focused:          subject.Flag() == types.FlagTypeFocused,
		announceProgress: announceProgress,
		stateMutex:       &sync.Mutex{},
	}

	spec.processFlag(subject.Flag())
	for i := len(containers) - 1; i >= 0; i-- {
		spec.processFlag(containers[i].Flag())
	}

	return spec
}

func (spec *Spec) processFlag(flag types.FlagType) {
	if flag == types.FlagTypeFocused {
		spec.focused = true
	} else if flag == types.FlagTypePending {
		spec.setState(types.SpecStatePending)
	}
}

func (spec *Spec) Skip() {
	spec.setState(types.SpecStateSkipped)
}

func (spec *Spec) Failed() bool {
	return spec.getState() == types.SpecStateFailed || spec.getState() == types.SpecStatePanicked || spec.getState() == types.SpecStateTimedOut
}

func (spec *Spec) Passed() bool {
	return spec.getState() == types.SpecStatePassed
}

func (spec *Spec) Flaked() bool {
	return spec.getState() == types.SpecStatePassed && spec.previousFailures
}

func (spec *Spec) Pending() bool {
	return spec.getState() == types.SpecStatePending
}

func (spec *Spec) Skipped() bool {
	return spec.getState() == types.SpecStateSkipped
}

func (spec *Spec) Focused() bool {
	return spec.focused
}

func (spec *Spec) IsMeasurement() bool {
	return spec.subject.Type() == types.SpecComponentTypeMeasure
}

func (spec *Spec) Summary(suiteID string) *types.SpecSummary {
	componentTexts := make([]string, len(spec.containers)+1)
	componentCodeLocations := make([]types.CodeLocation, len(spec.containers)+1)

	for i, container := range spec.containers {
		componentTexts[i] = container.Text()
		componentCodeLocations[i] = container.CodeLocation()
	}

	componentTexts[len(spec.containers)] = spec.subject.Text()
	componentCodeLocations[len(spec.containers)] = spec.subject.CodeLocation()

	runTime := spec.runTime
	if runTime == 0 && !spec.startTime.IsZero() {
		runTime = time.Since(spec.startTime)
	}

	return &types.SpecSummary{
		IsMeasurement:          spec.IsMeasurement(),
		NumberOfSamples:        spec.subject.Samples(),
		ComponentTexts:         componentTexts,
		ComponentCodeLocations: componentCodeLocations,
		State:        spec.getState(),
		RunTime:      runTime,
		Failure:      spec.failure,
		Measurements: spec.measurementsReport(),
		SuiteID:      suiteID,
	}
}

func (spec *Spec) ConcatenatedString() string {
	s := ""
	for _, container := range spec.containers {
		s += container.Text() + " "
	}

	return s + spec.subject.Text()
}

func (spec *Spec) Run(writer io.Writer) {
	if spec.getState() == types.SpecStateFailed {
		spec.previousFailures = true
	}

	spec.startTime = time.Now()
	defer func() {
		spec.runTime = time.Since(spec.startTime)
	}()

	for sample := 0; sample < spec.subject.Samples(); sample++ {
		spec.runSample(sample, writer)

		if spec.getState() != types.SpecStatePassed {
			return
		}
	}
}

func (spec *Spec) getState() types.SpecState {
	spec.stateMutex.Lock()
	defer spec.stateMutex.Unlock()
	return spec.state
}

func (spec *Spec) setState(state types.SpecState) {
	spec.stateMutex.Lock()
	defer spec.stateMutex.Unlock()
	spec.state = state
}

func (spec *Spec) runSample(sample int, writer io.Writer) {
	spec.setState(types.SpecStatePassed)
	spec.failure = types.SpecFailure{}
	innerMostContainerIndexToUnwind := -1

	defer func() {
		for i := innerMostContainerIndexToUnwind; i >= 0; i-- {
			container := spec.containers[i]
			for _, justAfterEach := range container.SetupNodesOfType(types.SpecComponentTypeJustAfterEach) {
				spec.announceSetupNode(writer, "JustAfterEach", container, justAfterEach)
				justAfterEachState, justAfterEachFailure := justAfterEach.Run()
				if justAfterEachState != types.SpecStatePassed && spec.state == types.SpecStatePassed {
					spec.state = justAfterEachState
					spec.failure = justAfterEachFailure
				}
			}
		}

		for i := innerMostContainerIndexToUnwind; i >= 0; i-- {
			container := spec.containers[i]
			for _, afterEach := range container.SetupNodesOfType(types.SpecComponentTypeAfterEach) {
				spec.announceSetupNode(writer, "AfterEach", container, afterEach)
				afterEachState, afterEachFailure := afterEach.Run()
				if afterEachState != types.SpecStatePassed && spec.getState() == types.SpecStatePassed {
					spec.setState(afterEachState)
					spec.failure = afterEachFailure
				}
			}
		}
	}()

	for i, container := range spec.containers {
		innerMostContainerIndexToUnwind = i
		for _, beforeEach := range container.SetupNodesOfType(types.SpecComponentTypeBeforeEach) {
			spec.announceSetupNode(writer, "BeforeEach", container, beforeEach)
			s, f := beforeEach.Run()
			spec.failure = f
			spec.setState(s)
			if spec.getState() != types.SpecStatePassed {
				return
			}
		}
	}

	for _, container := range spec.containers {
		for _, justBeforeEach := range container.SetupNodesOfType(types.SpecComponentTypeJustBeforeEach) {
			spec.announceSetupNode(writer, "JustBeforeEach", container, justBeforeEach)
			s, f := justBeforeEach.Run()
			spec.failure = f
			spec.setState(s)
			if spec.getState() != types.SpecStatePassed {
				return
			}
		}
	}

	spec.announceSubject(writer, spec.subject)
	s, f := spec.subject.Run()
	spec.failure = f
	spec.setState(s)
}

func (spec *Spec) announceSetupNode(writer io.Writer, nodeType string, container *containernode.ContainerNode, setupNode leafnodes.BasicNode) {
	if spec.announceProgress {
		s := fmt.Sprintf("[%s] %s\n  %s\n", nodeType, container.Text(), setupNode.CodeLocation().String())
		writer.Write([]byte(s))
	}
}

func (spec *Spec) announceSubject(writer io.Writer, subject leafnodes.SubjectNode) {
	if spec.announceProgress {
		nodeType := ""
		switch subject.Type() {
		case types.SpecComponentTypeIt:
			nodeType = "It"
		case types.SpecComponentTypeMeasure:
			nodeType = "Measure"
		}
		s := fmt.Sprintf("[%s] %s\n  %s\n", nodeType, subject.Text(), subject.CodeLocation().String())
		writer.Write([]byte(s))
	}
}

func (spec *Spec) measurementsReport() map[string]*types.SpecMeasurement {
	if !spec.IsMeasurement() || spec.Failed() {
		return map[string]*types.SpecMeasurement{}
	}

	return spec.subject.(*leafnodes.MeasureNode).MeasurementsReport()
}
