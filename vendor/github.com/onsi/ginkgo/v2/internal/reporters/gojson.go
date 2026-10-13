package reporters

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2/types"
	"golang.org/x/tools/go/packages"
)

func ptr[T any](in T) *T {
	return &in
}

type encoder interface {
	Encode(v any) error
}

// gojsonEvent matches the format from go internals
// https://github.com/golang/go/blob/master/src/cmd/internal/test2json/test2json.go#L31-L41
// https://pkg.go.dev/cmd/test2json
type gojsonEvent struct {
	Time        *time.Time `json:",omitempty"`
	Action      GoJSONAction
	Package     string   `json:",omitempty"`
	Test        string   `json:",omitempty"`
	Elapsed     *float64 `json:",omitempty"`
	Output      *string  `json:",omitempty"`
	FailedBuild string   `json:",omitempty"`
}

type GoJSONAction string

const (
	// start  - the test binary is about to be executed
	GoJSONStart GoJSONAction = "start"
	// run    - the test has started running
	GoJSONRun GoJSONAction = "run"
	// pause  - the test has been paused
	GoJSONPause GoJSONAction = "pause"
	// cont   - the test has continued running
	GoJSONCont GoJSONAction = "cont"
	// pass   - the test passed
	GoJSONPass GoJSONAction = "pass"
	// bench  - the benchmark printed log output but did not fail
	GoJSONBench GoJSONAction = "bench"
	// fail   - the test or benchmark failed
	GoJSONFail GoJSONAction = "fail"
	// output - the test printed output
	GoJSONOutput GoJSONAction = "output"
	// skip   - the test was skipped or the package contained no tests
	GoJSONSkip GoJSONAction = "skip"
)

func goJSONActionFromSpecState(state types.SpecState) GoJSONAction {
	switch state {
	case types.SpecStateInvalid:
		return GoJSONFail
	case types.SpecStatePending:
		return GoJSONSkip
	case types.SpecStateSkipped:
		return GoJSONSkip
	case types.SpecStatePassed:
		return GoJSONPass
	case types.SpecStateFailed:
		return GoJSONFail
	case types.SpecStateAborted:
		return GoJSONFail
	case types.SpecStatePanicked:
		return GoJSONFail
	case types.SpecStateInterrupted:
		return GoJSONFail
	case types.SpecStateTimedout:
		return GoJSONFail
	default:
		panic("unexpected state should not happen")
	}
}

// gojsonReport wraps types.Report and calcualtes extra fields requires by gojson
type gojsonReport struct {
	o types.Report
	// Extra calculated fields
	goPkg   string
	elapsed float64
}

func newReport(in types.Report) *gojsonReport {
	return &gojsonReport{
		o: in,
	}
}

func (r *gojsonReport) Fill() error {
	// NOTE: could the types.Report include the go package name?
	goPkg, err := suitePathToPkg(r.o.SuitePath)
	if err != nil {
		return err
	}
	r.goPkg = goPkg
	r.elapsed = r.o.RunTime.Seconds()
	return nil
}

// gojsonSpecReport wraps types.SpecReport and calculates extra fields required by gojson
type gojsonSpecReport struct {
	o types.SpecReport
	// extra calculated fields
	testName string
	elapsed  float64
	action   GoJSONAction
}

func newSpecReport(in types.SpecReport) *gojsonSpecReport {
	return &gojsonSpecReport{
		o: in,
	}
}

func (sr *gojsonSpecReport) Fill() error {
	sr.elapsed = sr.o.RunTime.Seconds()
	sr.testName = createTestName(sr.o)
	sr.action = goJSONActionFromSpecState(sr.o.State)
	return nil
}

func suitePathToPkg(dir string) (string, error) {
	cfg := &packages.Config{
		Mode: packages.NeedFiles | packages.NeedSyntax,
	}
	pkgs, err := packages.Load(cfg, dir)
	if err != nil {
		return "", err
	}
	if len(pkgs) != 1 {
		return "", errors.New("error")
	}
	return pkgs[0].ID, nil
}

func createTestName(spec types.SpecReport) string {
	name := fmt.Sprintf("[%s]", spec.LeafNodeType)
	if spec.FullText() != "" {
		name = name + " " + spec.FullText()
	}
	labels := spec.Labels()
	if len(labels) > 0 {
		name = name + " [" + strings.Join(labels, ", ") + "]"
	}
	semVerConstraints := spec.SemVerConstraints()
	if len(semVerConstraints) > 0 {
		name = name + " [" + strings.Join(semVerConstraints, ", ") + "]"
	}
	componentSemVerConstraints := spec.ComponentSemVerConstraints()
	if len(componentSemVerConstraints) > 0 {
		name = name + " [" + formatComponentSemVerConstraintsToString(componentSemVerConstraints) + "]"
	}
	name = strings.TrimSpace(name)
	return name
}

func formatComponentSemVerConstraintsToString(componentSemVerConstraints map[string][]string) string {
	var tmpStr string
	for component, semVerConstraints := range componentSemVerConstraints {
		tmpStr = tmpStr + fmt.Sprintf("%s: %s, ", component, semVerConstraints)
	}
	tmpStr = strings.TrimSuffix(tmpStr, ", ")
	return tmpStr
}
