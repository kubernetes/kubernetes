package godog

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/DATA-DOG/godog/gherkin"
)

const nanoSec = 1000000
const spec = "0.1.0"

func init() {
	Format("events", fmt.Sprintf("Produces JSON event stream, based on spec: %s.", spec), eventsFunc)
}

func eventsFunc(suite string, out io.Writer) Formatter {
	formatter := &events{
		basefmt: basefmt{
			started: timeNowFunc(),
			indent:  2,
			out:     out,
		},
	}

	formatter.event(&struct {
		Event     string `json:"event"`
		Version   string `json:"version"`
		Timestamp int64  `json:"timestamp"`
		Suite     string `json:"suite"`
	}{
		"TestRunStarted",
		spec,
		timeNowFunc().UnixNano() / nanoSec,
		suite,
	})

	return formatter
}

type events struct {
	basefmt

	// currently running feature path, to be part of id.
	// this is sadly not passed by gherkin nodes.
	// it restricts this formatter to run only in synchronous single
	// threaded execution. Unless running a copy of formatter for each feature
	path         string
	stat         stepType // last step status, before skipped
	outlineSteps int      // number of current outline scenario steps
}

func (f *events) event(ev interface{}) {
	data, err := json.Marshal(ev)
	if err != nil {
		panic(fmt.Sprintf("failed to marshal stream event: %+v - %v", ev, err))
	}
	fmt.Fprintln(f.out, string(data))
}

func (f *events) Node(n interface{}) {
	f.basefmt.Node(n)

	var id string
	var undefined bool
	switch t := n.(type) {
	case *gherkin.Scenario:
		id = fmt.Sprintf("%s:%d", f.path, t.Location.Line)
		undefined = len(t.Steps) == 0
	case *gherkin.TableRow:
		id = fmt.Sprintf("%s:%d", f.path, t.Location.Line)
		undefined = f.outlineSteps == 0
	case *gherkin.ScenarioOutline:
		f.outlineSteps = len(t.Steps)
	}

	if len(id) == 0 {
		return
	}

	f.event(&struct {
		Event     string `json:"event"`
		Location  string `json:"location"`
		Timestamp int64  `json:"timestamp"`
	}{
		"TestCaseStarted",
		id,
		timeNowFunc().UnixNano() / nanoSec,
	})

	if undefined {
		// @TODO: is status undefined or passed? when there are no steps
		// for this scenario
		f.event(&struct {
			Event     string `json:"event"`
			Location  string `json:"location"`
			Timestamp int64  `json:"timestamp"`
			Status    string `json:"status"`
		}{
			"TestCaseFinished",
			id,
			timeNowFunc().UnixNano() / nanoSec,
			"undefined",
		})
	}
}

func (f *events) Feature(ft *gherkin.Feature, p string, c []byte) {
	f.basefmt.Feature(ft, p, c)
	f.path = p
	f.event(&struct {
		Event    string `json:"event"`
		Location string `json:"location"`
		Source   string `json:"source"`
	}{
		"TestSource",
		fmt.Sprintf("%s:%d", p, ft.Location.Line),
		string(c),
	})
}

func (f *events) Summary() {
	// @TODO: determine status
	status := passed
	if len(f.failed) > 0 {
		status = failed
	} else if len(f.passed) == 0 {
		if len(f.undefined) > len(f.pending) {
			status = undefined
		} else {
			status = pending
		}
	}

	snips := f.snippets()
	if len(snips) > 0 {
		snips = "You can implement step definitions for undefined steps with these snippets:\n" + snips
	}

	f.event(&struct {
		Event     string `json:"event"`
		Status    string `json:"status"`
		Timestamp int64  `json:"timestamp"`
		Snippets  string `json:"snippets"`
		Memory    string `json:"memory"`
	}{
		"TestRunFinished",
		status.String(),
		timeNowFunc().UnixNano() / nanoSec,
		snips,
		"", // @TODO not sure that could be correctly implemented
	})
}

func (f *events) step(res *stepResult) {
	var errMsg string
	if res.err != nil {
		errMsg = res.err.Error()
	}
	f.event(&struct {
		Event     string `json:"event"`
		Location  string `json:"location"`
		Timestamp int64  `json:"timestamp"`
		Status    string `json:"status"`
		Summary   string `json:"summary,omitempty"`
	}{
		"TestStepFinished",
		fmt.Sprintf("%s:%d", f.path, res.step.Location.Line),
		timeNowFunc().UnixNano() / nanoSec,
		res.typ.String(),
		errMsg,
	})

	// determine if test case has finished
	var finished bool
	var line int
	switch t := f.owner.(type) {
	case *gherkin.TableRow:
		line = t.Location.Line
		finished = f.isLastStep(res.step)
	case *gherkin.Scenario:
		line = t.Location.Line
		finished = f.isLastStep(res.step)
	}

	if finished {
		f.event(&struct {
			Event     string `json:"event"`
			Location  string `json:"location"`
			Timestamp int64  `json:"timestamp"`
			Status    string `json:"status"`
		}{
			"TestCaseFinished",
			fmt.Sprintf("%s:%d", f.path, line),
			timeNowFunc().UnixNano() / nanoSec,
			f.stat.String(),
		})
	}
}

func (f *events) Defined(step *gherkin.Step, def *StepDef) {
	if def != nil {
		m := def.Expr.FindStringSubmatchIndex(step.Text)[2:]
		var args [][2]int
		for i := 0; i < len(m)/2; i++ {
			pair := m[i : i*2+2]
			var idxs [2]int
			idxs[0] = pair[0]
			idxs[1] = pair[1]
			args = append(args, idxs)
		}

		if len(args) == 0 {
			args = make([][2]int, 0)
		}

		f.event(&struct {
			Event    string   `json:"event"`
			Location string   `json:"location"`
			DefID    string   `json:"definition_id"`
			Args     [][2]int `json:"arguments"`
		}{
			"StepDefinitionFound",
			fmt.Sprintf("%s:%d", f.path, step.Location.Line),
			def.definitionID(),
			args,
		})
	}

	f.event(&struct {
		Event     string `json:"event"`
		Location  string `json:"location"`
		Timestamp int64  `json:"timestamp"`
	}{
		"TestStepStarted",
		fmt.Sprintf("%s:%d", f.path, step.Location.Line),
		timeNowFunc().UnixNano() / nanoSec,
	})
}

func (f *events) Passed(step *gherkin.Step, match *StepDef) {
	f.basefmt.Passed(step, match)
	f.stat = passed
	f.step(f.passed[len(f.passed)-1])
}

func (f *events) Skipped(step *gherkin.Step, match *StepDef) {
	f.basefmt.Skipped(step, match)
	f.step(f.skipped[len(f.skipped)-1])
}

func (f *events) Undefined(step *gherkin.Step, match *StepDef) {
	f.basefmt.Undefined(step, match)
	f.stat = undefined
	f.step(f.undefined[len(f.undefined)-1])
}

func (f *events) Failed(step *gherkin.Step, match *StepDef, err error) {
	f.basefmt.Failed(step, match, err)
	f.stat = failed
	f.step(f.failed[len(f.failed)-1])
}

func (f *events) Pending(step *gherkin.Step, match *StepDef) {
	f.stat = pending
	f.basefmt.Pending(step, match)
	f.step(f.pending[len(f.pending)-1])
}
