package testjson

import (
	"fmt"
	"io"
	"strings"

	"github.com/fatih/color"
)

func debugFormat(event TestEvent, _ *Execution) (string, error) {
	return fmt.Sprintf("%s %s %s (%.3f) [%d] %s\n",
		event.Package,
		event.Test,
		event.Action,
		event.Elapsed,
		event.Time.Unix(),
		event.Output), nil
}

// go test -v
func standardVerboseFormat(event TestEvent, _ *Execution) (string, error) {
	if event.Action == ActionOutput {
		return event.Output, nil
	}
	return "", nil
}

// go test
func standardQuietFormat(event TestEvent, _ *Execution) (string, error) {
	if !event.PackageEvent() {
		return "", nil
	}
	if event.Output == "PASS\n" || isCoverageOutput(event.Output) {
		return "", nil
	}
	if isWarningNoTestsToRunOutput(event.Output) {
		return "", nil
	}

	return event.Output, nil
}

func testNameFormat(event TestEvent, exec *Execution) (string, error) {
	result := colorEvent(event)(strings.ToUpper(string(event.Action)))
	formatTest := func() string {
		pkgPath := RelativePackagePath(event.Package)

		return fmt.Sprintf("%s %s%s %s\n",
			result,
			joinPkgToTestName(pkgPath, event.Test),
			formatRunID(event.RunID),
			event.ElapsedFormatted())
	}

	switch {
	case isPkgFailureOutput(event):
		return event.Output, nil

	case event.PackageEvent():
		if !event.Action.IsTerminal() {
			return "", nil
		}
		pkg := exec.Package(event.Package)
		if event.Action == ActionSkip || (event.Action == ActionPass && pkg.Total == 0) {
			result = colorEvent(event)("EMPTY")
		}

		var cached string
		if pkg.cached {
			cached = cachedMessage
		}
		return fmt.Sprintf("%s %s%s\n",
			result,
			RelativePackagePath(event.Package),
			cached), nil

	case event.Action == ActionFail:
		pkg := exec.Package(event.Package)
		tc := pkg.LastFailedByName(event.Test)
		return pkg.Output(tc.ID) + formatTest(), nil

	case event.Action == ActionPass:
		return formatTest(), nil
	}
	return "", nil
}

// joinPkgToTestName for formatting.
// If the package path isn't the current directory, we add a period to separate
// the test name and the package path. If it is the current directory, we don't
// show it at all. This prevents output like ..MyTest when the test is in the
// current directory.
func joinPkgToTestName(pkg string, test string) string {
	if pkg == "." {
		return test
	}
	return pkg + "." + test
}

// formatRunID returns a formatted string of the runID.
func formatRunID(runID int) string {
	if runID <= 0 {
		return ""
	}
	return fmt.Sprintf(" (re-run %d)", runID)
}

// isPkgFailureOutput returns true if the event is package output, and the output
// doesn't match any of the expected framing messages. Events which match this
// pattern should be package-level failures (ex: exit(1) or panic in an init() or
// TestMain).
func isPkgFailureOutput(event TestEvent) bool {
	out := event.Output
	return all(
		event.PackageEvent(),
		event.Action == ActionOutput,
		out != "PASS\n",
		out != "FAIL\n",
		!isWarningNoTestsToRunOutput(out),
		!strings.HasPrefix(out, "FAIL\t"+event.Package),
		!strings.HasPrefix(out, "ok  \t"+event.Package),
		!strings.HasPrefix(out, "?   \t"+event.Package),
	)
}

func all(cond ...bool) bool {
	for _, c := range cond {
		if !c {
			return false
		}
	}
	return true
}

const cachedMessage = " (cached)"

func pkgNameFormat(event TestEvent, exec *Execution) (string, error) {
	if !event.PackageEvent() {
		return "", nil
	}
	return shortFormatPackageEvent(event, exec)
}

func shortFormatPackageEvent(event TestEvent, exec *Execution) (string, error) {
	pkg := exec.Package(event.Package)

	fmtElapsed := func() string {
		if pkg.cached {
			return cachedMessage
		}
		d := elapsedDuration(event.Elapsed)
		if d == 0 {
			return ""
		}
		return fmt.Sprintf(" (%s)", d)
	}
	fmtCoverage := func() string {
		if pkg.coverage == "" {
			return ""
		}
		return " (" + pkg.coverage + ")"
	}
	fmtEvent := func(action string) (string, error) {
		return fmt.Sprintf("%s  %s%s%s\n",
			action,
			RelativePackagePath(event.Package),
			fmtElapsed(),
			fmtCoverage(),
		), nil
	}
	withColor := colorEvent(event)
	switch event.Action {
	case ActionSkip:
		return fmtEvent(withColor("∅"))
	case ActionPass:
		if pkg.Total == 0 {
			return fmtEvent(withColor("∅"))
		}
		return fmtEvent(withColor("✓"))
	case ActionFail:
		return fmtEvent(withColor("✖"))
	}
	return "", nil
}

func pkgNameWithFailuresFormat(event TestEvent, exec *Execution) (string, error) {
	if !event.PackageEvent() {
		if event.Action == ActionFail {
			pkg := exec.Package(event.Package)
			tc := pkg.LastFailedByName(event.Test)
			return pkg.Output(tc.ID), nil
		}
		return "", nil
	}
	return shortFormatPackageEvent(event, exec)
}

func colorEvent(event TestEvent) func(format string, a ...interface{}) string {
	switch event.Action {
	case ActionPass:
		return color.GreenString
	case ActionFail:
		return color.RedString
	case ActionSkip:
		return color.YellowString
	}
	return color.WhiteString
}

// EventFormatter is a function which handles an event and returns a string to
// output for the event.
type EventFormatter interface {
	Format(event TestEvent, output *Execution) error
}

// NewEventFormatter returns a formatter for printing events.
func NewEventFormatter(out io.Writer, format string) EventFormatter {
	switch format {
	case "debug":
		return &formatAdapter{out, debugFormat}
	case "standard-verbose":
		return &formatAdapter{out, standardVerboseFormat}
	case "standard-quiet":
		return &formatAdapter{out, standardQuietFormat}
	case "dots", "dots-v1":
		return &formatAdapter{out, dotsFormatV1}
	case "dots-v2":
		return newDotFormatter(out)
	case "testname", "short-verbose":
		return &formatAdapter{out, testNameFormat}
	case "pkgname", "short":
		return &formatAdapter{out, pkgNameFormat}
	case "pkgname-and-test-fails", "short-with-failures":
		return &formatAdapter{out, pkgNameWithFailuresFormat}
	default:
		return nil
	}
}

type formatAdapter struct {
	out    io.Writer
	format func(TestEvent, *Execution) (string, error)
}

func (f *formatAdapter) Format(event TestEvent, exec *Execution) error {
	o, err := f.format(event, exec)
	if err != nil {
		return err
	}
	_, err = f.out.Write([]byte(o))
	return err
}
