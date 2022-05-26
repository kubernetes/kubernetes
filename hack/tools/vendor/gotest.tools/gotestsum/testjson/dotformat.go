package testjson

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"golang.org/x/crypto/ssh/terminal"
	"gotest.tools/gotestsum/internal/dotwriter"
	"gotest.tools/gotestsum/log"
)

func dotsFormatV1(event TestEvent, exec *Execution) (string, error) {
	pkg := exec.Package(event.Package)
	switch {
	case event.PackageEvent():
		return "", nil
	case event.Action == ActionRun && pkg.Total == 1:
		return "[" + RelativePackagePath(event.Package) + "]", nil
	}
	return fmtDot(event), nil
}

func fmtDot(event TestEvent) string {
	withColor := colorEvent(event)
	switch event.Action {
	case ActionPass:
		return withColor("·")
	case ActionFail:
		return withColor("✖")
	case ActionSkip:
		return withColor("↷")
	}
	return ""
}

type dotFormatter struct {
	pkgs      map[string]*dotLine
	order     []string
	writer    *dotwriter.Writer
	termWidth int
}

type dotLine struct {
	runes      int
	builder    *strings.Builder
	lastUpdate time.Time
}

func (l *dotLine) update(dot string) {
	if dot == "" {
		return
	}
	l.builder.WriteString(dot)
	l.runes++
}

// checkWidth marks the line as full when the width of the line hits the
// terminal width.
func (l *dotLine) checkWidth(prefix, terminal int) {
	if prefix+l.runes >= terminal {
		l.builder.WriteString("\n" + strings.Repeat(" ", prefix))
		l.runes = 0
	}
}

func newDotFormatter(out io.Writer) EventFormatter {
	w, _, err := terminal.GetSize(int(os.Stdout.Fd()))
	if err != nil || w == 0 {
		log.Warnf("Failed to detect terminal width for dots format, error: %v", err)
		return &formatAdapter{format: dotsFormatV1, out: out}
	}
	return &dotFormatter{
		pkgs:      make(map[string]*dotLine),
		writer:    dotwriter.New(out),
		termWidth: w,
	}
}

func (d *dotFormatter) Format(event TestEvent, exec *Execution) error {
	if d.pkgs[event.Package] == nil {
		d.pkgs[event.Package] = &dotLine{builder: new(strings.Builder)}
		d.order = append(d.order, event.Package)
	}
	line := d.pkgs[event.Package]
	line.lastUpdate = event.Time

	if !event.PackageEvent() {
		line.update(fmtDot(event))
	}
	switch event.Action {
	case ActionOutput, ActionBench:
		return nil
	}

	// Add an empty header to work around incorrect line counting
	fmt.Fprint(d.writer, "\n\n")

	sort.Slice(d.order, d.orderByLastUpdated)
	for _, pkg := range d.order {
		line := d.pkgs[pkg]
		pkgname := RelativePackagePath(pkg) + " "
		prefix := fmtDotElapsed(exec.Package(pkg))
		line.checkWidth(len(prefix+pkgname), d.termWidth)
		fmt.Fprintf(d.writer, prefix+pkgname+line.builder.String()+"\n")
	}
	PrintSummary(d.writer, exec, SummarizeNone)
	return d.writer.Flush()
}

// orderByLastUpdated so that the most recently updated packages move to the
// bottom of the list, leaving completed package in the same order at the top.
func (d *dotFormatter) orderByLastUpdated(i, j int) bool {
	return d.pkgs[d.order[i]].lastUpdate.Before(d.pkgs[d.order[j]].lastUpdate)
}

func fmtDotElapsed(p *Package) string {
	f := func(v string) string {
		return fmt.Sprintf(" %5s ", v)
	}

	elapsed := p.Elapsed()
	switch {
	case p.cached:
		return f("🖴 ")
	case elapsed <= 0:
		return f("")
	case elapsed >= time.Hour:
		return f("⏳ ")
	case elapsed < time.Second:
		return f(elapsed.String())
	}

	const maxWidth = 7
	var steps = []time.Duration{
		time.Millisecond,
		10 * time.Millisecond,
		100 * time.Millisecond,
		time.Second,
		10 * time.Second,
		time.Minute,
		10 * time.Minute,
	}

	for _, trunc := range steps {
		r := f(elapsed.Truncate(trunc).String())
		if len(r) <= maxWidth {
			return r
		}
	}
	return f("")
}
