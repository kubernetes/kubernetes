package godog

import (
	"fmt"
	"io"
	"math"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/DATA-DOG/godog/colors"
	"github.com/DATA-DOG/godog/gherkin"
)

func init() {
	Format("pretty", "Prints every feature with runtime statuses.", prettyFunc)
}

func prettyFunc(suite string, out io.Writer) Formatter {
	return &pretty{
		basefmt: basefmt{
			started: timeNowFunc(),
			indent:  2,
			out:     out,
		},
	}
}

var outlinePlaceholderRegexp = regexp.MustCompile("<[^>]+>")

// a built in default pretty formatter
type pretty struct {
	basefmt

	// currently processed
	feature  *gherkin.Feature
	scenario *gherkin.Scenario
	outline  *gherkin.ScenarioOutline

	// state
	bgSteps      int
	totalBgSteps int
	steps        int
	commentPos   int

	// whether scenario or scenario outline keyword was printed
	scenarioKeyword bool

	// outline
	outlineSteps       []*stepResult
	outlineNumExample  int
	outlineNumExamples int
}

func (f *pretty) Feature(ft *gherkin.Feature, p string, c []byte) {
	if len(f.features) != 0 {
		// not a first feature, add a newline
		fmt.Fprintln(f.out, "")
	}
	f.features = append(f.features, &feature{Path: p, Feature: ft})
	fmt.Fprintln(f.out, keywordAndName(ft.Keyword, ft.Name))
	if strings.TrimSpace(ft.Description) != "" {
		for _, line := range strings.Split(ft.Description, "\n") {
			fmt.Fprintln(f.out, s(f.indent)+strings.TrimSpace(line))
		}
	}

	f.feature = ft
	f.scenario = nil
	f.outline = nil
	f.bgSteps = 0
	f.totalBgSteps = 0
	if ft.Background != nil {
		f.bgSteps = len(ft.Background.Steps)
		f.totalBgSteps = len(ft.Background.Steps)
	}
}

// Node takes a gherkin node for formatting
func (f *pretty) Node(node interface{}) {
	f.basefmt.Node(node)

	switch t := node.(type) {
	case *gherkin.Examples:
		f.outlineNumExamples = len(t.TableBody)
		f.outlineNumExample++
	case *gherkin.Scenario:
		f.scenario = t
		f.outline = nil
		f.steps = len(t.Steps) + f.totalBgSteps
		f.scenarioKeyword = false
		if isEmptyScenario(t) {
			f.printUndefinedScenario(t)
		}
	case *gherkin.ScenarioOutline:
		f.outline = t
		f.scenario = nil
		f.outlineNumExample = -1
		f.scenarioKeyword = false
		if isEmptyScenario(t) {
			f.printUndefinedScenario(t)
		}
	case *gherkin.TableRow:
		f.steps = len(f.outline.Steps) + f.totalBgSteps
		f.outlineSteps = []*stepResult{}
	}
}

func keywordAndName(keyword, name string) string {
	title := whiteb(keyword + ":")
	if len(name) > 0 {
		title += " " + name
	}
	return title
}

func (f *pretty) printUndefinedScenario(sc interface{}) {
	if f.bgSteps > 0 {
		bg := f.feature.Background
		f.commentPos = f.longestStep(bg.Steps, f.length(bg))
		fmt.Fprintln(f.out, "\n"+s(f.indent)+keywordAndName(bg.Keyword, bg.Name))

		for _, step := range bg.Steps {
			f.bgSteps--
			f.printStep(step, nil, colors.Cyan)
		}
	}

	switch t := sc.(type) {
	case *gherkin.Scenario:
		f.commentPos = f.longestStep(t.Steps, f.length(sc))
		text := s(f.indent) + keywordAndName(t.Keyword, t.Name)
		text += s(f.commentPos-f.length(t)+1) + f.line(t.Location)
		fmt.Fprintln(f.out, "\n"+text)
	case *gherkin.ScenarioOutline:
		f.commentPos = f.longestStep(t.Steps, f.length(sc))
		text := s(f.indent) + keywordAndName(t.Keyword, t.Name)
		text += s(f.commentPos-f.length(t)+1) + f.line(t.Location)
		fmt.Fprintln(f.out, "\n"+text)

		for _, example := range t.Examples {
			max := longest(example, cyan)
			f.printExampleHeader(example, max)
			for _, row := range example.TableBody {
				f.printExampleRow(row, max, cyan)
			}
		}
	}
}

// Summary sumarize the feature formatter output
func (f *pretty) Summary() {
	if len(f.failed) > 0 {
		fmt.Fprintln(f.out, "\n--- "+red("Failed steps:")+"\n")
		for _, fail := range f.failed {
			fmt.Fprintln(f.out, s(2)+red(fail.scenarioDesc())+black(" # "+fail.scenarioLine()))
			fmt.Fprintln(f.out, s(4)+red(strings.TrimSpace(fail.step.Keyword)+" "+fail.step.Text)+black(" # "+fail.line()))
			fmt.Fprintln(f.out, s(6)+red("Error: ")+redb(fmt.Sprintf("%+v", fail.err))+"\n")
		}
	}
	f.basefmt.Summary()
}

func (f *pretty) printOutlineExample(outline *gherkin.ScenarioOutline) {
	var msg string
	var clr colors.ColorFunc

	ex := outline.Examples[f.outlineNumExample]
	example, hasExamples := examples(ex)
	if !hasExamples {
		// do not print empty examples
		return
	}

	firstExample := f.outlineNumExamples == len(example.TableBody)
	printSteps := firstExample && f.outlineNumExample == 0

	for i, res := range f.outlineSteps {
		// determine example row status
		switch {
		case res.typ == failed:
			msg = res.err.Error()
			clr = res.typ.clr()
		case res.typ == undefined || res.typ == pending:
			clr = res.typ.clr()
		case res.typ == skipped && clr == nil:
			clr = cyan
		}
		if printSteps && i >= f.totalBgSteps {
			// in first example, we need to print steps
			var text string
			ostep := outline.Steps[i-f.totalBgSteps]
			if res.def != nil {
				if m := outlinePlaceholderRegexp.FindAllStringIndex(ostep.Text, -1); len(m) > 0 {
					var pos int
					for i := 0; i < len(m); i++ {
						pair := m[i]
						text += cyan(ostep.Text[pos:pair[0]])
						text += cyanb(ostep.Text[pair[0]:pair[1]])
						pos = pair[1]
					}
					text += cyan(ostep.Text[pos:len(ostep.Text)])
				} else {
					text = cyan(ostep.Text)
				}
				text += s(f.commentPos-f.length(ostep)+1) + black(fmt.Sprintf("# %s", res.def.definitionID()))
			} else {
				text = cyan(ostep.Text)
			}
			// print the step outline
			fmt.Fprintln(f.out, s(f.indent*2)+cyan(strings.TrimSpace(ostep.Keyword))+" "+text)

			// print step argument
			// @TODO: need to make example header cells bold
			switch t := ostep.Argument.(type) {
			case *gherkin.DataTable:
				f.printTable(t, cyan)
			case *gherkin.DocString:
				var ct string
				if len(t.ContentType) > 0 {
					ct = " " + cyan(t.ContentType)
				}
				fmt.Fprintln(f.out, s(f.indent*3)+cyan(t.Delimitter)+ct)
				for _, ln := range strings.Split(t.Content, "\n") {
					fmt.Fprintln(f.out, s(f.indent*3)+cyan(ln))
				}
				fmt.Fprintln(f.out, s(f.indent*3)+cyan(t.Delimitter))
			}
		}
	}

	if clr == nil {
		clr = green
	}

	max := longest(example, clr, cyan)
	// an example table header
	if firstExample {
		f.printExampleHeader(example, max)
	}

	// an example table row
	row := example.TableBody[len(example.TableBody)-f.outlineNumExamples]
	f.printExampleRow(row, max, clr)

	// if there is an error
	if msg != "" {
		fmt.Fprintln(f.out, s(f.indent*4)+redb(msg))
	}
}

func (f *pretty) printExampleRow(row *gherkin.TableRow, max []int, clr colors.ColorFunc) {
	cells := make([]string, len(row.Cells))
	for i, cell := range row.Cells {
		val := clr(cell.Value)
		ln := utf8.RuneCountInString(val)
		cells[i] = val + s(max[i]-ln)
	}
	fmt.Fprintln(f.out, s(f.indent*3)+"| "+strings.Join(cells, " | ")+" |")
}

func (f *pretty) printExampleHeader(example *gherkin.Examples, max []int) {
	cells := make([]string, len(example.TableHeader.Cells))
	// an example table header
	fmt.Fprintln(f.out, "")
	fmt.Fprintln(f.out, s(f.indent*2)+keywordAndName(example.Keyword, example.Name))

	for i, cell := range example.TableHeader.Cells {
		val := cyan(cell.Value)
		ln := utf8.RuneCountInString(val)
		cells[i] = val + s(max[i]-ln)
	}
	fmt.Fprintln(f.out, s(f.indent*3)+"| "+strings.Join(cells, " | ")+" |")
}

func (f *pretty) printStep(step *gherkin.Step, def *StepDef, c colors.ColorFunc) {
	text := s(f.indent*2) + c(strings.TrimSpace(step.Keyword)) + " "
	switch {
	case def != nil:
		if m := def.Expr.FindStringSubmatchIndex(step.Text)[2:]; len(m) > 0 {
			var pos, i int
			for pos, i = 0, 0; i < len(m); i++ {
				if m[i] == -1 {
					continue // no index for this match
				}
				if math.Mod(float64(i), 2) == 0 {
					text += c(step.Text[pos:m[i]])
				} else {
					text += colors.Bold(c)(step.Text[pos:m[i]])
				}
				pos = m[i]
			}
			text += c(step.Text[pos:len(step.Text)])
		} else {
			text += c(step.Text)
		}
		text += s(f.commentPos-f.length(step)+1) + black(fmt.Sprintf("# %s", def.definitionID()))
	default:
		text += c(step.Text)
	}

	fmt.Fprintln(f.out, text)
	switch t := step.Argument.(type) {
	case *gherkin.DataTable:
		f.printTable(t, c)
	case *gherkin.DocString:
		var ct string
		if len(t.ContentType) > 0 {
			ct = " " + c(t.ContentType)
		}
		fmt.Fprintln(f.out, s(f.indent*3)+c(t.Delimitter)+ct)
		for _, ln := range strings.Split(t.Content, "\n") {
			fmt.Fprintln(f.out, s(f.indent*3)+c(ln))
		}
		fmt.Fprintln(f.out, s(f.indent*3)+c(t.Delimitter))
	}
}

func (f *pretty) printStepKind(res *stepResult) {
	f.steps--
	if f.outline != nil {
		f.outlineSteps = append(f.outlineSteps, res)
	}
	var bgStep bool
	bg := f.feature.Background

	// if has not printed background yet
	switch {
	// first background step
	case f.bgSteps > 0 && f.bgSteps == len(bg.Steps):
		f.commentPos = f.longestStep(bg.Steps, f.length(bg))
		fmt.Fprintln(f.out, "\n"+s(f.indent)+keywordAndName(bg.Keyword, bg.Name))
		f.bgSteps--
		bgStep = true
	// subsequent background steps
	case f.bgSteps > 0:
		f.bgSteps--
		bgStep = true
	// first step of scenario, print header and calculate comment position
	case f.scenario != nil:
		// print scenario keyword and value if first example
		if !f.scenarioKeyword {
			f.commentPos = f.longestStep(f.scenario.Steps, f.length(f.scenario))
			if bg != nil {
				if bgLen := f.longestStep(bg.Steps, f.length(bg)); bgLen > f.commentPos {
					f.commentPos = bgLen
				}
			}
			text := s(f.indent) + keywordAndName(f.scenario.Keyword, f.scenario.Name)
			text += s(f.commentPos-f.length(f.scenario)+1) + f.line(f.scenario.Location)
			fmt.Fprintln(f.out, "\n"+text)
			f.scenarioKeyword = true
		}
	// first step of outline scenario, print header and calculate comment position
	case f.outline != nil:
		// print scenario keyword and value if first example
		if !f.scenarioKeyword {
			f.commentPos = f.longestStep(f.outline.Steps, f.length(f.outline))
			if bg != nil {
				if bgLen := f.longestStep(bg.Steps, f.length(bg)); bgLen > f.commentPos {
					f.commentPos = bgLen
				}
			}
			text := s(f.indent) + keywordAndName(f.outline.Keyword, f.outline.Name)
			text += s(f.commentPos-f.length(f.outline)+1) + f.line(f.outline.Location)
			fmt.Fprintln(f.out, "\n"+text)
			f.scenarioKeyword = true
		}
		if len(f.outlineSteps) == len(f.outline.Steps)+f.totalBgSteps {
			// an outline example steps has went through
			f.printOutlineExample(f.outline)
			f.outlineNumExamples--
		}
		return
	}

	if !f.isBackgroundStep(res.step) || bgStep {
		f.printStep(res.step, res.def, res.typ.clr())
	}
	if res.err != nil {
		fmt.Fprintln(f.out, s(f.indent*2)+redb(fmt.Sprintf("%+v", res.err)))
	}
	if res.typ == pending {
		fmt.Fprintln(f.out, s(f.indent*3)+yellow("TODO: write pending definition"))
	}
}

func (f *pretty) isBackgroundStep(step *gherkin.Step) bool {
	if f.feature.Background == nil {
		return false
	}

	for _, bstep := range f.feature.Background.Steps {
		if bstep.Location.Line == step.Location.Line {
			return true
		}
	}
	return false
}

// print table with aligned table cells
func (f *pretty) printTable(t *gherkin.DataTable, c colors.ColorFunc) {
	var l = longest(t, c)
	var cols = make([]string, len(t.Rows[0].Cells))
	for _, row := range t.Rows {
		for i, cell := range row.Cells {
			val := c(cell.Value)
			ln := utf8.RuneCountInString(val)
			cols[i] = val + s(l[i]-ln)
		}
		fmt.Fprintln(f.out, s(f.indent*3)+"| "+strings.Join(cols, " | ")+" |")
	}
}

func (f *pretty) Passed(step *gherkin.Step, match *StepDef) {
	f.basefmt.Passed(step, match)
	f.printStepKind(f.passed[len(f.passed)-1])
}

func (f *pretty) Skipped(step *gherkin.Step, match *StepDef) {
	f.basefmt.Skipped(step, match)
	f.printStepKind(f.skipped[len(f.skipped)-1])
}

func (f *pretty) Undefined(step *gherkin.Step, match *StepDef) {
	f.basefmt.Undefined(step, match)
	f.printStepKind(f.undefined[len(f.undefined)-1])
}

func (f *pretty) Failed(step *gherkin.Step, match *StepDef, err error) {
	f.basefmt.Failed(step, match, err)
	f.printStepKind(f.failed[len(f.failed)-1])
}

func (f *pretty) Pending(step *gherkin.Step, match *StepDef) {
	f.basefmt.Pending(step, match)
	f.printStepKind(f.pending[len(f.pending)-1])
}

// longest gives a list of longest columns of all rows in Table
func longest(tbl interface{}, clrs ...colors.ColorFunc) []int {
	var rows []*gherkin.TableRow
	switch t := tbl.(type) {
	case *gherkin.Examples:
		rows = append(rows, t.TableHeader)
		rows = append(rows, t.TableBody...)
	case *gherkin.DataTable:
		rows = append(rows, t.Rows...)
	}

	longest := make([]int, len(rows[0].Cells))
	for _, row := range rows {
		for i, cell := range row.Cells {
			for _, c := range clrs {
				ln := utf8.RuneCountInString(c(cell.Value))
				if longest[i] < ln {
					longest[i] = ln
				}
			}

			ln := utf8.RuneCountInString(cell.Value)
			if longest[i] < ln {
				longest[i] = ln
			}
		}
	}
	return longest
}

func (f *pretty) longestStep(steps []*gherkin.Step, base int) int {
	ret := base
	for _, step := range steps {
		length := f.length(step)
		if length > ret {
			ret = length
		}
	}
	return ret
}

// a line number representation in feature file
func (f *pretty) line(loc *gherkin.Location) string {
	return black(fmt.Sprintf("# %s:%d", f.features[len(f.features)-1].Path, loc.Line))
}

func (f *pretty) length(node interface{}) int {
	switch t := node.(type) {
	case *gherkin.Background:
		return f.indent + utf8.RuneCountInString(strings.TrimSpace(t.Keyword)+": "+t.Name)
	case *gherkin.Step:
		return f.indent*2 + utf8.RuneCountInString(strings.TrimSpace(t.Keyword)+" "+t.Text)
	case *gherkin.Scenario:
		return f.indent + utf8.RuneCountInString(strings.TrimSpace(t.Keyword)+": "+t.Name)
	case *gherkin.ScenarioOutline:
		return f.indent + utf8.RuneCountInString(strings.TrimSpace(t.Keyword)+": "+t.Name)
	}
	panic(fmt.Sprintf("unexpected node %T to determine length", node))
}
