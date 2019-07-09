package godog

/*
   The specification for the formatting originated from https://www.relishapp.com/cucumber/cucumber/docs/formatters/json-output-formatter.
   I found that documentation was misleading or out dated.  To validate formatting I create a ruby cucumber test harness and ran the
   same feature files through godog and the ruby cucumber.

   The docstrings in the cucumber.feature represent the cucumber output for those same feature definitions.

   I did note that comments in ruby could be at just about any level in particular Feature, Scenario and Step.  In godog I
   could only find comments under the Feature data structure.
*/

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/DATA-DOG/godog/gherkin"
)

func init() {
	Format("cucumber", "Produces cucumber JSON format output.", cucumberFunc)
}

func cucumberFunc(suite string, out io.Writer) Formatter {
	formatter := &cukefmt{
		basefmt: basefmt{
			started: timeNowFunc(),
			indent:  2,
			out:     out,
		},
	}

	return formatter
}

// Replace spaces with - This function is used to create the "id" fields of the cucumber output.
func makeID(name string) string {
	return strings.Replace(strings.ToLower(name), " ", "-", -1)
}

// The sequence of type structs are used to marshall the json object.
type cukeComment struct {
	Value string `json:"value"`
	Line  int    `json:"line"`
}

type cukeDocstring struct {
	Value       string `json:"value"`
	ContentType string `json:"content_type"`
	Line        int    `json:"line"`
}

type cukeTag struct {
	Name string `json:"name"`
	Line int    `json:"line"`
}

type cukeResult struct {
	Status   string `json:"status"`
	Error    string `json:"error_message,omitempty"`
	Duration *int   `json:"duration,omitempty"`
}

type cukeMatch struct {
	Location string `json:"location"`
}

type cukeStep struct {
	Keyword   string              `json:"keyword"`
	Name      string              `json:"name"`
	Line      int                 `json:"line"`
	Docstring *cukeDocstring      `json:"doc_string,omitempty"`
	Match     cukeMatch           `json:"match"`
	Result    cukeResult          `json:"result"`
	DataTable []*cukeDataTableRow `json:"rows,omitempty"`
}

type cukeDataTableRow struct {
	Cells []string `json:"cells"`
}

type cukeElement struct {
	ID          string     `json:"id"`
	Keyword     string     `json:"keyword"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Line        int        `json:"line"`
	Type        string     `json:"type"`
	Tags        []cukeTag  `json:"tags,omitempty"`
	Steps       []cukeStep `json:"steps,omitempty"`
}

type cukeFeatureJSON struct {
	URI         string        `json:"uri"`
	ID          string        `json:"id"`
	Keyword     string        `json:"keyword"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Line        int           `json:"line"`
	Comments    []cukeComment `json:"comments,omitempty"`
	Tags        []cukeTag     `json:"tags,omitempty"`
	Elements    []cukeElement `json:"elements,omitempty"`
}

type cukefmt struct {
	basefmt

	// currently running feature path, to be part of id.
	// this is sadly not passed by gherkin nodes.
	// it restricts this formatter to run only in synchronous single
	// threaded execution. Unless running a copy of formatter for each feature
	path       string
	stat       stepType          // last step status, before skipped
	ID         string            // current test id.
	results    []cukeFeatureJSON // structure that represent cuke results
	curStep    *cukeStep         // track the current step
	curElement *cukeElement      // track the current element
	curFeature *cukeFeatureJSON  // track the current feature
	curOutline cukeElement       // Each example show up as an outline element but the outline is parsed only once
	// so I need to keep track of the current outline
	curRow         int       // current row of the example table as it is being processed.
	curExampleTags []cukeTag // temporary storage for tags associate with the current example table.
	startTime      time.Time // used to time duration of the step execution
	curExampleName string    // Due to the fact that examples are parsed once and then iterated over for each result then we need to keep track
	// of the example name inorder to build id fields.
}

func (f *cukefmt) Node(n interface{}) {
	f.basefmt.Node(n)

	switch t := n.(type) {

	// When the example definition is seen we just need track the id and
	// append the name associated with the example as part of the id.
	case *gherkin.Examples:

		f.curExampleName = makeID(t.Name)
		f.curRow = 2 // there can be more than one example set per outline so reset row count.
		// cucumber counts the header row as an example when creating the id.

		// store any example level tags in a  temp location.
		f.curExampleTags = make([]cukeTag, len(t.Tags))
		for idx, element := range t.Tags {
			f.curExampleTags[idx].Line = element.Location.Line
			f.curExampleTags[idx].Name = element.Name
		}

	// The outline node creates a placeholder and the actual element is added as each TableRow is processed.
	case *gherkin.ScenarioOutline:

		f.curOutline = cukeElement{}
		f.curOutline.Name = t.Name
		f.curOutline.Line = t.Location.Line
		f.curOutline.Description = t.Description
		f.curOutline.Keyword = t.Keyword
		f.curOutline.ID = f.curFeature.ID + ";" + makeID(t.Name)
		f.curOutline.Type = "scenario"
		f.curOutline.Tags = make([]cukeTag, len(t.Tags)+len(f.curFeature.Tags))

		// apply feature level tags
		if len(f.curOutline.Tags) > 0 {
			copy(f.curOutline.Tags, f.curFeature.Tags)

			// apply outline level tags.
			for idx, element := range t.Tags {
				f.curOutline.Tags[idx+len(f.curFeature.Tags)].Line = element.Location.Line
				f.curOutline.Tags[idx+len(f.curFeature.Tags)].Name = element.Name
			}
		}

	// This scenario adds the element to the output immediately.
	case *gherkin.Scenario:
		f.curFeature.Elements = append(f.curFeature.Elements, cukeElement{})
		f.curElement = &f.curFeature.Elements[len(f.curFeature.Elements)-1]

		f.curElement.Name = t.Name
		f.curElement.Line = t.Location.Line
		f.curElement.Description = t.Description
		f.curElement.Keyword = t.Keyword
		f.curElement.ID = f.curFeature.ID + ";" + makeID(t.Name)
		f.curElement.Type = "scenario"
		f.curElement.Tags = make([]cukeTag, len(t.Tags)+len(f.curFeature.Tags))

		if len(f.curElement.Tags) > 0 {
			// apply feature level tags
			copy(f.curElement.Tags, f.curFeature.Tags)

			// apply scenario level tags.
			for idx, element := range t.Tags {
				f.curElement.Tags[idx+len(f.curFeature.Tags)].Line = element.Location.Line
				f.curElement.Tags[idx+len(f.curFeature.Tags)].Name = element.Name
			}
		}

	// This is an outline scenario and the element is added to the output as
	// the TableRows are encountered.
	case *gherkin.TableRow:
		tmpElem := f.curOutline
		tmpElem.Line = t.Location.Line
		tmpElem.ID = tmpElem.ID + ";" + f.curExampleName + ";" + strconv.Itoa(f.curRow)
		f.curRow++
		f.curFeature.Elements = append(f.curFeature.Elements, tmpElem)
		f.curElement = &f.curFeature.Elements[len(f.curFeature.Elements)-1]

		// copy in example level tags.
		f.curElement.Tags = append(f.curElement.Tags, f.curExampleTags...)

	}

}

func (f *cukefmt) Feature(ft *gherkin.Feature, p string, c []byte) {

	f.basefmt.Feature(ft, p, c)
	f.path = p
	f.ID = makeID(ft.Name)
	f.results = append(f.results, cukeFeatureJSON{})

	f.curFeature = &f.results[len(f.results)-1]
	f.curFeature.URI = p
	f.curFeature.Name = ft.Name
	f.curFeature.Keyword = ft.Keyword
	f.curFeature.Line = ft.Location.Line
	f.curFeature.Description = ft.Description
	f.curFeature.ID = f.ID
	f.curFeature.Tags = make([]cukeTag, len(ft.Tags))

	for idx, element := range ft.Tags {
		f.curFeature.Tags[idx].Line = element.Location.Line
		f.curFeature.Tags[idx].Name = element.Name
	}

	f.curFeature.Comments = make([]cukeComment, len(ft.Comments))
	for idx, comment := range ft.Comments {
		f.curFeature.Comments[idx].Value = strings.TrimSpace(comment.Text)
		f.curFeature.Comments[idx].Line = comment.Location.Line
	}

}

func (f *cukefmt) Summary() {
	dat, err := json.MarshalIndent(f.results, "", "    ")
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(f.out, "%s\n", string(dat))
}

func (f *cukefmt) step(res *stepResult) {

	// determine if test case has finished
	switch t := f.owner.(type) {
	case *gherkin.TableRow:
		d := int(timeNowFunc().Sub(f.startTime).Nanoseconds())
		f.curStep.Result.Duration = &d
		f.curStep.Line = t.Location.Line
		f.curStep.Result.Status = res.typ.String()
		if res.err != nil {
			f.curStep.Result.Error = res.err.Error()
		}
	case *gherkin.Scenario:
		d := int(timeNowFunc().Sub(f.startTime).Nanoseconds())
		f.curStep.Result.Duration = &d
		f.curStep.Result.Status = res.typ.String()
		if res.err != nil {
			f.curStep.Result.Error = res.err.Error()
		}
	}
}

func (f *cukefmt) Defined(step *gherkin.Step, def *StepDef) {

	f.startTime = timeNowFunc() // start timing the step
	f.curElement.Steps = append(f.curElement.Steps, cukeStep{})
	f.curStep = &f.curElement.Steps[len(f.curElement.Steps)-1]

	f.curStep.Name = step.Text
	f.curStep.Line = step.Location.Line
	f.curStep.Keyword = step.Keyword

	if _, ok := step.Argument.(*gherkin.DocString); ok {
		f.curStep.Docstring = &cukeDocstring{}
		f.curStep.Docstring.ContentType = strings.TrimSpace(step.Argument.(*gherkin.DocString).ContentType)
		f.curStep.Docstring.Line = step.Argument.(*gherkin.DocString).Location.Line
		f.curStep.Docstring.Value = step.Argument.(*gherkin.DocString).Content
	}

	if _, ok := step.Argument.(*gherkin.DataTable); ok {
		dataTable := step.Argument.(*gherkin.DataTable)

		f.curStep.DataTable = make([]*cukeDataTableRow, len(dataTable.Rows))
		for i, row := range dataTable.Rows {
			cells := make([]string, len(row.Cells))
			for j, cell := range row.Cells {
				cells[j] = cell.Value
			}
			f.curStep.DataTable[i] = &cukeDataTableRow{Cells: cells}
		}
	}

	if def != nil {
		f.curStep.Match.Location = strings.Split(def.definitionID(), " ")[0]
	}
}

func (f *cukefmt) Passed(step *gherkin.Step, match *StepDef) {
	f.basefmt.Passed(step, match)
	f.stat = passed
	f.step(f.passed[len(f.passed)-1])
}

func (f *cukefmt) Skipped(step *gherkin.Step, match *StepDef) {
	f.basefmt.Skipped(step, match)
	f.step(f.skipped[len(f.skipped)-1])

	// no duration reported for skipped.
	f.curStep.Result.Duration = nil
}

func (f *cukefmt) Undefined(step *gherkin.Step, match *StepDef) {
	f.basefmt.Undefined(step, match)
	f.stat = undefined
	f.step(f.undefined[len(f.undefined)-1])

	// the location for undefined is the feature file location not the step file.
	f.curStep.Match.Location = fmt.Sprintf("%s:%d", f.path, step.Location.Line)
	f.curStep.Result.Duration = nil
}

func (f *cukefmt) Failed(step *gherkin.Step, match *StepDef, err error) {
	f.basefmt.Failed(step, match, err)
	f.stat = failed
	f.step(f.failed[len(f.failed)-1])
}

func (f *cukefmt) Pending(step *gherkin.Step, match *StepDef) {
	f.stat = pending
	f.basefmt.Pending(step, match)
	f.step(f.pending[len(f.pending)-1])

	// the location for pending is the feature file location not the step file.
	f.curStep.Match.Location = fmt.Sprintf("%s:%d", f.path, step.Location.Line)
	f.curStep.Result.Duration = nil
}
