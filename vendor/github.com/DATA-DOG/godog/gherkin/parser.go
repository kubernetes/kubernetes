//
// This file is generated. Do not edit! Edit gherkin-golang.razor instead.

//
package gherkin

import (
	"fmt"
	"strings"
)

type TokenType int

const (
	TokenType_None TokenType = iota
	TokenType_EOF
	TokenType_Empty
	TokenType_Comment
	TokenType_TagLine
	TokenType_FeatureLine
	TokenType_BackgroundLine
	TokenType_ScenarioLine
	TokenType_ScenarioOutlineLine
	TokenType_ExamplesLine
	TokenType_StepLine
	TokenType_DocStringSeparator
	TokenType_TableRow
	TokenType_Language
	TokenType_Other
)

func tokenTypeForRule(rt RuleType) TokenType {
	return TokenType_None
}

func (t TokenType) Name() string {
	switch t {
	case TokenType_EOF:
		return "EOF"
	case TokenType_Empty:
		return "Empty"
	case TokenType_Comment:
		return "Comment"
	case TokenType_TagLine:
		return "TagLine"
	case TokenType_FeatureLine:
		return "FeatureLine"
	case TokenType_BackgroundLine:
		return "BackgroundLine"
	case TokenType_ScenarioLine:
		return "ScenarioLine"
	case TokenType_ScenarioOutlineLine:
		return "ScenarioOutlineLine"
	case TokenType_ExamplesLine:
		return "ExamplesLine"
	case TokenType_StepLine:
		return "StepLine"
	case TokenType_DocStringSeparator:
		return "DocStringSeparator"
	case TokenType_TableRow:
		return "TableRow"
	case TokenType_Language:
		return "Language"
	case TokenType_Other:
		return "Other"
	}
	return ""
}

func (t TokenType) RuleType() RuleType {
	switch t {
	case TokenType_EOF:
		return RuleType__EOF
	case TokenType_Empty:
		return RuleType__Empty
	case TokenType_Comment:
		return RuleType__Comment
	case TokenType_TagLine:
		return RuleType__TagLine
	case TokenType_FeatureLine:
		return RuleType__FeatureLine
	case TokenType_BackgroundLine:
		return RuleType__BackgroundLine
	case TokenType_ScenarioLine:
		return RuleType__ScenarioLine
	case TokenType_ScenarioOutlineLine:
		return RuleType__ScenarioOutlineLine
	case TokenType_ExamplesLine:
		return RuleType__ExamplesLine
	case TokenType_StepLine:
		return RuleType__StepLine
	case TokenType_DocStringSeparator:
		return RuleType__DocStringSeparator
	case TokenType_TableRow:
		return RuleType__TableRow
	case TokenType_Language:
		return RuleType__Language
	case TokenType_Other:
		return RuleType__Other
	}
	return RuleType_None
}

type RuleType int

const (
	RuleType_None RuleType = iota

	RuleType__EOF
	RuleType__Empty
	RuleType__Comment
	RuleType__TagLine
	RuleType__FeatureLine
	RuleType__BackgroundLine
	RuleType__ScenarioLine
	RuleType__ScenarioOutlineLine
	RuleType__ExamplesLine
	RuleType__StepLine
	RuleType__DocStringSeparator
	RuleType__TableRow
	RuleType__Language
	RuleType__Other
	RuleType_Feature
	RuleType_Feature_Header
	RuleType_Background
	RuleType_Scenario_Definition
	RuleType_Scenario
	RuleType_ScenarioOutline
	RuleType_Examples_Definition
	RuleType_Examples
	RuleType_Scenario_Step
	RuleType_ScenarioOutline_Step
	RuleType_Step
	RuleType_Step_Arg
	RuleType_DataTable
	RuleType_DocString
	RuleType_Tags
	RuleType_Feature_Description
	RuleType_Background_Description
	RuleType_Scenario_Description
	RuleType_ScenarioOutline_Description
	RuleType_Examples_Description
	RuleType_Description_Helper
	RuleType_Description
)

func (t RuleType) IsEOF() bool {
	return t == RuleType__EOF
}
func (t RuleType) Name() string {
	switch t {
	case RuleType__EOF:
		return "#EOF"
	case RuleType__Empty:
		return "#Empty"
	case RuleType__Comment:
		return "#Comment"
	case RuleType__TagLine:
		return "#TagLine"
	case RuleType__FeatureLine:
		return "#FeatureLine"
	case RuleType__BackgroundLine:
		return "#BackgroundLine"
	case RuleType__ScenarioLine:
		return "#ScenarioLine"
	case RuleType__ScenarioOutlineLine:
		return "#ScenarioOutlineLine"
	case RuleType__ExamplesLine:
		return "#ExamplesLine"
	case RuleType__StepLine:
		return "#StepLine"
	case RuleType__DocStringSeparator:
		return "#DocStringSeparator"
	case RuleType__TableRow:
		return "#TableRow"
	case RuleType__Language:
		return "#Language"
	case RuleType__Other:
		return "#Other"
	case RuleType_Feature:
		return "Feature"
	case RuleType_Feature_Header:
		return "Feature_Header"
	case RuleType_Background:
		return "Background"
	case RuleType_Scenario_Definition:
		return "Scenario_Definition"
	case RuleType_Scenario:
		return "Scenario"
	case RuleType_ScenarioOutline:
		return "ScenarioOutline"
	case RuleType_Examples_Definition:
		return "Examples_Definition"
	case RuleType_Examples:
		return "Examples"
	case RuleType_Scenario_Step:
		return "Scenario_Step"
	case RuleType_ScenarioOutline_Step:
		return "ScenarioOutline_Step"
	case RuleType_Step:
		return "Step"
	case RuleType_Step_Arg:
		return "Step_Arg"
	case RuleType_DataTable:
		return "DataTable"
	case RuleType_DocString:
		return "DocString"
	case RuleType_Tags:
		return "Tags"
	case RuleType_Feature_Description:
		return "Feature_Description"
	case RuleType_Background_Description:
		return "Background_Description"
	case RuleType_Scenario_Description:
		return "Scenario_Description"
	case RuleType_ScenarioOutline_Description:
		return "ScenarioOutline_Description"
	case RuleType_Examples_Description:
		return "Examples_Description"
	case RuleType_Description_Helper:
		return "Description_Helper"
	case RuleType_Description:
		return "Description"
	}
	return ""
}

type parseError struct {
	msg string
	loc *Location
}

func (a *parseError) Error() string {
	return fmt.Sprintf("(%d:%d): %s", a.loc.Line, a.loc.Column, a.msg)
}

type parseErrors []error

func (pe parseErrors) Error() string {
	var ret = []string{"Parser errors:"}
	for i := range pe {
		ret = append(ret, pe[i].Error())
	}
	return strings.Join(ret, "\n")
}

func (p *parser) Parse(s Scanner, m Matcher) (err error) {
	p.builder.Reset()
	m.Reset()
	ctxt := &parseContext{p, s, p.builder, m, nil, nil}
	var state int
	ctxt.startRule(RuleType_Feature)
	for {
		gl, eof, err := ctxt.scan()
		if err != nil {
			ctxt.addError(err)
			if p.stopAtFirstError {
				break
			}
		}
		state, err = ctxt.match(state, gl)
		if err != nil {
			ctxt.addError(err)
			if p.stopAtFirstError {
				break
			}
		}
		if eof {
			// done! \o/
			break
		}
	}
	ctxt.endRule(RuleType_Feature)
	if len(ctxt.errors) > 0 {
		return ctxt.errors
	}
	return
}

type parseContext struct {
	p      *parser
	s      Scanner
	b      Builder
	m      Matcher
	queue  []*scanResult
	errors parseErrors
}

func (ctxt *parseContext) addError(e error) {
	ctxt.errors = append(ctxt.errors, e)
	// if (p.errors.length > 10)
	//   throw Errors.CompositeParserException.create(p.errors);
}

type scanResult struct {
	line  *Line
	atEof bool
	err   error
}

func (ctxt *parseContext) scan() (*Line, bool, error) {
	l := len(ctxt.queue)
	if l > 0 {
		x := ctxt.queue[0]
		ctxt.queue = ctxt.queue[1:]
		return x.line, x.atEof, x.err
	}
	return ctxt.s.Scan()
}

func (ctxt *parseContext) startRule(r RuleType) (bool, error) {
	ok, err := ctxt.b.StartRule(r)
	if err != nil {
		ctxt.addError(err)
	}
	return ok, err
}

func (ctxt *parseContext) endRule(r RuleType) (bool, error) {
	ok, err := ctxt.b.EndRule(r)
	if err != nil {
		ctxt.addError(err)
	}
	return ok, err
}

func (ctxt *parseContext) build(t *Token) (bool, error) {
	ok, err := ctxt.b.Build(t)
	if err != nil {
		ctxt.addError(err)
	}
	return ok, err
}

func (ctxt *parseContext) match(state int, line *Line) (newState int, err error) {
	switch state {
	case 0:
		return ctxt.matchAt_0(line)
	case 1:
		return ctxt.matchAt_1(line)
	case 2:
		return ctxt.matchAt_2(line)
	case 3:
		return ctxt.matchAt_3(line)
	case 4:
		return ctxt.matchAt_4(line)
	case 5:
		return ctxt.matchAt_5(line)
	case 6:
		return ctxt.matchAt_6(line)
	case 7:
		return ctxt.matchAt_7(line)
	case 8:
		return ctxt.matchAt_8(line)
	case 9:
		return ctxt.matchAt_9(line)
	case 10:
		return ctxt.matchAt_10(line)
	case 11:
		return ctxt.matchAt_11(line)
	case 12:
		return ctxt.matchAt_12(line)
	case 13:
		return ctxt.matchAt_13(line)
	case 14:
		return ctxt.matchAt_14(line)
	case 15:
		return ctxt.matchAt_15(line)
	case 16:
		return ctxt.matchAt_16(line)
	case 17:
		return ctxt.matchAt_17(line)
	case 18:
		return ctxt.matchAt_18(line)
	case 19:
		return ctxt.matchAt_19(line)
	case 20:
		return ctxt.matchAt_20(line)
	case 21:
		return ctxt.matchAt_21(line)
	case 22:
		return ctxt.matchAt_22(line)
	case 23:
		return ctxt.matchAt_23(line)
	case 24:
		return ctxt.matchAt_24(line)
	case 25:
		return ctxt.matchAt_25(line)
	case 26:
		return ctxt.matchAt_26(line)
	case 27:
		return ctxt.matchAt_27(line)
	case 29:
		return ctxt.matchAt_29(line)
	case 30:
		return ctxt.matchAt_30(line)
	case 31:
		return ctxt.matchAt_31(line)
	case 32:
		return ctxt.matchAt_32(line)
	case 33:
		return ctxt.matchAt_33(line)
	case 34:
		return ctxt.matchAt_34(line)
	default:
		return state, fmt.Errorf("Unknown state: %+v", state)
	}
}

// Start
func (ctxt *parseContext) matchAt_0(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_Language(line); ok {
		ctxt.startRule(RuleType_Feature_Header)
		ctxt.build(token)
		return 1, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.startRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 2, err
	}
	if ok, token, err := ctxt.match_FeatureLine(line); ok {
		ctxt.startRule(RuleType_Feature_Header)
		ctxt.build(token)
		return 3, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 0, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 0, err
	}

	// var stateComment = "State: 0 - Start"
	var expectedTokens = []string{"#Language", "#TagLine", "#FeatureLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 0, err
}

// Feature:0>Feature_Header:0>#Language:0
func (ctxt *parseContext) matchAt_1(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 2, err
	}
	if ok, token, err := ctxt.match_FeatureLine(line); ok {
		ctxt.build(token)
		return 3, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 1, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 1, err
	}

	// var stateComment = "State: 1 - Feature:0>Feature_Header:0>#Language:0"
	var expectedTokens = []string{"#TagLine", "#FeatureLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 1, err
}

// Feature:0>Feature_Header:1>Tags:0>#TagLine:0
func (ctxt *parseContext) matchAt_2(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.build(token)
		return 2, err
	}
	if ok, token, err := ctxt.match_FeatureLine(line); ok {
		ctxt.endRule(RuleType_Tags)
		ctxt.build(token)
		return 3, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 2, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 2, err
	}

	// var stateComment = "State: 2 - Feature:0>Feature_Header:1>Tags:0>#TagLine:0"
	var expectedTokens = []string{"#TagLine", "#FeatureLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 2, err
}

// Feature:0>Feature_Header:2>#FeatureLine:0
func (ctxt *parseContext) matchAt_3(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 3, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 5, err
	}
	if ok, token, err := ctxt.match_BackgroundLine(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Background)
		ctxt.build(token)
		return 6, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.startRule(RuleType_Description)
		ctxt.build(token)
		return 4, err
	}

	// var stateComment = "State: 3 - Feature:0>Feature_Header:2>#FeatureLine:0"
	var expectedTokens = []string{"#EOF", "#Empty", "#Comment", "#BackgroundLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 3, err
}

// Feature:0>Feature_Header:3>Feature_Description:0>Description_Helper:1>Description:0>#Other:0
func (ctxt *parseContext) matchAt_4(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.build(token)
		return 5, err
	}
	if ok, token, err := ctxt.match_BackgroundLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Background)
		ctxt.build(token)
		return 6, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.build(token)
		return 4, err
	}

	// var stateComment = "State: 4 - Feature:0>Feature_Header:3>Feature_Description:0>Description_Helper:1>Description:0>#Other:0"
	var expectedTokens = []string{"#EOF", "#Comment", "#BackgroundLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 4, err
}

// Feature:0>Feature_Header:3>Feature_Description:0>Description_Helper:2>#Comment:0
func (ctxt *parseContext) matchAt_5(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 5, err
	}
	if ok, token, err := ctxt.match_BackgroundLine(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Background)
		ctxt.build(token)
		return 6, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Feature_Header)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 5, err
	}

	// var stateComment = "State: 5 - Feature:0>Feature_Header:3>Feature_Description:0>Description_Helper:2>#Comment:0"
	var expectedTokens = []string{"#EOF", "#Comment", "#BackgroundLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 5, err
}

// Feature:1>Background:0>#BackgroundLine:0
func (ctxt *parseContext) matchAt_6(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Background)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 6, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 8, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 9, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.startRule(RuleType_Description)
		ctxt.build(token)
		return 7, err
	}

	// var stateComment = "State: 6 - Feature:1>Background:0>#BackgroundLine:0"
	var expectedTokens = []string{"#EOF", "#Empty", "#Comment", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 6, err
}

// Feature:1>Background:1>Background_Description:0>Description_Helper:1>Description:0>#Other:0
func (ctxt *parseContext) matchAt_7(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Background)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.build(token)
		return 8, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 9, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.build(token)
		return 7, err
	}

	// var stateComment = "State: 7 - Feature:1>Background:1>Background_Description:0>Description_Helper:1>Description:0>#Other:0"
	var expectedTokens = []string{"#EOF", "#Comment", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 7, err
}

// Feature:1>Background:1>Background_Description:0>Description_Helper:2>#Comment:0
func (ctxt *parseContext) matchAt_8(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Background)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 8, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 9, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 8, err
	}

	// var stateComment = "State: 8 - Feature:1>Background:1>Background_Description:0>Description_Helper:2>#Comment:0"
	var expectedTokens = []string{"#EOF", "#Comment", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 8, err
}

// Feature:1>Background:2>Scenario_Step:0>Step:0>#StepLine:0
func (ctxt *parseContext) matchAt_9(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.startRule(RuleType_DataTable)
		ctxt.build(token)
		return 10, err
	}
	if ok, token, err := ctxt.match_DocStringSeparator(line); ok {
		ctxt.startRule(RuleType_DocString)
		ctxt.build(token)
		return 33, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 9, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 9, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 9, err
	}

	// var stateComment = "State: 9 - Feature:1>Background:2>Scenario_Step:0>Step:0>#StepLine:0"
	var expectedTokens = []string{"#EOF", "#TableRow", "#DocStringSeparator", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 9, err
}

// Feature:1>Background:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:0>DataTable:0>#TableRow:0
func (ctxt *parseContext) matchAt_10(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.build(token)
		return 10, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 9, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 10, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 10, err
	}

	// var stateComment = "State: 10 - Feature:1>Background:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:0>DataTable:0>#TableRow:0"
	var expectedTokens = []string{"#EOF", "#TableRow", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 10, err
}

// Feature:2>Scenario_Definition:0>Tags:0>#TagLine:0
func (ctxt *parseContext) matchAt_11(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Tags)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Tags)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 11, err
	}

	// var stateComment = "State: 11 - Feature:2>Scenario_Definition:0>Tags:0>#TagLine:0"
	var expectedTokens = []string{"#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 11, err
}

// Feature:2>Scenario_Definition:1>__alt0:0>Scenario:0>#ScenarioLine:0
func (ctxt *parseContext) matchAt_12(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 14, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 15, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.startRule(RuleType_Description)
		ctxt.build(token)
		return 13, err
	}

	// var stateComment = "State: 12 - Feature:2>Scenario_Definition:1>__alt0:0>Scenario:0>#ScenarioLine:0"
	var expectedTokens = []string{"#EOF", "#Empty", "#Comment", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 12, err
}

// Feature:2>Scenario_Definition:1>__alt0:0>Scenario:1>Scenario_Description:0>Description_Helper:1>Description:0>#Other:0
func (ctxt *parseContext) matchAt_13(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.build(token)
		return 14, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 15, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.build(token)
		return 13, err
	}

	// var stateComment = "State: 13 - Feature:2>Scenario_Definition:1>__alt0:0>Scenario:1>Scenario_Description:0>Description_Helper:1>Description:0>#Other:0"
	var expectedTokens = []string{"#EOF", "#Comment", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 13, err
}

// Feature:2>Scenario_Definition:1>__alt0:0>Scenario:1>Scenario_Description:0>Description_Helper:2>#Comment:0
func (ctxt *parseContext) matchAt_14(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 14, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 15, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 14, err
	}

	// var stateComment = "State: 14 - Feature:2>Scenario_Definition:1>__alt0:0>Scenario:1>Scenario_Description:0>Description_Helper:2>#Comment:0"
	var expectedTokens = []string{"#EOF", "#Comment", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 14, err
}

// Feature:2>Scenario_Definition:1>__alt0:0>Scenario:2>Scenario_Step:0>Step:0>#StepLine:0
func (ctxt *parseContext) matchAt_15(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.startRule(RuleType_DataTable)
		ctxt.build(token)
		return 16, err
	}
	if ok, token, err := ctxt.match_DocStringSeparator(line); ok {
		ctxt.startRule(RuleType_DocString)
		ctxt.build(token)
		return 31, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 15, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 15, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 15, err
	}

	// var stateComment = "State: 15 - Feature:2>Scenario_Definition:1>__alt0:0>Scenario:2>Scenario_Step:0>Step:0>#StepLine:0"
	var expectedTokens = []string{"#EOF", "#TableRow", "#DocStringSeparator", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 15, err
}

// Feature:2>Scenario_Definition:1>__alt0:0>Scenario:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:0>DataTable:0>#TableRow:0
func (ctxt *parseContext) matchAt_16(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.build(token)
		return 16, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 15, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 16, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 16, err
	}

	// var stateComment = "State: 16 - Feature:2>Scenario_Definition:1>__alt0:0>Scenario:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:0>DataTable:0>#TableRow:0"
	var expectedTokens = []string{"#EOF", "#TableRow", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 16, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:0>#ScenarioOutlineLine:0
func (ctxt *parseContext) matchAt_17(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 19, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 20, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 22, err
	}
	if ok, token, err := ctxt.match_ExamplesLine(line); ok {
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Examples)
		ctxt.build(token)
		return 23, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.startRule(RuleType_Description)
		ctxt.build(token)
		return 18, err
	}

	// var stateComment = "State: 17 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:0>#ScenarioOutlineLine:0"
	var expectedTokens = []string{"#Empty", "#Comment", "#StepLine", "#TagLine", "#ExamplesLine", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 17, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:1>ScenarioOutline_Description:0>Description_Helper:1>Description:0>#Other:0
func (ctxt *parseContext) matchAt_18(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.build(token)
		return 19, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 20, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 22, err
	}
	if ok, token, err := ctxt.match_ExamplesLine(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Examples)
		ctxt.build(token)
		return 23, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.build(token)
		return 18, err
	}

	// var stateComment = "State: 18 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:1>ScenarioOutline_Description:0>Description_Helper:1>Description:0>#Other:0"
	var expectedTokens = []string{"#Comment", "#StepLine", "#TagLine", "#ExamplesLine", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 18, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:1>ScenarioOutline_Description:0>Description_Helper:2>#Comment:0
func (ctxt *parseContext) matchAt_19(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 19, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 20, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 22, err
	}
	if ok, token, err := ctxt.match_ExamplesLine(line); ok {
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Examples)
		ctxt.build(token)
		return 23, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 19, err
	}

	// var stateComment = "State: 19 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:1>ScenarioOutline_Description:0>Description_Helper:2>#Comment:0"
	var expectedTokens = []string{"#Comment", "#StepLine", "#TagLine", "#ExamplesLine", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 19, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:2>ScenarioOutline_Step:0>Step:0>#StepLine:0
func (ctxt *parseContext) matchAt_20(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.startRule(RuleType_DataTable)
		ctxt.build(token)
		return 21, err
	}
	if ok, token, err := ctxt.match_DocStringSeparator(line); ok {
		ctxt.startRule(RuleType_DocString)
		ctxt.build(token)
		return 29, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 20, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 22, err
	}
	if ok, token, err := ctxt.match_ExamplesLine(line); ok {
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Examples)
		ctxt.build(token)
		return 23, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 20, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 20, err
	}

	// var stateComment = "State: 20 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:2>ScenarioOutline_Step:0>Step:0>#StepLine:0"
	var expectedTokens = []string{"#TableRow", "#DocStringSeparator", "#StepLine", "#TagLine", "#ExamplesLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 20, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:2>ScenarioOutline_Step:0>Step:1>Step_Arg:0>__alt1:0>DataTable:0>#TableRow:0
func (ctxt *parseContext) matchAt_21(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.build(token)
		return 21, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 20, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 22, err
	}
	if ok, token, err := ctxt.match_ExamplesLine(line); ok {
		ctxt.endRule(RuleType_DataTable)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Examples)
		ctxt.build(token)
		return 23, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 21, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 21, err
	}

	// var stateComment = "State: 21 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:2>ScenarioOutline_Step:0>Step:1>Step_Arg:0>__alt1:0>DataTable:0>#TableRow:0"
	var expectedTokens = []string{"#TableRow", "#StepLine", "#TagLine", "#ExamplesLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 21, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:0>Tags:0>#TagLine:0
func (ctxt *parseContext) matchAt_22(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.build(token)
		return 22, err
	}
	if ok, token, err := ctxt.match_ExamplesLine(line); ok {
		ctxt.endRule(RuleType_Tags)
		ctxt.startRule(RuleType_Examples)
		ctxt.build(token)
		return 23, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 22, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 22, err
	}

	// var stateComment = "State: 22 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:0>Tags:0>#TagLine:0"
	var expectedTokens = []string{"#TagLine", "#ExamplesLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 22, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:0>#ExamplesLine:0
func (ctxt *parseContext) matchAt_23(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 23, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 25, err
	}
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.build(token)
		return 26, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.startRule(RuleType_Description)
		ctxt.build(token)
		return 24, err
	}

	// var stateComment = "State: 23 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:0>#ExamplesLine:0"
	var expectedTokens = []string{"#Empty", "#Comment", "#TableRow", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 23, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:1>Examples_Description:0>Description_Helper:1>Description:0>#Other:0
func (ctxt *parseContext) matchAt_24(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.build(token)
		return 25, err
	}
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.endRule(RuleType_Description)
		ctxt.build(token)
		return 26, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.build(token)
		return 24, err
	}

	// var stateComment = "State: 24 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:1>Examples_Description:0>Description_Helper:1>Description:0>#Other:0"
	var expectedTokens = []string{"#Comment", "#TableRow", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 24, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:1>Examples_Description:0>Description_Helper:2>#Comment:0
func (ctxt *parseContext) matchAt_25(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 25, err
	}
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.build(token)
		return 26, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 25, err
	}

	// var stateComment = "State: 25 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:1>Examples_Description:0>Description_Helper:2>#Comment:0"
	var expectedTokens = []string{"#Comment", "#TableRow", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 25, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:2>#TableRow:0
func (ctxt *parseContext) matchAt_26(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.build(token)
		return 27, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 26, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 26, err
	}

	// var stateComment = "State: 26 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:2>#TableRow:0"
	var expectedTokens = []string{"#TableRow", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 26, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:3>#TableRow:0
func (ctxt *parseContext) matchAt_27(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_Examples)
		ctxt.endRule(RuleType_Examples_Definition)
		ctxt.endRule(RuleType_ScenarioOutline)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_TableRow(line); ok {
		ctxt.build(token)
		return 27, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		if ctxt.lookahead_0(line) {
			ctxt.endRule(RuleType_Examples)
			ctxt.endRule(RuleType_Examples_Definition)
			ctxt.startRule(RuleType_Examples_Definition)
			ctxt.startRule(RuleType_Tags)
			ctxt.build(token)
			return 22, err
		}
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_Examples)
		ctxt.endRule(RuleType_Examples_Definition)
		ctxt.endRule(RuleType_ScenarioOutline)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ExamplesLine(line); ok {
		ctxt.endRule(RuleType_Examples)
		ctxt.endRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Examples)
		ctxt.build(token)
		return 23, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_Examples)
		ctxt.endRule(RuleType_Examples_Definition)
		ctxt.endRule(RuleType_ScenarioOutline)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_Examples)
		ctxt.endRule(RuleType_Examples_Definition)
		ctxt.endRule(RuleType_ScenarioOutline)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 27, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 27, err
	}

	// var stateComment = "State: 27 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:3>Examples_Definition:1>Examples:3>#TableRow:0"
	var expectedTokens = []string{"#EOF", "#TableRow", "#TagLine", "#ExamplesLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 27, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:2>ScenarioOutline_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:0>#DocStringSeparator:0
func (ctxt *parseContext) matchAt_29(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_DocStringSeparator(line); ok {
		ctxt.build(token)
		return 30, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.build(token)
		return 29, err
	}

	// var stateComment = "State: 29 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:2>ScenarioOutline_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:0>#DocStringSeparator:0"
	var expectedTokens = []string{"#DocStringSeparator", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 29, err
}

// Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:2>ScenarioOutline_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:2>#DocStringSeparator:0
func (ctxt *parseContext) matchAt_30(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 20, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 22, err
	}
	if ok, token, err := ctxt.match_ExamplesLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Examples_Definition)
		ctxt.startRule(RuleType_Examples)
		ctxt.build(token)
		return 23, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 30, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 30, err
	}

	// var stateComment = "State: 30 - Feature:2>Scenario_Definition:1>__alt0:1>ScenarioOutline:2>ScenarioOutline_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:2>#DocStringSeparator:0"
	var expectedTokens = []string{"#StepLine", "#TagLine", "#ExamplesLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 30, err
}

// Feature:2>Scenario_Definition:1>__alt0:0>Scenario:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:0>#DocStringSeparator:0
func (ctxt *parseContext) matchAt_31(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_DocStringSeparator(line); ok {
		ctxt.build(token)
		return 32, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.build(token)
		return 31, err
	}

	// var stateComment = "State: 31 - Feature:2>Scenario_Definition:1>__alt0:0>Scenario:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:0>#DocStringSeparator:0"
	var expectedTokens = []string{"#DocStringSeparator", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 31, err
}

// Feature:2>Scenario_Definition:1>__alt0:0>Scenario:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:2>#DocStringSeparator:0
func (ctxt *parseContext) matchAt_32(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 15, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Scenario)
		ctxt.endRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 32, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 32, err
	}

	// var stateComment = "State: 32 - Feature:2>Scenario_Definition:1>__alt0:0>Scenario:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:2>#DocStringSeparator:0"
	var expectedTokens = []string{"#EOF", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 32, err
}

// Feature:1>Background:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:0>#DocStringSeparator:0
func (ctxt *parseContext) matchAt_33(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_DocStringSeparator(line); ok {
		ctxt.build(token)
		return 34, err
	}
	if ok, token, err := ctxt.match_Other(line); ok {
		ctxt.build(token)
		return 33, err
	}

	// var stateComment = "State: 33 - Feature:1>Background:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:0>#DocStringSeparator:0"
	var expectedTokens = []string{"#DocStringSeparator", "#Other"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 33, err
}

// Feature:1>Background:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:2>#DocStringSeparator:0
func (ctxt *parseContext) matchAt_34(line *Line) (newState int, err error) {
	if ok, token, err := ctxt.match_EOF(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.build(token)
		return 28, err
	}
	if ok, token, err := ctxt.match_StepLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.startRule(RuleType_Step)
		ctxt.build(token)
		return 9, err
	}
	if ok, token, err := ctxt.match_TagLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Tags)
		ctxt.build(token)
		return 11, err
	}
	if ok, token, err := ctxt.match_ScenarioLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_Scenario)
		ctxt.build(token)
		return 12, err
	}
	if ok, token, err := ctxt.match_ScenarioOutlineLine(line); ok {
		ctxt.endRule(RuleType_DocString)
		ctxt.endRule(RuleType_Step)
		ctxt.endRule(RuleType_Background)
		ctxt.startRule(RuleType_Scenario_Definition)
		ctxt.startRule(RuleType_ScenarioOutline)
		ctxt.build(token)
		return 17, err
	}
	if ok, token, err := ctxt.match_Comment(line); ok {
		ctxt.build(token)
		return 34, err
	}
	if ok, token, err := ctxt.match_Empty(line); ok {
		ctxt.build(token)
		return 34, err
	}

	// var stateComment = "State: 34 - Feature:1>Background:2>Scenario_Step:0>Step:1>Step_Arg:0>__alt1:1>DocString:2>#DocStringSeparator:0"
	var expectedTokens = []string{"#EOF", "#StepLine", "#TagLine", "#ScenarioLine", "#ScenarioOutlineLine", "#Comment", "#Empty"}
	if line.IsEof() {
		err = fmt.Errorf("(%d:0): unexpected end of file, expected: %s", line.LineNumber, strings.Join(expectedTokens, ", "))
	} else {
		err = fmt.Errorf("(%d:%d): expected: %s, got '%s'", line.LineNumber, line.Indent()+1, strings.Join(expectedTokens, ", "), line.LineText)
	}
	// if (ctxt.p.stopAtFirstError) throw error;
	//ctxt.addError(err)
	return 34, err
}

type Matcher interface {
	MatchEOF(line *Line) (bool, *Token, error)
	MatchEmpty(line *Line) (bool, *Token, error)
	MatchComment(line *Line) (bool, *Token, error)
	MatchTagLine(line *Line) (bool, *Token, error)
	MatchFeatureLine(line *Line) (bool, *Token, error)
	MatchBackgroundLine(line *Line) (bool, *Token, error)
	MatchScenarioLine(line *Line) (bool, *Token, error)
	MatchScenarioOutlineLine(line *Line) (bool, *Token, error)
	MatchExamplesLine(line *Line) (bool, *Token, error)
	MatchStepLine(line *Line) (bool, *Token, error)
	MatchDocStringSeparator(line *Line) (bool, *Token, error)
	MatchTableRow(line *Line) (bool, *Token, error)
	MatchLanguage(line *Line) (bool, *Token, error)
	MatchOther(line *Line) (bool, *Token, error)
	Reset()
}

func (ctxt *parseContext) isMatch_EOF(line *Line) bool {
	ok, _, _ := ctxt.match_EOF(line)
	return ok
}
func (ctxt *parseContext) match_EOF(line *Line) (bool, *Token, error) {
	return ctxt.m.MatchEOF(line)
}

func (ctxt *parseContext) isMatch_Empty(line *Line) bool {
	ok, _, _ := ctxt.match_Empty(line)
	return ok
}
func (ctxt *parseContext) match_Empty(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchEmpty(line)
}

func (ctxt *parseContext) isMatch_Comment(line *Line) bool {
	ok, _, _ := ctxt.match_Comment(line)
	return ok
}
func (ctxt *parseContext) match_Comment(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchComment(line)
}

func (ctxt *parseContext) isMatch_TagLine(line *Line) bool {
	ok, _, _ := ctxt.match_TagLine(line)
	return ok
}
func (ctxt *parseContext) match_TagLine(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchTagLine(line)
}

func (ctxt *parseContext) isMatch_FeatureLine(line *Line) bool {
	ok, _, _ := ctxt.match_FeatureLine(line)
	return ok
}
func (ctxt *parseContext) match_FeatureLine(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchFeatureLine(line)
}

func (ctxt *parseContext) isMatch_BackgroundLine(line *Line) bool {
	ok, _, _ := ctxt.match_BackgroundLine(line)
	return ok
}
func (ctxt *parseContext) match_BackgroundLine(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchBackgroundLine(line)
}

func (ctxt *parseContext) isMatch_ScenarioLine(line *Line) bool {
	ok, _, _ := ctxt.match_ScenarioLine(line)
	return ok
}
func (ctxt *parseContext) match_ScenarioLine(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchScenarioLine(line)
}

func (ctxt *parseContext) isMatch_ScenarioOutlineLine(line *Line) bool {
	ok, _, _ := ctxt.match_ScenarioOutlineLine(line)
	return ok
}
func (ctxt *parseContext) match_ScenarioOutlineLine(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchScenarioOutlineLine(line)
}

func (ctxt *parseContext) isMatch_ExamplesLine(line *Line) bool {
	ok, _, _ := ctxt.match_ExamplesLine(line)
	return ok
}
func (ctxt *parseContext) match_ExamplesLine(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchExamplesLine(line)
}

func (ctxt *parseContext) isMatch_StepLine(line *Line) bool {
	ok, _, _ := ctxt.match_StepLine(line)
	return ok
}
func (ctxt *parseContext) match_StepLine(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchStepLine(line)
}

func (ctxt *parseContext) isMatch_DocStringSeparator(line *Line) bool {
	ok, _, _ := ctxt.match_DocStringSeparator(line)
	return ok
}
func (ctxt *parseContext) match_DocStringSeparator(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchDocStringSeparator(line)
}

func (ctxt *parseContext) isMatch_TableRow(line *Line) bool {
	ok, _, _ := ctxt.match_TableRow(line)
	return ok
}
func (ctxt *parseContext) match_TableRow(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchTableRow(line)
}

func (ctxt *parseContext) isMatch_Language(line *Line) bool {
	ok, _, _ := ctxt.match_Language(line)
	return ok
}
func (ctxt *parseContext) match_Language(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchLanguage(line)
}

func (ctxt *parseContext) isMatch_Other(line *Line) bool {
	ok, _, _ := ctxt.match_Other(line)
	return ok
}
func (ctxt *parseContext) match_Other(line *Line) (bool, *Token, error) {
	if line.IsEof() {
		return false, nil, nil
	}
	return ctxt.m.MatchOther(line)
}

func (ctxt *parseContext) lookahead_0(initialLine *Line) bool {
	var queue []*scanResult
	var match bool

	for {
		line, atEof, err := ctxt.scan()
		queue = append(queue, &scanResult{line, atEof, err})

		if false || ctxt.isMatch_ExamplesLine(line) {
			match = true
			break
		}
		if !(false || ctxt.isMatch_Empty(line) || ctxt.isMatch_Comment(line) || ctxt.isMatch_TagLine(line)) {
			break
		}
		if atEof {
			break
		}
	}

	ctxt.queue = append(ctxt.queue, queue...)

	return match
}
