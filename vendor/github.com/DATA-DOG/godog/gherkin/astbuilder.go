package gherkin

import (
	"strings"
)

type AstBuilder interface {
	Builder
	GetFeature() *Feature
}

type astBuilder struct {
	stack    []*astNode
	comments []*Comment
}

func (t *astBuilder) Reset() {
	t.comments = []*Comment{}
	t.stack = []*astNode{}
	t.push(newAstNode(RuleType_None))
}

func (t *astBuilder) GetFeature() *Feature {
	res := t.currentNode().getSingle(RuleType_Feature)
	if val, ok := res.(*Feature); ok {
		return val
	}
	return nil
}

type astNode struct {
	ruleType RuleType
	subNodes map[RuleType][]interface{}
}

func (a *astNode) add(rt RuleType, obj interface{}) {
	a.subNodes[rt] = append(a.subNodes[rt], obj)
}

func (a *astNode) getSingle(rt RuleType) interface{} {
	if val, ok := a.subNodes[rt]; ok {
		for i := range val {
			return val[i]
		}
	}
	return nil
}

func (a *astNode) getItems(rt RuleType) []interface{} {
	var res []interface{}
	if val, ok := a.subNodes[rt]; ok {
		for i := range val {
			res = append(res, val[i])
		}
	}
	return res
}

func (a *astNode) getToken(tt TokenType) *Token {
	if val, ok := a.getSingle(tt.RuleType()).(*Token); ok {
		return val
	}
	return nil
}

func (a *astNode) getTokens(tt TokenType) []*Token {
	var items = a.getItems(tt.RuleType())
	var tokens []*Token
	for i := range items {
		if val, ok := items[i].(*Token); ok {
			tokens = append(tokens, val)
		}
	}
	return tokens
}

func (t *astBuilder) currentNode() *astNode {
	if len(t.stack) > 0 {
		return t.stack[len(t.stack)-1]
	}
	return nil
}

func newAstNode(rt RuleType) *astNode {
	return &astNode{
		ruleType: rt,
		subNodes: make(map[RuleType][]interface{}),
	}
}

func NewAstBuilder() AstBuilder {
	builder := new(astBuilder)
	builder.comments = []*Comment{}
	builder.push(newAstNode(RuleType_None))
	return builder
}

func (t *astBuilder) push(n *astNode) {
	t.stack = append(t.stack, n)
}

func (t *astBuilder) pop() *astNode {
	x := t.stack[len(t.stack)-1]
	t.stack = t.stack[:len(t.stack)-1]
	return x
}

func (t *astBuilder) Build(tok *Token) (bool, error) {
	if tok.Type == TokenType_Comment {
		comment := new(Comment)
		comment.Type = "Comment"
		comment.Location = astLocation(tok)
		comment.Text = tok.Text
		t.comments = append(t.comments, comment)
	} else {
		t.currentNode().add(tok.Type.RuleType(), tok)
	}
	return true, nil
}
func (t *astBuilder) StartRule(r RuleType) (bool, error) {
	t.push(newAstNode(r))
	return true, nil
}
func (t *astBuilder) EndRule(r RuleType) (bool, error) {
	node := t.pop()
	transformedNode, err := t.transformNode(node)
	t.currentNode().add(node.ruleType, transformedNode)
	return true, err
}

func (t *astBuilder) transformNode(node *astNode) (interface{}, error) {
	switch node.ruleType {

	case RuleType_Step:
		stepLine := node.getToken(TokenType_StepLine)
		step := new(Step)
		step.Type = "Step"
		step.Location = astLocation(stepLine)
		step.Keyword = stepLine.Keyword
		step.Text = stepLine.Text
		step.Argument = node.getSingle(RuleType_DataTable)
		if step.Argument == nil {
			step.Argument = node.getSingle(RuleType_DocString)
		}
		return step, nil

	case RuleType_DocString:
		separatorToken := node.getToken(TokenType_DocStringSeparator)
		contentType := separatorToken.Text
		lineTokens := node.getTokens(TokenType_Other)
		var text string
		for i := range lineTokens {
			if i > 0 {
				text += "\n"
			}
			text += lineTokens[i].Text
		}
		ds := new(DocString)
		ds.Type = "DocString"
		ds.Location = astLocation(separatorToken)
		ds.ContentType = contentType
		ds.Content = text
		ds.Delimitter = DOCSTRING_SEPARATOR // TODO: remember separator
		return ds, nil

	case RuleType_DataTable:
		rows, err := astTableRows(node)
		dt := new(DataTable)
		dt.Type = "DataTable"
		dt.Location = rows[0].Location
		dt.Rows = rows
		return dt, err

	case RuleType_Background:
		backgroundLine := node.getToken(TokenType_BackgroundLine)
		description, _ := node.getSingle(RuleType_Description).(string)
		bg := new(Background)
		bg.Type = "Background"
		bg.Location = astLocation(backgroundLine)
		bg.Keyword = backgroundLine.Keyword
		bg.Name = backgroundLine.Text
		bg.Description = description
		bg.Steps = astSteps(node)
		return bg, nil

	case RuleType_Scenario_Definition:
		tags := astTags(node)
		scenarioNode, _ := node.getSingle(RuleType_Scenario).(*astNode)
		if scenarioNode != nil {
			scenarioLine := scenarioNode.getToken(TokenType_ScenarioLine)
			description, _ := scenarioNode.getSingle(RuleType_Description).(string)
			sc := new(Scenario)
			sc.Type = "Scenario"
			sc.Tags = tags
			sc.Location = astLocation(scenarioLine)
			sc.Keyword = scenarioLine.Keyword
			sc.Name = scenarioLine.Text
			sc.Description = description
			sc.Steps = astSteps(scenarioNode)
			return sc, nil
		} else {
			scenarioOutlineNode, ok := node.getSingle(RuleType_ScenarioOutline).(*astNode)
			if !ok {
				panic("Internal grammar error")
			}
			scenarioOutlineLine := scenarioOutlineNode.getToken(TokenType_ScenarioOutlineLine)
			description, _ := scenarioOutlineNode.getSingle(RuleType_Description).(string)
			sc := new(ScenarioOutline)
			sc.Type = "ScenarioOutline"
			sc.Tags = tags
			sc.Location = astLocation(scenarioOutlineLine)
			sc.Keyword = scenarioOutlineLine.Keyword
			sc.Name = scenarioOutlineLine.Text
			sc.Description = description
			sc.Steps = astSteps(scenarioOutlineNode)
			sc.Examples = astExamples(scenarioOutlineNode)
			return sc, nil
		}

	case RuleType_Examples_Definition:
		tags := astTags(node)
		examplesNode, _ := node.getSingle(RuleType_Examples).(*astNode)
		examplesLine := examplesNode.getToken(TokenType_ExamplesLine)
		description, _ := examplesNode.getSingle(RuleType_Description).(string)
		allRows, err := astTableRows(examplesNode)
		ex := new(Examples)
		ex.Type = "Examples"
		ex.Tags = tags
		ex.Location = astLocation(examplesLine)
		ex.Keyword = examplesLine.Keyword
		ex.Name = examplesLine.Text
		ex.Description = description
		ex.TableHeader = allRows[0]
		ex.TableBody = allRows[1:]
		return ex, err

	case RuleType_Description:
		lineTokens := node.getTokens(TokenType_Other)
		// Trim trailing empty lines
		end := len(lineTokens)
		for end > 0 && strings.TrimSpace(lineTokens[end-1].Text) == "" {
			end--
		}
		var desc []string
		for i := range lineTokens[0:end] {
			desc = append(desc, lineTokens[i].Text)
		}
		return strings.Join(desc, "\n"), nil

	case RuleType_Feature:
		header, ok := node.getSingle(RuleType_Feature_Header).(*astNode)
		if !ok {
			return nil, nil
		}
		tags := astTags(header)
		featureLine := header.getToken(TokenType_FeatureLine)
		if featureLine == nil {
			return nil, nil
		}
		background, _ := node.getSingle(RuleType_Background).(*Background)
		scenarioDefinitions := node.getItems(RuleType_Scenario_Definition)
		if scenarioDefinitions == nil {
			scenarioDefinitions = []interface{}{}
		}
		description, _ := header.getSingle(RuleType_Description).(string)

		feat := new(Feature)
		feat.Type = "Feature"
		feat.Tags = tags
		feat.Location = astLocation(featureLine)
		feat.Language = featureLine.GherkinDialect
		feat.Keyword = featureLine.Keyword
		feat.Name = featureLine.Text
		feat.Description = description
		feat.Background = background
		feat.ScenarioDefinitions = scenarioDefinitions
		feat.Comments = t.comments
		return feat, nil
	}
	return node, nil
}

func astLocation(t *Token) *Location {
	return &Location{
		Line:   t.Location.Line,
		Column: t.Location.Column,
	}
}

func astTableRows(t *astNode) (rows []*TableRow, err error) {
	rows = []*TableRow{}
	tokens := t.getTokens(TokenType_TableRow)
	for i := range tokens {
		row := new(TableRow)
		row.Type = "TableRow"
		row.Location = astLocation(tokens[i])
		row.Cells = astTableCells(tokens[i])
		rows = append(rows, row)
	}
	err = ensureCellCount(rows)
	return
}

func ensureCellCount(rows []*TableRow) error {
	if len(rows) <= 1 {
		return nil
	}
	cellCount := len(rows[0].Cells)
	for i := range rows {
		if cellCount != len(rows[i].Cells) {
			return &parseError{"inconsistent cell count within the table", &Location{
				Line:   rows[i].Location.Line,
				Column: rows[i].Location.Column,
			}}
		}
	}
	return nil
}

func astTableCells(t *Token) (cells []*TableCell) {
	cells = []*TableCell{}
	for i := range t.Items {
		item := t.Items[i]
		cell := new(TableCell)
		cell.Type = "TableCell"
		cell.Location = &Location{
			Line:   t.Location.Line,
			Column: item.Column,
		}
		cell.Value = item.Text
		cells = append(cells, cell)
	}
	return
}

func astSteps(t *astNode) (steps []*Step) {
	steps = []*Step{}
	tokens := t.getItems(RuleType_Step)
	for i := range tokens {
		step, _ := tokens[i].(*Step)
		steps = append(steps, step)
	}
	return
}

func astExamples(t *astNode) (examples []*Examples) {
	examples = []*Examples{}
	tokens := t.getItems(RuleType_Examples_Definition)
	for i := range tokens {
		example, _ := tokens[i].(*Examples)
		examples = append(examples, example)
	}
	return
}

func astTags(node *astNode) (tags []*Tag) {
	tags = []*Tag{}
	tagsNode, ok := node.getSingle(RuleType_Tags).(*astNode)
	if !ok {
		return
	}
	tokens := tagsNode.getTokens(TokenType_TagLine)
	for i := range tokens {
		token := tokens[i]
		for k := range token.Items {
			item := token.Items[k]
			tag := new(Tag)
			tag.Type = "Tag"
			tag.Location = &Location{
				Line:   token.Location.Line,
				Column: item.Column,
			}
			tag.Name = item.Text
			tags = append(tags, tag)
		}
	}
	return
}
