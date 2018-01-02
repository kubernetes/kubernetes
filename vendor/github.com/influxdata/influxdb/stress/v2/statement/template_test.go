package statement

import (
	"testing"
)

func TestNewTagFunc(t *testing.T) {
	wtags := newTestTagsTemplate()
	wfunc := newTestFunctionTemplate()

	expected := wtags.Tags[0]
	got := wtags.NewTagFunc()()
	if got != expected {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
	expected = "EMPTY TAGS"
	got = wfunc.NewTagFunc()()
	if got != expected {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func TestNumSeries(t *testing.T) {
	wtags := newTestTagsTemplate()
	wfunc := newTestFunctionTemplate()

	expected := len(wtags.Tags)
	got := wtags.numSeries()
	if got != expected {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
	expected = wfunc.Function.Count
	got = wfunc.numSeries()
	if got != expected {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func TestTemplatesInit(t *testing.T) {
	tmpls := newTestTemplates()
	s := tmpls.Init(5)
	vals := s.Eval(spoofTime)
	expected := tmpls[0].Tags[0]
	got := vals[0]
	if got != expected {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
	expected = "0i"
	got = vals[1]
	if got != expected {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}

func newTestTemplates() Templates {
	return []*Template{
		newTestTagsTemplate(),
		newTestFunctionTemplate(),
	}
}

func newTestTagsTemplate() *Template {
	return &Template{
		Tags: []string{"thing", "other_thing"},
	}
}

func newTestFunctionTemplate() *Template {
	return &Template{
		Function: newIntIncFunction(),
	}
}
