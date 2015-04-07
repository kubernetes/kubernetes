package restful

import "testing"

var tempregexs = []struct {
	template, regex        string
	literalCount, varCount int
}{
	{"", "^(/.*)?$", 0, 0},
	{"/a/{b}/c/", "^/a/([^/]+?)/c(/.*)?$", 2, 1},
	{"/{a}/{b}/{c-d-e}/", "^/([^/]+?)/([^/]+?)/([^/]+?)(/.*)?$", 0, 3},
	{"/{p}/abcde", "^/([^/]+?)/abcde(/.*)?$", 5, 1},
	{"/a/{b:*}", "^/a/(.*)(/.*)?$", 1, 1},
	{"/a/{b:[a-z]+}", "^/a/([a-z]+)(/.*)?$", 1, 1},
}

func TestTemplateToRegularExpression(t *testing.T) {
	ok := true
	for i, fixture := range tempregexs {
		actual, lCount, vCount, _ := templateToRegularExpression(fixture.template)
		if actual != fixture.regex {
			t.Logf("regex mismatch, expected:%v , actual:%v, line:%v\n", fixture.regex, actual, i) // 11 = where the data starts
			ok = false
		}
		if lCount != fixture.literalCount {
			t.Logf("literal count mismatch, expected:%v , actual:%v, line:%v\n", fixture.literalCount, lCount, i)
			ok = false
		}
		if vCount != fixture.varCount {
			t.Logf("variable count mismatch, expected:%v , actual:%v, line:%v\n", fixture.varCount, vCount, i)
			ok = false
		}
	}
	if !ok {
		t.Fatal("one or more expression did not match")
	}
}
