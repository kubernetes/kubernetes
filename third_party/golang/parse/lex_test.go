package parse

import "testing"

func compare(t *testing.T, l *lexer, expectType []itemType, expectVal []string) {
	for i:=0; i<len(expectType); i++ {
		item := l.nextItem()
		if item.typ != expectType[i] {
			t.Errorf("expect to get %v, got %v", expectType[i], item.typ)
		}
		if item.val != expectVal[i] {
			t.Errorf("expect to get %v, got %v", expectVal[i], item.val)
		}
	}
}

func TestLexPlainText(t *testing.T) {
	text := "hello jsonpath"
	l := lex("hello", text, "${", "}")
	expectType := []itemType{itemText, itemEOF}
	expectVal := []string{"hello jsonpath", ""}
	compare(t, l, expectType, expectVal)

}

func TestLexVariable(t *testing.T) {
	text := "hello ${.foo}"
	l := lex("hello", text, "${", "}")
	expectType := []itemType{itemText, itemLeftDelim, itemDot, itemField, itemRightDelim, itemEOF}
	expectVal := []string{"hello ", "${", ".", "foo", "}", ""}
	compare(t, l, expectType, expectVal)
}

func TestLexQuote(t *testing.T) {
	text := `test ${"${"}`
	l := lex("quoto", text, "${", "}")
	expectType := []itemType{itemText, itemLeftDelim, itemString, itemRightDelim, itemEOF}
	expectVal := []string{"test ", "${", `"${"`, "}", ""}
	compare(t, l, expectType, expectVal)
}
