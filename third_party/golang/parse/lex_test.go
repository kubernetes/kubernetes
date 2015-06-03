package parse

import "testing"

func TestLexPlainText(t *testing.T) {
	text := "hello jsonpath"
	l := lex("hello", text, "${", "}")
	item := l.nextItem()
	if item.typ != itemText {
		t.Errorf("expect to get itemText, got %v", item)
	}
	if item.val != "hello jsonpath" {
		t.Errorf("expect to get %v, got %v", text, item.val)
	}
	item = l.nextItem()
	if item.typ != itemEOF {
		t.Errorf("expect to get itemEOF, got %v", item)
	}
}

func TestLexVariable(t *testing.T) {
	text := "hello ${.foo}"
	l := lex("hello", text, "${", "}")
	expect := []itemType{itemText, itemLeftDelim, itemDot, itemField, itemRightDelim, itemEOF}

	for i := 0; i < 6; i++ {
		item := l.nextItem()
		if item.typ != expect[i] {
			t.Errorf("expect to get %v, got %v", expect[i], item)
		}
	}
}
