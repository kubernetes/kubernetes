package yh

import (
	"testing"
)

func TestIsKeyValue(t *testing.T) {
	l := yamlLine{raw: "serviceAccount: hashicorp-consul-server"}
	if l.isKeyValue() != true {
		t.Errorf("Expected True but got %v - %v%v", l.isKeyValue(), l.key, l.value)
	}

	l = yamlLine{raw: "https://github.com"}
	if l.isKeyValue() != false {
		t.Errorf("Expected False for URL but got %v", l.isKeyValue())
	}

	l = yamlLine{raw: "f:deployment.kubernetes.io/revision: {}"}
	if l.isKeyValue() != true {
		t.Errorf("Expected True but got %v - '%v%v'", l.isKeyValue(), l.key, l.value)
	}

	l = yamlLine{raw: "k{\"type\"\"Progressing\"}:"}
	if l.isKeyValue() != true {
		t.Errorf("Expected True but got %v - '%v%v'", l.isKeyValue(), l.key, l.value)
	}

	//Malformed line
	l = yamlLine{raw: "serviceAccount:hashicorp-consul-server"}
	if l.isKeyValue() != false {
		t.Errorf("Expected False but got %v - %v%v", l.isKeyValue(), l.key, l.value)
	}
}

func TestGetKeyValue(t *testing.T) {
	l := yamlLine{
		raw: "foo: bar",
	}

	l.isKeyValue()

	if l.key != "foo" {
		t.Errorf("Expected 'foo' but got %v", l.key)
	}
	if l.value != "bar" {
		t.Errorf("Expected 'bar' but got %v", l.value)
	}
}

func TestIsComment(t *testing.T) {
	l := yamlLine{raw: "\t#this is a comment"}
	if l.isComment() != true {
		t.Errorf("Expected true but got %v", l.isComment())
	}

	l = yamlLine{raw: "#this is another comment"}
	if l.isComment() != true {
		t.Errorf("Expected true but got %v", l.isComment())
	}

	l = yamlLine{raw: "- this is not a comment"}
	if l.isComment() != false {
		t.Errorf("Expected false but got %v", l.isComment())
	}

}

func TestValueIsBoolean(t *testing.T) {
	l := yamlLine{value: "true"}
	if l.valueIsBoolean() != true {
		t.Errorf("Expected true but got %v", l.valueIsBoolean())
	}

	l = yamlLine{value: "false"}
	if l.valueIsBoolean() != true {
		t.Errorf("Expected true but got %v", l.valueIsBoolean())
	}

	l = yamlLine{value: "False"}
	if l.valueIsBoolean() != true {
		t.Errorf("Expected true but got %v", l.valueIsBoolean())
	}

	l = yamlLine{value: "not boolean"}
	if l.valueIsBoolean() != false {
		t.Errorf("Expected false but got %v", l.valueIsBoolean())
	}

}

func TestValueIsNumberOrIP(t *testing.T) {
	l := yamlLine{value: "657890"}
	if l.valueIsNumberOrIP() != true {
		t.Errorf("Expected true but got %v", l.valueIsNumberOrIP())
	}

	l = yamlLine{value: "123123.12312"}
	if l.valueIsNumberOrIP() != true {
		t.Errorf("Expected true but got %v", l.valueIsNumberOrIP())
	}

	l = yamlLine{value: "127.0.0.1"}
	if l.valueIsNumberOrIP() != true {
		t.Errorf("Expected true but got %v", l.valueIsNumberOrIP())
	}

	l = yamlLine{value: "test not a number"}
	if l.valueIsNumberOrIP() != false {
		t.Errorf("Expected false but got %v", l.valueIsNumberOrIP())
	}

}

func TestIsEmptyLine(t *testing.T) {
	l := yamlLine{raw: " "}

	if l.isEmptyLine() != true {
		t.Errorf("Expected true but got %v", l.isEmptyLine())
	}

	l = yamlLine{raw: `
`}
	if l.isEmptyLine() != true {
		t.Errorf("Expected true but got %v", l.isEmptyLine())
	}

	l = yamlLine{raw: "\t \n"}
	if l.isEmptyLine() != true {
		t.Errorf("Expected true but got %v", l.isEmptyLine())
	}

	l = yamlLine{raw: "asdasd"}
	if l.isEmptyLine() != false {
		t.Errorf("Expected false but got %v", l.isEmptyLine())
	}

}

func TestIsElementOfList(t *testing.T) {
	l := yamlLine{raw: "- apple"}

	if l.isElementOfList() != true {
		t.Errorf("Expected true but got %v", l.isElementOfList())
	}

	l = yamlLine{raw: "- -apple"}
	if l.isElementOfList() != true {
		t.Errorf("Expected true but got %v", l.isElementOfList())
	}

	l = yamlLine{raw: "apple"}
	if l.isElementOfList() != false {
		t.Errorf("Expected false but got %v", l.isElementOfList())
	}

	l = yamlLine{raw: "apple-"}
	if l.isElementOfList() != false {
		t.Errorf("Expected false but got %v", l.isElementOfList())
	}

	//TODO: make this test pass
	//l = yamlLine{ raw: "--apple"}
	//if l.isElementOfList() != false {
	//	t.Errorf("Expected false but got %v", l.isElementOfList())
	//}

}

func TestIndentationSapces(t *testing.T) {
	l := yamlLine{raw: "  foo: bar"}

	if l.indentationSpaces() != 2 {
		t.Errorf("Expected 2 but got %v", l.indentationSpaces())
	}

	l = yamlLine{raw: "    foo: bar"}

	if l.indentationSpaces() != 4 {
		t.Errorf("Expected 4 but got %v", l.indentationSpaces())
	}

	l = yamlLine{raw: "foo: bar"}

	if l.indentationSpaces() != 0 {
		t.Errorf("Expected 0 but got %v", l.indentationSpaces())
	}

}

func TestValueContainsChompingIndicator(t *testing.T) {
	l := yamlLine{value: "something >"}

	if l.valueContainsChompingIndicator() != true {
		t.Errorf("Expected true but got %v", l.valueContainsChompingIndicator())
	}

	l = yamlLine{value: "|-"}
	if l.valueContainsChompingIndicator() != true {
		t.Errorf("Expected true but got %v", l.valueContainsChompingIndicator())
	}

	l = yamlLine{value: "something |"}
	if l.valueContainsChompingIndicator() != true {
		t.Errorf("Expected true but got %v", l.valueContainsChompingIndicator())
	}

	l = yamlLine{value: "something --"}
	if l.valueContainsChompingIndicator() != false {
		t.Errorf("Expected false but got %v", l.valueContainsChompingIndicator())
	}

}

func TestIsURL(t *testing.T) {
	l := yamlLine{raw: "https://github.com"}

	if l.isUrl() != true {
		t.Errorf("Expected True but got %v", l.isUrl())
	}

	l = yamlLine{raw: "https//github.com"}
	if l.isUrl() != false {
		t.Errorf("Expected False for malformed URL but got %v", l.isUrl())
	}

	l = yamlLine{raw: "randomValuesNotURL: //6574738:123"}
	if l.isUrl() != false {
		t.Errorf("Expected False but got %v", l.isUrl())
	}
}
