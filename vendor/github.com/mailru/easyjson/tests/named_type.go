package tests

//easyjson:json
type NamedType struct {
	Inner struct {
		// easyjson is mistakenly naming the type of this field 'tests.MyString' in the generated output
		// something about a named type inside an anonmymous type is triggering this bug
		Field  MyString `tag:"value"`
		Field2 int      "tag:\"value with ` in it\""
	}
}

type MyString string

var namedTypeValue NamedType

func init() {
	namedTypeValue.Inner.Field = "test"
	namedTypeValue.Inner.Field2 = 123
}

var namedTypeValueString = `{"Inner":{"Field":"test","Field2":123}}`
