package tests

//easyjson:json
type OmitEmptyDefault struct {
	Field string
	Str   string
	Str1  string `json:"s,!omitempty"`
	Str2  string `json:",!omitempty"`
}

var omitEmptyDefaultValue = OmitEmptyDefault{Field: "test"}
var omitEmptyDefaultString = `{"Field":"test","s":"","Str2":""}`
