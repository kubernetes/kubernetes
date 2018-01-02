package test

type typeForTest struct {
	F struct{} `json:"f,omitempty"` // omitempty is meaningless here
}
