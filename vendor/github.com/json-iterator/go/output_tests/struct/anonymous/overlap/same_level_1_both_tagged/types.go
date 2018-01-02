package test

// Embedded1 TEST ONLY
type Embedded1 struct {
	F1 int32 `json:"F1"`
}

// Embedded2 TEST ONLY
type Embedded2 struct {
	F1 int32 `json:"F1"`
}

type typeForTest struct {
	Embedded1
	Embedded2
}
