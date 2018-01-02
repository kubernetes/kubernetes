package test

// DoubleEmbedded1 TEST ONLY
type DoubleEmbedded1 struct {
	F1 int32
}

// Embedded1 TEST ONLY
type Embedded1 struct {
	DoubleEmbedded1
}

// DoubleEmbedded2 TEST ONLY
type DoubleEmbedded2 struct {
	F1 int32 `json:"F1"`
}

// Embedded2 TEST ONLY
type Embedded2 struct {
	DoubleEmbedded2
}

type typeForTest struct {
	Embedded1
	Embedded2
}
