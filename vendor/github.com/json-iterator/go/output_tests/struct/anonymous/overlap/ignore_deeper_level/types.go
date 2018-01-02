package test

// DoubleEmbedded TEST ONLY
type DoubleEmbedded struct {
	F1 int32 `json:"F1"`
}

// Embedded1 TEST ONLY
type Embedded1 struct {
	DoubleEmbedded
	F1 int32
}

// Embedded2 TEST ONLY
type Embedded2 struct {
	F1 int32 `json:"F1"`
	DoubleEmbedded
}

type typeForTest struct {
	Embedded1
	Embedded2
}
