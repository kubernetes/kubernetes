package test

// E1 TEST ONLY
type E1 struct {
	F1 int32
}

// E2 TEST ONLY
type E2 struct {
	F2 string
}

type typeForTest struct {
	E1
	E2
	F1 string
}
