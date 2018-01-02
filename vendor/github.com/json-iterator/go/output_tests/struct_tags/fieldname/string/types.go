package test

// E TEST ONLY
type E struct {
	E1 string
}

type typeForTest struct {
	F1 string `json:"F1"`
	F2 string `json:"f2"`
	F3 string `json:"-"`
	F4 string `json:"-,"`
	F5 string `json:","`
	F6 string `json:""`
	E  `json:"e"`
}
