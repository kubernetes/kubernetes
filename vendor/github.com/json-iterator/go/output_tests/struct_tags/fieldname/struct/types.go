package test

// S1 TEST ONLY
type S1 struct {
	S1F string
}

// S2 TEST ONLY
type S2 struct {
	S2F string
}

// S3 TEST ONLY
type S3 struct {
	S3F string
}

// S4 TEST ONLY
type S4 struct {
	S4F string
}

// S5 TEST ONLY
type S5 struct {
	S5F string
}

// S6 TEST ONLY
type S6 struct {
	S6F string
}

type typeForTest struct {
	F1 S1 `json:"F1"`
	F2 S2 `json:"f2"`
	F3 S3 `json:"-"`
	F4 S4 `json:"-,"`
	F5 S5 `json:","`
	F6 S6 `json:""`
}
