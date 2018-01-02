package test

type typeA1 string
type typeA2 [4]typeA1

type typeForTest struct {
	F1 [4]typeA1
	F2 typeA2
}
