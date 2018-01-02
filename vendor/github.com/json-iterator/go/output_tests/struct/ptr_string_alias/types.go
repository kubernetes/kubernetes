package test

type typeA1 string
type typeA2 *string

type typeForTest struct {
	F1 *typeA1
	F2 typeA2
	F3 *typeA2
}
