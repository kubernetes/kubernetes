package test

type typeA1 int32

type typeA2 *int32

type typeForTest struct {
	F1 *typeA1
	F2 typeA2
	F3 *typeA2
}
