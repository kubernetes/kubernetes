// The package is intended for testing the openapi-gen API rule
// checker. The API rule violations are in format of:
//
// `{rule-name},{package},{type},{(optional) field}`
//
// The checker should sort the violations before
// reporting to a file or stderr.
//
// We have the dummytype package separately from the listtype
// package to test the sorting behavior on package level, e.g.
//
//   -i "./testdata/listtype,./testdata/dummytype"
//   -i "./testdata/dummytype,./testdata/listtype"
//
// The violations from dummytype should always come first in
// report.

package dummytype

// +k8s:openapi-gen=true
type Waldo struct {
	First  int
	Second string
}
