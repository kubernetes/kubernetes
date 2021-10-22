package mergo

import (
	"testing"
)

type Student struct {
	Name  string
	Books []string
}

var testData = []struct {
	S1            Student
	S2            Student
	ExpectedSlice []string
}{
	{Student{"Jack", []string{"a", "B"}}, Student{"Tom", []string{"1"}}, []string{"a", "B"}},
	{Student{"Jack", []string{"a", "B"}}, Student{"Tom", []string{}}, []string{"a", "B"}},
	{Student{"Jack", []string{}}, Student{"Tom", []string{"1"}}, []string{"1"}},
	{Student{"Jack", []string{}}, Student{"Tom", []string{}}, []string{}},
}

func TestIssue64MergeSliceWithOverride(t *testing.T) {
	for _, data := range testData {
		err := Merge(&data.S2, data.S1, WithOverride)
		if err != nil {
			t.Errorf("Error while merging %s", err)
		}
		if len(data.S2.Books) != len(data.ExpectedSlice) {
			t.Fatalf("Got %d elements in slice, but expected %d", len(data.S2.Books), len(data.ExpectedSlice))
		}
		for i, val := range data.S2.Books {
			if val != data.ExpectedSlice[i] {
				t.Fatalf("Expected %s, but got %s while merging slice with override", data.ExpectedSlice[i], val)
			}
		}
	}
}
