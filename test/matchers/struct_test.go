/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package matchers

import (
	"testing"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/types"
	"github.com/stretchr/testify/assert"
)

func TestStructMatcher(t *testing.T) {
	allFields := struct{ A, B string }{"a", "b"}
	missingFields := struct{ A string }{"a"}
	extraFields := struct{ A, B, C string }{"a", "b", "c"}
	emptyFields := struct{ A, B string }{}

	strict := StrictStruct(Fields{
		"B": gomega.Equal("b"),
		"A": gomega.Equal("a"),
	})
	strictFail := StrictStruct(Fields{
		"A": gomega.Equal("a"),
		"B": gomega.Equal("fail"),
	})
	strictIgnore := StrictStruct(Fields{
		"A": Ignore(),
		"B": Ignore(),
	})
	loose := LooseStruct(Fields{
		"B": gomega.Equal("b"),
		"A": gomega.Equal("a"),
	})
	looseFail := LooseStruct(Fields{
		"A": gomega.Equal("a"),
		"B": gomega.Equal("fail"),
	})

	tests := []struct {
		actual      interface{}
		matcher     types.GomegaMatcher
		expectMatch bool
		msg         string
	}{
		{allFields, strict, true, "StrictStruct should match all fields"},
		{missingFields, strict, false, "StrictStruct should fail with missing fields"},
		{extraFields, strict, false, "StrictStruct should fail with extra fields"},
		{allFields, strictFail, false, "StrictStruct should fail with fail"},
		{emptyFields, strictIgnore, true, "StrictStruct should handle empty fields"},
		{allFields, loose, true, "LooseStruct should match all fields"},
		{missingFields, loose, false, "LooseStruct should fail with missing fields"},
		{extraFields, loose, true, "LooseStruct should ignore extra fields"},
		{allFields, looseFail, false, "LooseStruct should fail with fail"},
	}

	for i, test := range tests {
		match, err := test.matcher.Match(test.actual)
		assert.NoError(t, err, "[%d] %s", i, test.msg)
		assert.Equal(t, test.expectMatch, match,
			"[%d] %s: %s", i, test.msg, test.matcher.FailureMessage(test.actual))
	}
}
