// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

type translationSample struct {
	str, out string
}

func titleize(s string) string { return strings.ToTitle(s[:1]) + lower(s[1:]) }

func TestToGoName(t *testing.T) {
	samples := []translationSample{
		{"sample text", "SampleText"},
		{"sample-text", "SampleText"},
		{"sample_text", "SampleText"},
		{"sampleText", "SampleText"},
		{"sample 2 Text", "Sample2Text"},
		{"findThingById", "FindThingByID"},
		{"日本語sample 2 Text", "X日本語sample2Text"},
		{"日本語findThingById", "X日本語findThingByID"},
	}

	for k := range commonInitialisms {
		samples = append(samples,
			translationSample{"sample " + lower(k) + " text", "Sample" + k + "Text"},
			translationSample{"sample-" + lower(k) + "-text", "Sample" + k + "Text"},
			translationSample{"sample_" + lower(k) + "_text", "Sample" + k + "Text"},
			translationSample{"sample" + titleize(k) + "Text", "Sample" + k + "Text"},
			translationSample{"sample " + lower(k), "Sample" + k},
			translationSample{"sample-" + lower(k), "Sample" + k},
			translationSample{"sample_" + lower(k), "Sample" + k},
			translationSample{"sample" + titleize(k), "Sample" + k},
			translationSample{"sample " + titleize(k) + " text", "Sample" + k + "Text"},
			translationSample{"sample-" + titleize(k) + "-text", "Sample" + k + "Text"},
			translationSample{"sample_" + titleize(k) + "_text", "Sample" + k + "Text"},
		)
	}

	for _, sample := range samples {
		assert.Equal(t, sample.out, ToGoName(sample.str))
	}
}

func TestContainsStringsCI(t *testing.T) {
	list := []string{"hello", "world", "and", "such"}

	assert.True(t, ContainsStringsCI(list, "hELLo"))
	assert.True(t, ContainsStringsCI(list, "world"))
	assert.True(t, ContainsStringsCI(list, "AND"))
	assert.False(t, ContainsStringsCI(list, "nuts"))
}

func TestSplitByFormat(t *testing.T) {
	expected := []string{"one", "two", "three"}
	for _, fmt := range []string{"csv", "pipes", "tsv", "ssv", "multi"} {

		var actual []string
		switch fmt {
		case "multi":
			assert.Nil(t, SplitByFormat("", fmt))
			assert.Nil(t, SplitByFormat("blah", fmt))
		case "ssv":
			actual = SplitByFormat(strings.Join(expected, " "), fmt)
			assert.EqualValues(t, expected, actual)
		case "pipes":
			actual = SplitByFormat(strings.Join(expected, "|"), fmt)
			assert.EqualValues(t, expected, actual)
		case "tsv":
			actual = SplitByFormat(strings.Join(expected, "\t"), fmt)
			assert.EqualValues(t, expected, actual)
		default:
			actual = SplitByFormat(strings.Join(expected, ","), fmt)
			assert.EqualValues(t, expected, actual)
		}
	}
}

func TestJoinByFormat(t *testing.T) {
	for _, fmt := range []string{"csv", "pipes", "tsv", "ssv", "multi"} {

		lval := []string{"one", "two", "three"}
		var expected []string
		switch fmt {
		case "multi":
			expected = lval
		case "ssv":
			expected = []string{strings.Join(lval, " ")}
		case "pipes":
			expected = []string{strings.Join(lval, "|")}
		case "tsv":
			expected = []string{strings.Join(lval, "\t")}
		default:
			expected = []string{strings.Join(lval, ",")}
		}
		assert.Nil(t, JoinByFormat(nil, fmt))
		assert.EqualValues(t, expected, JoinByFormat(lval, fmt))
	}
}

func TestToFileName(t *testing.T) {
	samples := []translationSample{
		{"SampleText", "sample_text"},
		{"FindThingByID", "find_thing_by_id"},
	}

	for k := range commonInitialisms {
		samples = append(samples,
			translationSample{"Sample" + k + "Text", "sample_" + lower(k) + "_text"},
		)
	}

	for _, sample := range samples {
		assert.Equal(t, sample.out, ToFileName(sample.str))
	}
}

func TestToCommandName(t *testing.T) {
	samples := []translationSample{
		{"SampleText", "sample-text"},
		{"FindThingByID", "find-thing-by-id"},
	}

	for k := range commonInitialisms {
		samples = append(samples,
			translationSample{"Sample" + k + "Text", "sample-" + lower(k) + "-text"},
		)
	}

	for _, sample := range samples {
		assert.Equal(t, sample.out, ToCommandName(sample.str))
	}
}

func TestToHumanName(t *testing.T) {
	samples := []translationSample{
		{"SampleText", "sample text"},
		{"FindThingByID", "find thing by ID"},
	}

	for k := range commonInitialisms {
		samples = append(samples,
			translationSample{"Sample" + k + "Text", "sample " + k + " text"},
		)
	}

	for _, sample := range samples {
		assert.Equal(t, sample.out, ToHumanNameLower(sample.str))
	}
}

func TestToJSONName(t *testing.T) {
	samples := []translationSample{
		{"SampleText", "sampleText"},
		{"FindThingByID", "findThingById"},
	}

	for k := range commonInitialisms {
		samples = append(samples,
			translationSample{"Sample" + k + "Text", "sample" + titleize(k) + "Text"},
		)
	}

	for _, sample := range samples {
		assert.Equal(t, sample.out, ToJSONName(sample.str))
	}
}

type SimpleZeroes struct {
	ID   string
	Name string
}
type ZeroesWithTime struct {
	Time time.Time
}

func TestIsZero(t *testing.T) {
	var strs [5]string
	var strss []string
	var a int
	var b int8
	var c int16
	var d int32
	var e int64
	var f uint
	var g uint8
	var h uint16
	var i uint32
	var j uint64
	var k map[string]string
	var l interface{}
	var m *SimpleZeroes
	var n string
	var o SimpleZeroes
	var p ZeroesWithTime
	var q time.Time
	data := []struct {
		Data     interface{}
		Expected bool
	}{
		{a, true},
		{b, true},
		{c, true},
		{d, true},
		{e, true},
		{f, true},
		{g, true},
		{h, true},
		{i, true},
		{j, true},
		{k, true},
		{l, true},
		{m, true},
		{n, true},
		{o, true},
		{p, true},
		{q, true},
		{strss, true},
		{strs, true},
		{"", true},
		{nil, true},
		{1, false},
		{0, true},
		{int8(1), false},
		{int8(0), true},
		{int16(1), false},
		{int16(0), true},
		{int32(1), false},
		{int32(0), true},
		{int64(1), false},
		{int64(0), true},
		{uint(1), false},
		{uint(0), true},
		{uint8(1), false},
		{uint8(0), true},
		{uint16(1), false},
		{uint16(0), true},
		{uint32(1), false},
		{uint32(0), true},
		{uint64(1), false},
		{uint64(0), true},
		{0.0, true},
		{0.1, false},
		{float32(0.0), true},
		{float32(0.1), false},
		{float64(0.0), true},
		{float64(0.1), false},
		{[...]string{}, true},
		{[...]string{"hello"}, false},
		{[]string(nil), true},
		{[]string{"a"}, false},
	}

	for _, it := range data {
		assert.Equal(t, it.Expected, IsZero(it.Data), fmt.Sprintf("%#v", it.Data))
	}
}
