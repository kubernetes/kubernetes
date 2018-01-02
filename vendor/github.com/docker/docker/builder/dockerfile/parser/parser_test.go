package parser

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testDir = "testfiles"
const negativeTestDir = "testfiles-negative"
const testFileLineInfo = "testfile-line/Dockerfile"

func getDirs(t *testing.T, dir string) []string {
	f, err := os.Open(dir)
	require.NoError(t, err)
	defer f.Close()

	dirs, err := f.Readdirnames(0)
	require.NoError(t, err)
	return dirs
}

func TestParseErrorCases(t *testing.T) {
	for _, dir := range getDirs(t, negativeTestDir) {
		dockerfile := filepath.Join(negativeTestDir, dir, "Dockerfile")

		df, err := os.Open(dockerfile)
		require.NoError(t, err, dockerfile)
		defer df.Close()

		_, err = Parse(df)
		assert.Error(t, err, dockerfile)
	}
}

func TestParseCases(t *testing.T) {
	for _, dir := range getDirs(t, testDir) {
		dockerfile := filepath.Join(testDir, dir, "Dockerfile")
		resultfile := filepath.Join(testDir, dir, "result")

		df, err := os.Open(dockerfile)
		require.NoError(t, err, dockerfile)
		defer df.Close()

		result, err := Parse(df)
		require.NoError(t, err, dockerfile)

		content, err := ioutil.ReadFile(resultfile)
		require.NoError(t, err, resultfile)

		if runtime.GOOS == "windows" {
			// CRLF --> CR to match Unix behavior
			content = bytes.Replace(content, []byte{'\x0d', '\x0a'}, []byte{'\x0a'}, -1)
		}
		assert.Equal(t, result.AST.Dump()+"\n", string(content), "In "+dockerfile)
	}
}

func TestParseWords(t *testing.T) {
	tests := []map[string][]string{
		{
			"input":  {"foo"},
			"expect": {"foo"},
		},
		{
			"input":  {"foo bar"},
			"expect": {"foo", "bar"},
		},
		{
			"input":  {"foo\\ bar"},
			"expect": {"foo\\ bar"},
		},
		{
			"input":  {"foo=bar"},
			"expect": {"foo=bar"},
		},
		{
			"input":  {"foo bar 'abc xyz'"},
			"expect": {"foo", "bar", "'abc xyz'"},
		},
		{
			"input":  {`foo bar "abc xyz"`},
			"expect": {"foo", "bar", `"abc xyz"`},
		},
		{
			"input":  {"àöû"},
			"expect": {"àöû"},
		},
		{
			"input":  {`föo bàr "âbc xÿz"`},
			"expect": {"föo", "bàr", `"âbc xÿz"`},
		},
	}

	for _, test := range tests {
		words := parseWords(test["input"][0], NewDefaultDirective())
		assert.Equal(t, test["expect"], words)
	}
}

func TestParseIncludesLineNumbers(t *testing.T) {
	df, err := os.Open(testFileLineInfo)
	require.NoError(t, err)
	defer df.Close()

	result, err := Parse(df)
	require.NoError(t, err)

	ast := result.AST
	assert.Equal(t, 5, ast.StartLine)
	assert.Equal(t, 31, ast.endLine)
	assert.Len(t, ast.Children, 3)
	expected := [][]int{
		{5, 5},
		{11, 12},
		{17, 31},
	}
	for i, child := range ast.Children {
		msg := fmt.Sprintf("Child %d", i)
		assert.Equal(t, expected[i], []int{child.StartLine, child.endLine}, msg)
	}
}

func TestParseWarnsOnEmptyContinutationLine(t *testing.T) {
	dockerfile := bytes.NewBufferString(`
FROM alpine:3.6

RUN something \

    following \

    more

RUN another \

    thing
	`)

	result, err := Parse(dockerfile)
	require.NoError(t, err)
	warnings := result.Warnings
	assert.Len(t, warnings, 3)
	assert.Contains(t, warnings[0], "Empty continuation line found in")
	assert.Contains(t, warnings[0], "RUN something     following     more")
	assert.Contains(t, warnings[1], "RUN another     thing")
	assert.Contains(t, warnings[2], "will become errors in a future release")
}
