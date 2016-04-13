package cmd

import (
	"fmt"
	"os"
	"testing"
)

var _ = fmt.Println
var _ = os.Stderr

func checkGuess(t *testing.T, wd, input, expected string) {
	testWd = wd
	inputPath = input
	guessProjectPath()

	if projectPath != expected {
		t.Errorf("Unexpected Project Path. \n Got: %q\nExpected: %q\n", projectPath, expected)
	}

	reset()
}

func reset() {
	testWd = ""
	inputPath = ""
	projectPath = ""
}

func TestProjectPath(t *testing.T) {
	checkGuess(t, "", "github.com/spf13/hugo", getSrcPath()+"github.com/spf13/hugo")
	checkGuess(t, "", "spf13/hugo", getSrcPath()+"github.com/spf13/hugo")
	checkGuess(t, "", "/bar/foo", "/bar/foo")
	checkGuess(t, "/bar/foo", "baz", "/bar/foo/baz")
	checkGuess(t, "/bar/foo/cmd", "", "/bar/foo")
	checkGuess(t, "/bar/foo/command", "", "/bar/foo")
	checkGuess(t, "/bar/foo/commands", "", "/bar/foo")
	checkGuess(t, "github.com/spf13/hugo/../hugo", "", "github.com/spf13/hugo")
}
