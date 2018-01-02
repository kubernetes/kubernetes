// short-regexp is a command-line utility that reads strings from standard input
// (one per line) and outputs a regexp that matches only those strings.
package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"vbom.ml/util"
)

func main() {
	data, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %s", err)
		os.Exit(1)
	}
	lines := strings.Split(string(data), "\n")
	// Remove trailing empty line if present
	if N := len(lines); N > 0 && lines[N-1] == "" {
		lines = lines[:N-1]
	}
	os.Stdout.WriteString(util.ShortRegexpString(lines...))
}
