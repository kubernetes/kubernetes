package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/pelletier/go-toml"
)

func main() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, `tomljson can be used in two ways:
Writing to STDIN and reading from STDOUT:
  cat file.toml | tomljson > file.json

Reading from a file name:
  tomljson file.toml
`)
	}
	flag.Parse()
	os.Exit(processMain(flag.Args(), os.Stdin, os.Stdout, os.Stderr))
}

func processMain(files []string, defaultInput io.Reader, output io.Writer, errorOutput io.Writer) int {
	// read from stdin and print to stdout
	inputReader := defaultInput

	if len(files) > 0 {
		var err error
		inputReader, err = os.Open(files[0])
		if err != nil {
			printError(err, errorOutput)
			return -1
		}
	}
	s, err := reader(inputReader)
	if err != nil {
		printError(err, errorOutput)
		return -1
	}
	io.WriteString(output, s+"\n")
	return 0
}

func printError(err error, output io.Writer) {
	io.WriteString(output, err.Error()+"\n")
}

func reader(r io.Reader) (string, error) {
	tree, err := toml.LoadReader(r)
	if err != nil {
		return "", err
	}
	return mapToJSON(tree)
}

func mapToJSON(tree *toml.TomlTree) (string, error) {
	treeMap := tree.ToMap()
	bytes, err := json.MarshalIndent(treeMap, "", "  ")
	if err != nil {
		return "", err
	}
	return string(bytes[:]), nil
}
