/*Basic command line interface for debug and testing purposes.

Examples:

Only print the AST for the expression:

    jp.go -ast "foo.bar.baz"

Evaluate the JMESPath expression against JSON data from a file:

    jp.go -input /tmp/data.json "foo.bar.baz"

This program can also be used as an executable to the jp-compliance
runner (github.com/jmespath/jmespath.test).

*/
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
)

import (
	"encoding/json"

	"github.com/jmespath/go-jmespath"
)

func errMsg(msg string, a ...interface{}) int {
	fmt.Fprintf(os.Stderr, msg, a...)
	fmt.Fprintln(os.Stderr)
	return 1
}

func run() int {

	astOnly := flag.Bool("ast", false, "Print the AST for the input expression and exit.")
	inputFile := flag.String("input", "", "Filename containing JSON data to search. If not provided, data is read from stdin.")

	flag.Parse()
	args := flag.Args()
	if len(args) != 1 {
		fmt.Fprintf(os.Stderr, "Usage:\n\n")
		flag.PrintDefaults()
		return errMsg("\nError: expected a single argument (the JMESPath expression).")
	}

	expression := args[0]
	parser := jmespath.NewParser()
	parsed, err := parser.Parse(expression)
	if err != nil {
		if syntaxError, ok := err.(jmespath.SyntaxError); ok {
			return errMsg("%s\n%s\n", syntaxError, syntaxError.HighlightLocation())
		}
		return errMsg("%s", err)
	}
	if *astOnly {
		fmt.Println("")
		fmt.Printf("%s\n", parsed)
		return 0
	}

	var inputData []byte
	if *inputFile != "" {
		inputData, err = ioutil.ReadFile(*inputFile)
		if err != nil {
			return errMsg("Error loading file %s: %s", *inputFile, err)
		}
	} else {
		// If an input data file is not provided then we read the
		// data from stdin.
		inputData, err = ioutil.ReadAll(os.Stdin)
		if err != nil {
			return errMsg("Error reading from stdin: %s", err)
		}
	}
	var data interface{}
	json.Unmarshal(inputData, &data)
	result, err := jmespath.Search(expression, data)
	if err != nil {
		return errMsg("Error executing expression: %s", err)
	}
	toJSON, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return errMsg("Error serializing result to JSON: %s", err)
	}
	fmt.Println(string(toJSON))
	return 0
}

func main() {
	os.Exit(run())
}
