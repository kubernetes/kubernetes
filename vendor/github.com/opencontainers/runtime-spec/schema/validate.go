package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/xeipuuv/gojsonschema"
)

const usage = `Validate is used to check document with specified schema.
You can use validate in following ways:

   1.specify document file as an argument
      validate <schema.json> <document.json>

   2.pass document content through a pipe
      cat <document.json> | validate <schema.json>

   3.input document content manually, ended with ctrl+d(or your self-defined EOF keys)
      validate <schema.json>
      [INPUT DOCUMENT CONTENT HERE]
`

func main() {
	nargs := len(os.Args[1:])
	if nargs == 0 || nargs > 2 {
		fmt.Printf("ERROR: invalid arguments number\n\n%s\n", usage)
		os.Exit(1)
	}

	if os.Args[1] == "help" ||
		os.Args[1] == "--help" ||
		os.Args[1] == "-h" {
		fmt.Printf("%s\n", usage)
		os.Exit(1)
	}

	schemaPath := os.Args[1]
	if !strings.Contains(schemaPath, "://") {
		var err error
		schemaPath, err = formatFilePath(schemaPath)
		if err != nil {
			fmt.Printf("ERROR: invalid schema-file path: %s\n", err)
			os.Exit(1)
		}
		schemaPath = "file://" + schemaPath
	}

	schemaLoader := gojsonschema.NewReferenceLoader(schemaPath)

	var documentLoader gojsonschema.JSONLoader

	if nargs > 1 {
		documentPath, err := formatFilePath(os.Args[2])
		if err != nil {
			fmt.Printf("ERROR: invalid document-file path: %s\n", err)
			os.Exit(1)
		}
		documentLoader = gojsonschema.NewReferenceLoader("file://" + documentPath)
	} else {
		documentBytes, err := ioutil.ReadAll(os.Stdin)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		documentString := string(documentBytes)
		documentLoader = gojsonschema.NewStringLoader(documentString)
	}

	result, err := gojsonschema.Validate(schemaLoader, documentLoader)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	if result.Valid() {
		fmt.Printf("The document is valid\n")
	} else {
		fmt.Printf("The document is not valid. see errors :\n")
		for _, desc := range result.Errors() {
			fmt.Printf("- %s\n", desc)
		}
		os.Exit(1)
	}
}

func formatFilePath(path string) (string, error) {
	if _, err := os.Stat(path); err != nil {
		return "", err
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	return absPath, nil
}
