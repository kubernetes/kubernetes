package gofmt

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"io/ioutil"
)

func Run(filename string, needSimplify bool) ([]byte, error) {
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	initParserMode()

	file, sourceAdj, indentAdj, err := parse(fileSet, filename, src, false)
	if err != nil {
		return nil, err
	}

	ast.SortImports(fileSet, file)

	if needSimplify {
		simplify(file)
	}

	res, err := format(fileSet, file, sourceAdj, indentAdj, src, printer.Config{Mode: printerMode, Tabwidth: tabWidth})
	if err != nil {
		return nil, err
	}

	if bytes.Equal(src, res) {
		return nil, nil
	}

	// formatting has changed
	data, err := diff(src, res, filename)
	if err != nil {
		return nil, fmt.Errorf("error computing diff: %s", err)
	}

	return data, nil
}
