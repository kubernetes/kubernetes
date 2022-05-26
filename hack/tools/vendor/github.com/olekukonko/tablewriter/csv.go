// Copyright 2014 Oleku Konko All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

// This module is a Table Writer  API for the Go Programming Language.
// The protocols were written in pure Go and works on windows and unix systems

package tablewriter

import (
	"encoding/csv"
	"io"
	"os"
)

// Start A new table by importing from a CSV file
// Takes io.Writer and csv File name
func NewCSV(writer io.Writer, fileName string, hasHeader bool) (*Table, error) {
	file, err := os.Open(fileName)
	if err != nil {
		return &Table{}, err
	}
	defer file.Close()
	csvReader := csv.NewReader(file)
	t, err := NewCSVReader(writer, csvReader, hasHeader)
	return t, err
}

//  Start a New Table Writer with csv.Reader
// This enables customisation such as reader.Comma = ';'
// See http://golang.org/src/pkg/encoding/csv/reader.go?s=3213:3671#L94
func NewCSVReader(writer io.Writer, csvReader *csv.Reader, hasHeader bool) (*Table, error) {
	t := NewWriter(writer)
	if hasHeader {
		// Read the first row
		headers, err := csvReader.Read()
		if err != nil {
			return &Table{}, err
		}
		t.SetHeader(headers)
	}
	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return &Table{}, err
		}
		t.Append(record)
	}
	return t, nil
}
