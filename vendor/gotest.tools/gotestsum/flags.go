package main

import (
	"encoding/csv"
	"strings"

	"github.com/pkg/errors"
	"gotest.tools/gotestsum/testjson"
)

type noSummaryValue struct {
	value testjson.Summary
}

func newNoSummaryValue() *noSummaryValue {
	return &noSummaryValue{value: testjson.SummarizeAll}
}

func readAsCSV(val string) ([]string, error) {
	if val == "" {
		return nil, nil
	}
	return csv.NewReader(strings.NewReader(val)).Read()
}

func (s *noSummaryValue) Set(val string) error {
	v, err := readAsCSV(val)
	if err != nil {
		return err
	}
	for _, item := range v {
		summary, ok := testjson.NewSummary(item)
		if !ok {
			return errors.Errorf("value must be one or more of: %s",
				testjson.SummarizeAll.String())
		}
		s.value -= summary
	}
	return nil
}

func (s *noSummaryValue) Type() string {
	return "summary"
}

func (s *noSummaryValue) String() string {
	// flip all the bits, since the flag value is the negative of what is stored
	return (testjson.SummarizeAll ^ s.value).String()
}
