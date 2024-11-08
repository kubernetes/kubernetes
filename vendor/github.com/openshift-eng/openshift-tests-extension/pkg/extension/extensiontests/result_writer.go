package extensiontests

import (
	"encoding/json"
	"fmt"
	"io"
)

type ResultWriter interface {
	Write(result *ExtensionTestResult)
	Flush()
}

type NullResultWriter struct{}

func (NullResultWriter) Write(*ExtensionTestResult) {}
func (NullResultWriter) Flush()                     {}

type ResultFormat string

var (
	JSON  ResultFormat = "json"
	JSONL ResultFormat = "jsonl"
)

type JSONResultWriter struct {
	out     io.Writer
	format  ResultFormat
	results ExtensionTestResults
}

func NewResultWriter(out io.Writer, format ResultFormat) (*JSONResultWriter, error) {
	switch format {
	case JSON, JSONL:
	// do nothing
	default:
		return nil, fmt.Errorf("unsupported result format: %s", format)
	}

	return &JSONResultWriter{
		out:    out,
		format: format,
	}, nil
}

func (w *JSONResultWriter) Write(result *ExtensionTestResult) {
	switch w.format {
	case JSONL:
		// JSONL gets written to out as we get the items
		data, err := json.Marshal(result)
		if err != nil {
			panic(err)
		}
		fmt.Fprintf(w.out, "%s\n", string(data))
	case JSON:
		w.results = append(w.results, result)
	}
}

func (w *JSONResultWriter) Flush() {
	switch w.format {
	case JSONL:
	// we already wrote it out
	case JSON:
		data, err := json.MarshalIndent(w.results, "", "  ")
		if err != nil {
			panic(err)
		}
		fmt.Fprintf(w.out, "%s\n", string(data))
	}
}
