package extensiontests

import (
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/openshift-eng/openshift-tests-extension/pkg/junit"
)

type ResultWriter interface {
	Write(result *ExtensionTestResult)
	Flush() error
}

type NullResultWriter struct{}

func (NullResultWriter) Write(*ExtensionTestResult) {}
func (NullResultWriter) Flush() error               { return nil }

type CompositeResultWriter struct {
	writers []ResultWriter
}

func NewCompositeResultWriter(writers ...ResultWriter) *CompositeResultWriter {
	return &CompositeResultWriter{
		writers: writers,
	}
}

func (w *CompositeResultWriter) AddWriter(writer ResultWriter) {
	w.writers = append(w.writers, writer)
}

func (w *CompositeResultWriter) Write(res *ExtensionTestResult) {
	for _, writer := range w.writers {
		writer.Write(res)
	}
}

func (w *CompositeResultWriter) Flush() error {
	var errs []error
	for _, writer := range w.writers {
		if err := writer.Flush(); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

type JUnitResultWriter struct {
	lock      sync.Mutex
	testSuite *junit.TestSuite
	out       *os.File
	suiteName string
	path      string
	results   ExtensionTestResults
}

func NewJUnitResultWriter(path, suiteName string) (ResultWriter, error) {
	file, err := os.Create(path)
	if err != nil {
		return nil, err
	}

	return &JUnitResultWriter{
		testSuite: &junit.TestSuite{
			Name: suiteName,
		},
		out:       file,
		suiteName: suiteName,
		path:      path,
	}, nil
}

func (w *JUnitResultWriter) Write(res *ExtensionTestResult) {
	w.lock.Lock()
	defer w.lock.Unlock()
	w.results = append(w.results, res)
}

func (w *JUnitResultWriter) Flush() error {
	w.lock.Lock()
	defer w.lock.Unlock()
	data, err := xml.MarshalIndent(w.results.ToJUnit(w.suiteName), "", "    ")
	if err != nil {
		return fmt.Errorf("failed to marshal JUnit XML: %w", err)
	}
	if _, err := w.out.Write(data); err != nil {
		return err
	}
	if err := w.out.Close(); err != nil {
		return err
	}

	return nil
}

type ResultFormat string

var (
	JSON  ResultFormat = "json"
	JSONL ResultFormat = "jsonl"
)

type JSONResultWriter struct {
	lock    sync.Mutex
	out     io.Writer
	format  ResultFormat
	results ExtensionTestResults
}

func NewJSONResultWriter(out io.Writer, format ResultFormat) (*JSONResultWriter, error) {
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
	w.lock.Lock()
	defer w.lock.Unlock()
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

func (w *JSONResultWriter) Flush() error {
	w.lock.Lock()
	defer w.lock.Unlock()
	switch w.format {
	case JSONL:
	// we already wrote it out
	case JSON:
		data, err := json.MarshalIndent(w.results, "", "  ")
		if err != nil {
			return err
		}
		_, err = w.out.Write(data)
		return err
	}

	return nil
}
