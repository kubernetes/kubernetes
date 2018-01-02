package warnings_test

import (
	"errors"
	"reflect"
	"testing"

	w "gopkg.in/warnings.v0"
)

var _ error = List{}

type warn string

func (w warn) Error() string { return string(w) }

func warning(s string) error { return warn(s) }
func fatal(s string) error   { return errors.New(s) }

func isFatal(err error) bool {
	_, ok := err.(warn)
	return !ok
}

func omitNils(errs []error) []error {
	if errs == nil {
		return nil
	}
	res := []error{}
	for _, err := range errs {
		if err != nil {
			res = append(res, err)
		}
	}
	return res
}

var collectorTests = [...]struct {
	collector w.Collector
	warnings  []error
	fatal     error
}{
	{w.Collector{IsFatal: isFatal}, nil, nil},
	{w.Collector{IsFatal: isFatal}, nil, fatal("1f")},
	{w.Collector{IsFatal: isFatal}, []error{warning("1w")}, nil},
	{w.Collector{IsFatal: isFatal}, []error{warning("1w")}, fatal("2f")},
	{w.Collector{IsFatal: isFatal}, []error{warning("1w"), warning("2w")}, fatal("3f")},
	{w.Collector{IsFatal: isFatal}, []error{warning("1w"), nil, warning("2w")}, fatal("3f")},
	{w.Collector{IsFatal: isFatal, FatalWithWarnings: true}, []error{warning("1w")}, fatal("2f")},
}

func TestCollector(t *testing.T) {
	for _, tt := range collectorTests {
		c := tt.collector
		for _, warn := range tt.warnings {
			err := c.Collect(warn)
			if err != nil {
				t.Fatalf("Collect(%v) = %v; want nil", warn, err)
			}
		}
		if tt.fatal != nil {
			err := c.Collect(tt.fatal)
			if err == nil || w.FatalOnly(err) != tt.fatal {
				t.Fatalf("Collect(%v) = %v; want fatal %v", tt.fatal,
					err, tt.fatal)
			}
		}
		err := c.Done()
		if tt.fatal != nil {
			if err == nil || w.FatalOnly(err) != tt.fatal {
				t.Fatalf("Done() = %v; want fatal %v", err, tt.fatal)
			}
		}
		if tt.fatal == nil || c.FatalWithWarnings {
			warns := w.WarningsOnly(err)
			if !reflect.DeepEqual(warns, omitNils(tt.warnings)) {
				t.Fatalf("Done() = %v; want warnings %v", err,
					omitNils(tt.warnings))
			}
		}
	}
}
