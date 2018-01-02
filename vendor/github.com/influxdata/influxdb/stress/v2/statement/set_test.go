package statement

import (
	"fmt"
	"testing"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

func TestSetSetID(t *testing.T) {
	e := newTestSet("database", "foo")
	newID := "oaijnifo"
	e.SetID(newID)
	if e.StatementID != newID {
		t.Errorf("Expected: %v\nGot: %v\n", newID, e.StatementID)
	}
}

func TestSetRun(t *testing.T) {
	properties := []string{
		"precision",
		"startdate",
		"batchsize",
		"resultsaddress",
		"testname",
		"addresses",
		"writeinterval",
		"queryinterval",
		"database",
		"writeconcurrency",
		"queryconcurrency",
	}
	for _, prop := range properties {
		testSetRunUtl(t, prop, "1")
	}
}

func testSetRunUtl(t *testing.T, property string, value string) {
	i := newTestSet(property, value)
	s, _, directiveCh := stressClient.NewTestStressTest()
	// Listen to the other side of the directiveCh
	go func() {
		for d := range directiveCh {
			if i.Var != d.Property {
				t.Errorf("wrong property sent to stressClient\n  expected: %v\n got: %v\n", i.Var, d.Property)
			}
			if i.Value != d.Value {
				t.Errorf("wrong value sent to stressClient\n  expected: %v\n  got: %v\n", i.Value, d.Value)
			}
			d.Tracer.Done()
		}
	}()
	// Run the statement
	i.Run(s)
	// Check the result
	switch i.Var {
	case "precision":
		if i.Value != s.Precision {
			t.Errorf("Failed to set %v\n", i.Var)
		}
	case "startdate":
		if i.Value != s.StartDate {
			t.Errorf("Failed to set %v\n", i.Var)
		}
	case "batchsize":
		if parseInt(i.Value) != s.BatchSize {
			t.Errorf("Failed to set %v\n", i.Var)
		}
	// TODO: Actually test this
	case "resultsaddress":
	default:
	}
}

func TestSetReport(t *testing.T) {
	set := newTestSet("this", "that")
	s, _, _ := stressClient.NewTestStressTest()
	rpt := set.Report(s)
	expected := fmt.Sprintf("SET %v = '%v'", set.Var, set.Value)
	if rpt != expected {
		t.Errorf("expected: %v\ngot: %v\n", expected, rpt)
	}
}

func newTestSet(toSet, value string) *SetStatement {
	return &SetStatement{
		Var:         toSet,
		Value:       value,
		Tracer:      stressClient.NewTracer(make(map[string]string)),
		StatementID: "fooID",
	}
}
