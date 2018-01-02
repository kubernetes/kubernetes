package statement

import (
	"testing"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

func TestExecSetID(t *testing.T) {
	e := newTestExec()
	newID := "oaijnifo"
	e.SetID(newID)
	if e.StatementID != newID {
		t.Errorf("Expected: %v\nGot: %v\n", newID, e.StatementID)
	}
}

func TestExecRun(t *testing.T) {
	e := newTestExec()
	s, _, _ := stressClient.NewTestStressTest()
	e.Run(s)
	if e == nil {
		t.Fail()
	}
}

func TestExecReport(t *testing.T) {
	e := newTestExec()
	s, _, _ := stressClient.NewTestStressTest()
	rep := e.Report(s)
	if rep != "" {
		t.Fail()
	}
}

func newTestExec() *ExecStatement {
	return &ExecStatement{
		StatementID: "fooID",
		Script:      "fooscript.txt",
	}
}
