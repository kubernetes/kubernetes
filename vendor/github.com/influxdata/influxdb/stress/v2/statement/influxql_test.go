package statement

import (
	"testing"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

func TestInfluxQlSetID(t *testing.T) {
	e := newTestInfluxQl()
	newID := "oaijnifo"
	e.SetID(newID)
	if e.StatementID != newID {
		t.Errorf("Expected: %v\nGot: %v\n", newID, e.StatementID)
	}
}

func TestInfluxQlRun(t *testing.T) {
	e := newTestInfluxQl()
	s, packageCh, _ := stressClient.NewTestStressTest()
	go func() {
		for pkg := range packageCh {
			if pkg.T != stressClient.Query {
				t.Errorf("Expected package to be Query\nGot: %v", pkg.T)
			}
			if string(pkg.Body) != e.Query {
				t.Errorf("Expected query: %v\nGot: %v", e.Query, string(pkg.Body))
			}
			if pkg.StatementID != e.StatementID {
				t.Errorf("Expected statementID: %v\nGot: %v", e.StatementID, pkg.StatementID)
			}
			pkg.Tracer.Done()
		}
	}()
	e.Run(s)
}

func newTestInfluxQl() *InfluxqlStatement {
	return &InfluxqlStatement{
		Query:       "CREATE DATABASE foo",
		Tracer:      stressClient.NewTracer(make(map[string]string)),
		StatementID: "fooID",
	}
}
