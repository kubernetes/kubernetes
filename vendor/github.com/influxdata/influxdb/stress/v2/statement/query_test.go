package statement

import (
	"testing"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

func TestQuerySetID(t *testing.T) {
	e := newTestQuery()
	newID := "oaijnifo"
	e.SetID(newID)
	if e.StatementID != newID {
		t.Errorf("Expected: %v\nGot: %v\n", newID, e.StatementID)
	}
}

func TestQueryRun(t *testing.T) {
	i := newTestQuery()
	s, packageCh, _ := stressClient.NewTestStressTest()
	// Listen to the other side of the directiveCh
	go func() {
		for pkg := range packageCh {
			if i.TemplateString != string(pkg.Body) {
				t.Fail()
			}
			pkg.Tracer.Done()
		}
	}()
	i.Run(s)
}

func newTestQuery() *QueryStatement {
	return &QueryStatement{
		StatementID:    "foo_ID",
		Name:           "foo_name",
		TemplateString: "SELECT count(value) FROM cpu",
		Args:           []string{},
		Count:          5,
		Tracer:         stressClient.NewTracer(map[string]string{}),
	}
}
