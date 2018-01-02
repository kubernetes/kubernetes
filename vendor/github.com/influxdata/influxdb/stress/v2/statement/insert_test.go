package statement

import (
	"strings"
	"testing"

	"github.com/influxdata/influxdb/stress/v2/stress_client"
)

func TestInsertSetID(t *testing.T) {
	e := newTestInsert()
	newID := "oaijnifo"
	e.SetID(newID)
	if e.StatementID != newID {
		t.Errorf("Expected: %v\nGot: %v\n", newID, e.StatementID)
	}
}

func TestInsertRun(t *testing.T) {
	i := newTestInsert()
	s, packageCh, _ := stressClient.NewTestStressTest()
	// Listen to the other side of the directiveCh
	go func() {
		for pkg := range packageCh {
			countPoints := i.Timestamp.Count
			batchSize := s.BatchSize
			got := len(strings.Split(string(pkg.Body), "\n"))
			switch got {
			case countPoints % batchSize:
			case batchSize:
			default:
				t.Errorf("countPoints: %v\nbatchSize: %v\ngot: %v\n", countPoints, batchSize, got)
			}
			pkg.Tracer.Done()
		}
	}()
	i.Run(s)
}

func newTestInsert() *InsertStatement {
	return &InsertStatement{
		TestID:         "foo_test",
		StatementID:    "foo_ID",
		Name:           "foo_name",
		TemplateString: "cpu,%v %v %v",
		Timestamp:      newTestTimestamp(),
		Templates:      newTestTemplates(),
		TagCount:       1,
	}
}
