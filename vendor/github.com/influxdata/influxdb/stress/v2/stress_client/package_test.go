package stressClient

import (
	"testing"
)

func TestNewPackage(t *testing.T) {
	qry := []byte("SELECT * FROM foo")
	statementID := "foo_id"
	tr := NewTracer(map[string]string{})
	pkg := NewPackage(Query, qry, statementID, tr)
	got := string(pkg.Body)
	if string(qry) != got {
		t.Errorf("expected: %v\ngot: %v\n", qry, got)
	}
}
