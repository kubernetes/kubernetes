package stressql

import "testing"

// Pulls the default configFile and makes sure it parses
func TestParseStatements(t *testing.T) {
	stmts, err := ParseStatements("../iql/file.iql")
	if err != nil {
		t.Error(err)
	}
	expected := 15
	got := len(stmts)
	if expected != got {
		t.Errorf("expected: %v\ngot: %v\n", expected, got)
	}
}
