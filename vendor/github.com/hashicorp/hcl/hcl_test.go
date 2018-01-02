package hcl

import (
	"io/ioutil"
	"path/filepath"
	"testing"
)

// This is the directory where our test fixtures are.
const fixtureDir = "./test-fixtures"

func testReadFile(t *testing.T, n string) string {
	d, err := ioutil.ReadFile(filepath.Join(fixtureDir, n))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	return string(d)
}
