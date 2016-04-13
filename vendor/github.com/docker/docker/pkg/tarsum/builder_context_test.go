package tarsum

import (
	"io"
	"io/ioutil"
	"os"
	"testing"
)

// Try to remove tarsum (in the BuilderContext) that do not exists, won't change a thing
func TestTarSumRemoveNonExistent(t *testing.T) {
	filename := "testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/layer.tar"
	reader, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	ts, err := NewTarSum(reader, false, Version0)
	if err != nil {
		t.Fatal(err)
	}

	// Read and discard bytes so that it populates sums
	_, err = io.Copy(ioutil.Discard, ts)
	if err != nil {
		t.Errorf("failed to read from %s: %s", filename, err)
	}

	expected := len(ts.GetSums())

	ts.(BuilderContext).Remove("")
	ts.(BuilderContext).Remove("Anything")

	if len(ts.GetSums()) != expected {
		t.Fatalf("Expected %v sums, go %v.", expected, ts.GetSums())
	}
}

// Remove a tarsum (in the BuilderContext)
func TestTarSumRemove(t *testing.T) {
	filename := "testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/layer.tar"
	reader, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	ts, err := NewTarSum(reader, false, Version0)
	if err != nil {
		t.Fatal(err)
	}

	// Read and discard bytes so that it populates sums
	_, err = io.Copy(ioutil.Discard, ts)
	if err != nil {
		t.Errorf("failed to read from %s: %s", filename, err)
	}

	expected := len(ts.GetSums()) - 1

	ts.(BuilderContext).Remove("etc/sudoers")

	if len(ts.GetSums()) != expected {
		t.Fatalf("Expected %v sums, go %v.", expected, len(ts.GetSums()))
	}
}
