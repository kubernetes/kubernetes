// +build linux

package seccomp

import (
	"io/ioutil"
	"testing"

	"github.com/docker/docker/oci"
)

func TestLoadProfile(t *testing.T) {
	f, err := ioutil.ReadFile("fixtures/example.json")
	if err != nil {
		t.Fatal(err)
	}
	rs := oci.DefaultSpec()
	if _, err := LoadProfile(string(f), &rs); err != nil {
		t.Fatal(err)
	}
}

func TestLoadDefaultProfile(t *testing.T) {
	f, err := ioutil.ReadFile("default.json")
	if err != nil {
		t.Fatal(err)
	}
	rs := oci.DefaultSpec()
	if _, err := LoadProfile(string(f), &rs); err != nil {
		t.Fatal(err)
	}
}
