// +build linux,cgo,seccomp

package seccomp

import "testing"

func TestParseStatusFile(t *testing.T) {
	s, err := parseStatusFile("fixtures/proc_self_status")
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := s["Seccomp"]; !ok {

		t.Fatal("expected to find 'Seccomp' in the map but did not.")
	}
}
