package seed

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

const (
	source  = "github://github.com/libopenstorage/openstorage"
	goodRev = "b74a0549f71af335e78a7b35057abadb790e48ef"
	dest    = "/tmp/foo"
)

var s Source

func TestSetup(t *testing.T) {
	var err error
	s, err = New("badscheme://github.com/libopenstorage/openstorage", nil)
	assert.Error(t, err, "invalid schemme should fail")
	s, err = New(source, map[string]string{GitRevision: goodRev})
	if err != nil {
		t.Fatalf("Failed to setup test %v", err)
	}
	os.RemoveAll(dest)
}

func TestLoad(t *testing.T) {
	os.MkdirAll(dest, 0755)
	assert.NoError(t, s.Load(dest), "Failed in load")
	os.RemoveAll(dest)
}
