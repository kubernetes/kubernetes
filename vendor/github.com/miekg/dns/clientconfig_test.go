package dns

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

const normal string = `
# Comment
domain somedomain.com
nameserver 10.28.10.2
nameserver 11.28.10.1
`

const missingNewline string = `
domain somedomain.com
nameserver 10.28.10.2
nameserver 11.28.10.1` // <- NOTE: NO newline.

func testConfig(t *testing.T, data string) {
	tempDir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("tempDir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	path := filepath.Join(tempDir, "resolv.conf")
	if err := ioutil.WriteFile(path, []byte(data), 0644); err != nil {
		t.Fatalf("writeFile: %v", err)
	}
	cc, err := ClientConfigFromFile(path)
	if err != nil {
		t.Errorf("error parsing resolv.conf: %v", err)
	}
	if l := len(cc.Servers); l != 2 {
		t.Errorf("incorrect number of nameservers detected: %d", l)
	}
	if l := len(cc.Search); l != 1 {
		t.Errorf("domain directive not parsed correctly: %v", cc.Search)
	} else {
		if cc.Search[0] != "somedomain.com" {
			t.Errorf("domain is unexpected: %v", cc.Search[0])
		}
	}
}

func TestNameserver(t *testing.T)          { testConfig(t, normal) }
func TestMissingFinalNewLine(t *testing.T) { testConfig(t, missingNewline) }
