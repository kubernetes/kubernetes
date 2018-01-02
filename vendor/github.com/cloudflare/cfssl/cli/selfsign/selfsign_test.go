package selfsign

import (
	"testing"

	"github.com/cloudflare/cfssl/cli"
)

func TestSelfSignMain(t *testing.T) {
	err := selfSignMain([]string{"cloudflare.com", "../../testdata/csr.json"}, cli.Config{Hostname: ""})
	if err != nil {
		t.Fatal(err)
	}
}

func TestBadSelfSignMain(t *testing.T) {
	err := selfSignMain([]string{"cloudflare.com"}, cli.Config{Hostname: ""})
	if err == nil {
		t.Fatal("No CSR, should report error")
	}

	err = selfSignMain([]string{}, cli.Config{Hostname: ""})
	if err == nil {
		t.Fatal("No server, should report error")
	}

	err = selfSignMain([]string{"cloudflare.com", "../../testdata/garbage.key"}, cli.Config{Hostname: ""})
	if err == nil {
		t.Fatal("Wrong CSR file, should report error")
	}
}
