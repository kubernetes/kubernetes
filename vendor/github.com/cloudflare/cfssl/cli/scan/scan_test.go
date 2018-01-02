package scan

import (
	"testing"

	"github.com/cloudflare/cfssl/cli"
)

var hosts = []string{"www.cloudflare.com", "google.com"}

func TestScanMain(t *testing.T) {
	err := scanMain(hosts, cli.Config{})
	if err != nil {
		t.Fatal(err)
	}

	err = scanMain(nil, cli.Config{Hostname: "www.cloudflare.com, google.com", List: true})
	if err != nil {
		t.Fatal(err)
	}
}
