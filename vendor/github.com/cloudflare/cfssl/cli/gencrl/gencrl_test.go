package gencrl

import (
	"testing"

	"github.com/cloudflare/cfssl/cli"
)

func TestGencrl(t *testing.T) {

	var err error

	err = gencrlMain([]string{"testdata/serialList", "testdata/caTwo.pem", "testdata/ca-keyTwo.pem"}, cli.Config{})
	if err != nil {
		t.Fatal(err)
	}

}
