package gencert

import (
	"testing"

	"github.com/cloudflare/cfssl/cli"
)

func TestGencertMain(t *testing.T) {

	c := cli.Config{
		IsCA: true,
	}

	err := gencertMain([]string{"../testdata/csr.json"}, c)
	if err != nil {
		t.Fatal(err)
	}

	c = cli.Config{
		IsCA:      true,
		CAKeyFile: "../testdata/ca-key.pem",
	}

	err = gencertMain([]string{"../testdata/csr.json"}, c)
	if err != nil {
		t.Fatal(err)
	}

	c = cli.Config{
		CAFile:    "../testdata/ca.pem",
		CAKeyFile: "../testdata/ca-key.pem",
	}

	err = gencertMain([]string{"../testdata/csr.json"}, c)
	if err != nil {
		t.Fatal(err)
	}

	c = cli.Config{
		RenewCA:   true,
		CAFile:    "../testdata/ca.pem",
		CAKeyFile: "../testdata/ca-key.pem",
	}
	err = gencertMain([]string{}, c)

	if err != nil {
		t.Fatal(err)
	}
}

func TestBadGencertMain(t *testing.T) {
	err := gencertMain([]string{"../testdata/csr.json"}, cli.Config{})
	if err != nil {
		t.Fatal(err)
	}

	err = gencertMain([]string{"../testdata/csr.json"}, cli.Config{CAFile: "../testdata/ca.pem"})
	if err != nil {
		t.Fatal(err)
	}

	err = gencertMain([]string{}, cli.Config{RenewCA: true})
	if err == nil {
		t.Fatal("No CA or Key provided, should report error")
	}

	err = gencertMain([]string{}, cli.Config{})
	if err == nil {
		t.Fatal("Not enough argument, should report error")
	}

	err = gencertMain([]string{"../testdata/bad_csr.json"}, cli.Config{})
	if err == nil {
		t.Fatal("Bad CSR JSON, should report error")
	}

	err = gencertMain([]string{"../testdata/nothing"}, cli.Config{})
	if err == nil {
		t.Fatal("Trying to read a non-existance file, should report error")
	}

	err = gencertMain([]string{"../testdata/csr.json"}, cli.Config{IsCA: true, CAKeyFile: "../../testdata/garbage.crt"})
	if err == nil {
		t.Fatal("Bad CA, should report error")
	}

	err = gencertMain([]string{"../testdata/csr.json"}, cli.Config{CAFile: "../testdata/ca.pem", Remote: "123::::123"})
	if err == nil {
		t.Fatal("Invalid remote, should reort error")
	}

}
