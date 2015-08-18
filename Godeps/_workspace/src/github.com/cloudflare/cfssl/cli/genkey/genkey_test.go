package genkey

import (
	"encoding/json"
	"os"
	"os/exec"
	"path"
	"testing"

	"github.com/cloudflare/cfssl/cli"
)

func TestGenkey(t *testing.T) {
	//testing through console
	gopath := os.Getenv("GOPATH")
	cfssl := path.Join(gopath, "bin", "cfssl")
	testdata := path.Join(gopath, "src", "github.com", "cloudflare", "cfssl", "testdata")

	out, err := exec.Command(cfssl, "genkey", path.Join(testdata, "csr.json")).Output()
	if err != nil {
		t.Fatal(err)
	}

	var response map[string]interface{}
	err = json.Unmarshal(out, &response)
	if err != nil {
		t.Fatal(err)
	}

	if response["key"] == nil {
		t.Fatal("No key is outputted.")
	}
	if response["csr"] == nil {
		t.Fatal("No csr is outputted.")
	}

	c := cli.Config{}

	err = genkeyMain([]string{path.Join(testdata, "csr.json")}, c)
	if err != nil {
		t.Fatal(err)
	}

	c.IsCA = true

	err = genkeyMain([]string{path.Join(testdata, "csr.json")}, c)
	if err != nil {
		t.Fatal(err)
	}
}
