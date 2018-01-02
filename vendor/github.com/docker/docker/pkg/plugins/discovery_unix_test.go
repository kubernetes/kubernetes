// +build !windows

package plugins

import (
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLocalSocket(t *testing.T) {
	// TODO Windows: Enable a similar version for Windows named pipes
	tmpdir, unregister := Setup(t)
	defer unregister()

	cases := []string{
		filepath.Join(tmpdir, "echo.sock"),
		filepath.Join(tmpdir, "echo", "echo.sock"),
	}

	for _, c := range cases {
		if err := os.MkdirAll(filepath.Dir(c), 0755); err != nil {
			t.Fatal(err)
		}

		l, err := net.Listen("unix", c)
		if err != nil {
			t.Fatal(err)
		}

		r := newLocalRegistry()
		p, err := r.Plugin("echo")
		if err != nil {
			t.Fatal(err)
		}

		pp, err := r.Plugin("echo")
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(p, pp) {
			t.Fatalf("Expected %v, was %v\n", p, pp)
		}

		if p.name != "echo" {
			t.Fatalf("Expected plugin `echo`, got %s\n", p.name)
		}

		addr := fmt.Sprintf("unix://%s", c)
		if p.Addr != addr {
			t.Fatalf("Expected plugin addr `%s`, got %s\n", addr, p.Addr)
		}
		if !p.TLSConfig.InsecureSkipVerify {
			t.Fatalf("Expected TLS verification to be skipped")
		}
		l.Close()
	}
}

func TestScan(t *testing.T) {
	tmpdir, unregister := Setup(t)
	defer unregister()

	pluginNames, err := Scan()
	if err != nil {
		t.Fatal(err)
	}
	if pluginNames != nil {
		t.Fatal("Plugin names should be empty.")
	}

	path := filepath.Join(tmpdir, "echo.spec")
	addr := "unix://var/lib/docker/plugins/echo.sock"
	name := "echo"

	err = os.MkdirAll(filepath.Dir(path), 0755)
	if err != nil {
		t.Fatal(err)
	}

	err = ioutil.WriteFile(path, []byte(addr), 0644)
	if err != nil {
		t.Fatal(err)
	}

	r := newLocalRegistry()
	p, err := r.Plugin(name)

	pluginNamesNotEmpty, err := Scan()
	if err != nil {
		t.Fatal(err)
	}
	if p.Name() != pluginNamesNotEmpty[0] {
		t.Fatalf("Unable to scan plugin with name %s", p.name)
	}
}
