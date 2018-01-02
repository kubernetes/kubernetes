package logrus

import (
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestRegister(t *testing.T) {
	current := len(handlers)
	RegisterExitHandler(func() {})
	if len(handlers) != current+1 {
		t.Fatalf("expected %d handlers, got %d", current+1, len(handlers))
	}
}

func TestHandler(t *testing.T) {
	tempDir, err := ioutil.TempDir("", "test_handler")
	if err != nil {
		log.Fatalf("can't create temp dir. %q", err)
	}
	defer os.RemoveAll(tempDir)

	gofile := filepath.Join(tempDir, "gofile.go")
	if err := ioutil.WriteFile(gofile, testprog, 0666); err != nil {
		t.Fatalf("can't create go file. %q", err)
	}

	outfile := filepath.Join(tempDir, "outfile.out")
	arg := time.Now().UTC().String()
	err = exec.Command("go", "run", gofile, outfile, arg).Run()
	if err == nil {
		t.Fatalf("completed normally, should have failed")
	}

	data, err := ioutil.ReadFile(outfile)
	if err != nil {
		t.Fatalf("can't read output file %s. %q", outfile, err)
	}

	if string(data) != arg {
		t.Fatalf("bad data. Expected %q, got %q", data, arg)
	}
}

var testprog = []byte(`
// Test program for atexit, gets output file and data as arguments and writes
// data to output file in atexit handler.
package main

import (
	"github.com/sirupsen/logrus"
	"flag"
	"fmt"
	"io/ioutil"
)

var outfile = ""
var data = ""

func handler() {
	ioutil.WriteFile(outfile, []byte(data), 0666)
}

func badHandler() {
	n := 0
	fmt.Println(1/n)
}

func main() {
	flag.Parse()
	outfile = flag.Arg(0)
	data = flag.Arg(1)

	logrus.RegisterExitHandler(handler)
	logrus.RegisterExitHandler(badHandler)
	logrus.Fatal("Bye bye")
}
`)
