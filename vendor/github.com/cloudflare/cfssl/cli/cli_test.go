package cli

import (
	"flag"
	"os"
	"testing"
)

var cfsslFlagSet = flag.NewFlagSet("cfssl", flag.ExitOnError)

// The testing style from this package is borrowed from the Go flag
// library's methodology for testing this. We set flag.Usage to nil,
// then replace it with a closure to ensure the usage function was
// called appropriately.

// 'cfssl -help' should be supported.
func TestHelp(t *testing.T) {
	called := false
	ResetForTesting(func() { called = true })
	os.Args = []string{"cfssl", "-help"}
	Start(nil)
	if !called {
		t.Fatal("flag -help is not recognized correctly.")
	}

}

// 'cfssl -badflag' should trigger parse error and usage invocation.
func TestUnknownFlag(t *testing.T) {
	called := false
	os.Args = []string{"cfssl", "-badflag"}
	ResetForTesting(func() { called = true })
	Start(nil)
	if !called {
		t.Fatal("Bad flag is not caught.")
	}

}

// 'cfssl badcommand' should trigger parse error and usage invocation.
func TestBadCommand(t *testing.T) {
	called := false
	ResetForTesting(func() { called = true })
	os.Args = []string{"cfssl", "badcommand"}
	Start(nil)
	if !called {
		t.Fatal("Bad command is not caught.")
	}
}

func TestCommandHelp(t *testing.T) {
	called := false
	ResetCFSSLFlagSetForTesting(func() { called = true })
	args := []string{"-help"}
	cfsslFlagSet.Parse(args)
	if !called {
		t.Fatal("sub-command -help is not recognized.")
	}
}

func TestCommandBadFlag(t *testing.T) {
	called := false
	ResetCFSSLFlagSetForTesting(func() { called = true })
	args := []string{"-help", "-badflag"}
	cfsslFlagSet.Parse(args)
	if !called {
		t.Fatal("bad flag for sub-command is not caught.")
	}
}

// Additional routines derived from flag unit testing

// ResetForTesting clears all flag state and sets the usage function as directed.
// After calling ResetForTesting, parse errors in flag handling will not
// exit the program.
func ResetForTesting(usage func()) {
	flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	flag.Usage = usage
}

// ResetCFSSLFlagSetForTesting reset cfsslFlagSet with flag.ContinueOnError so parse
// errors in flag will not exit the program
func ResetCFSSLFlagSetForTesting(usage func()) {
	var c Config
	cfsslFlagSet = flag.NewFlagSet("cfssl", flag.ContinueOnError)
	registerFlags(&c, cfsslFlagSet)
	cfsslFlagSet.Usage = usage
}

func TestReadStdin(t *testing.T) {
	fn, err := ReadStdin("./testdata/test.txt")
	if err != nil {
		t.Fatal(err)
	}

	if string(fn) != "This is a test file" {
		t.Fatal(err)
	}

	fn, err = ReadStdin("-")
	if err != nil {
		t.Fatal(err)
	}
}

func TestPopFirstArg(t *testing.T) {
	s, str, err := PopFirstArgument([]string{"a", "b", "c"})
	if s != "a" {
		t.Fatal("Did not pop first argument successfully")
	}
	if str == nil {
		t.Fatal("Did not return the rest of argument successfully")
	}
	if err != nil {
		t.Fatal(err)
	}

	//test invalid argument
	_, _, err = PopFirstArgument([]string{})
	if err == nil {
		t.Fatal("No argument given, should return error")
	}
}
