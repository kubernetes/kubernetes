package cobra

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"
)

var _ = fmt.Println
var _ = os.Stderr

func TestGenMdDoc(t *testing.T) {
	c := initializeWithRootCmd()
	// Need two commands to run the command alphabetical sort
	cmdEcho.AddCommand(cmdTimes, cmdEchoSub)
	c.AddCommand(cmdPrint, cmdEcho)
	cmdRootWithRun.PersistentFlags().StringVarP(&flags2a, "rootflag", "r", "two", strtwoParentHelp)

	out := new(bytes.Buffer)

	// We generate on s subcommand so we have both subcommands and parents
	GenMarkdown(cmdEcho, out)
	found := out.String()

	// Our description
	expected := cmdEcho.Long
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}

	// Better have our example
	expected = cmdEcho.Example
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}

	// A local flag
	expected = "boolone"
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}

	// persistent flag on parent
	expected = "rootflag"
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}

	// We better output info about our parent
	expected = cmdRootWithRun.Short
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}

	// And about subcommands
	expected = cmdEchoSub.Short
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}

	fmt.Fprintf(os.Stdout, "%s\n", found)
}
