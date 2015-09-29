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

func translate(in string) string {
	return strings.Replace(in, "-", "\\-", -1)
}

func TestGenManDoc(t *testing.T) {
	c := initializeWithRootCmd()
	// Need two commands to run the command alphabetical sort
	cmdEcho.AddCommand(cmdTimes, cmdEchoSub, cmdDeprecated)
	c.AddCommand(cmdPrint, cmdEcho)
	cmdRootWithRun.PersistentFlags().StringVarP(&flags2a, "rootflag", "r", "two", strtwoParentHelp)

	out := new(bytes.Buffer)

	// We generate on a subcommand so we have both subcommands and parents
	cmdEcho.GenMan("PROJECT", out)
	found := out.String()

	// Our description
	expected := translate(cmdEcho.Name())
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}

	// Better have our example
	expected = translate(cmdEcho.Name())
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
	expected = translate(cmdRootWithRun.Name())
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}

	// And about subcommands
	expected = translate(cmdEchoSub.Name())
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}

	unexpected := translate(cmdDeprecated.Name())
	if strings.Contains(found, unexpected) {
		t.Errorf("Unexpected response.\nFound: %v\nBut should not have!!\n", unexpected)
	}
}
