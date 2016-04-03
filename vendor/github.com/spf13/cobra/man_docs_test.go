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

	header := &GenManHeader{
		Title:   "Project",
		Section: "2",
	}
	// We generate on a subcommand so we have both subcommands and parents
	cmdEcho.GenMan(header, out)
	found := out.String()

	// Make sure parent has - in CommandPath() in SEE ALSO:
	parentPath := cmdEcho.Parent().CommandPath()
	dashParentPath := strings.Replace(parentPath, " ", "-", -1)
	expected := translate(dashParentPath)
	expected = expected + "(" + header.Section + ")"
	checkStringContains(t, found, expected)

	// Our description
	expected = translate(cmdEcho.Name())
	checkStringContains(t, found, expected)

	// Better have our example
	expected = translate(cmdEcho.Name())
	checkStringContains(t, found, expected)

	// A local flag
	expected = "boolone"
	checkStringContains(t, found, expected)

	// persistent flag on parent
	expected = "rootflag"
	checkStringContains(t, found, expected)

	// We better output info about our parent
	expected = translate(cmdRootWithRun.Name())
	checkStringContains(t, found, expected)

	// And about subcommands
	expected = translate(cmdEchoSub.Name())
	checkStringContains(t, found, expected)

	unexpected := translate(cmdDeprecated.Name())
	checkStringOmits(t, found, unexpected)

	// auto generated
	expected = translate("Auto generated")
	checkStringContains(t, found, expected)
}

func TestGenManNoGenTag(t *testing.T) {

	c := initializeWithRootCmd()
	// Need two commands to run the command alphabetical sort
	cmdEcho.AddCommand(cmdTimes, cmdEchoSub, cmdDeprecated)
	c.AddCommand(cmdPrint, cmdEcho)
	cmdRootWithRun.PersistentFlags().StringVarP(&flags2a, "rootflag", "r", "two", strtwoParentHelp)
	cmdEcho.DisableAutoGenTag = true
	out := new(bytes.Buffer)

	header := &GenManHeader{
		Title:   "Project",
		Section: "2",
	}
	// We generate on a subcommand so we have both subcommands and parents
	cmdEcho.GenMan(header, out)
	found := out.String()

	unexpected := translate("#HISTORY")
	checkStringOmits(t, found, unexpected)
}
