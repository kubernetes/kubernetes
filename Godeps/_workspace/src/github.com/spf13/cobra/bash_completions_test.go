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

func checkOmit(t *testing.T, found, unexpected string) {
	if strings.Contains(found, unexpected) {
		t.Errorf("Unexpected response.\nGot: %q\nBut should not have!\n", unexpected)
	}
}

func check(t *testing.T, found, expected string) {
	if !strings.Contains(found, expected) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	}
}

// World worst custom function, just keep telling you to enter hello!
const (
	bash_completion_func = `__custom_func() {
COMPREPLY=( "hello" )
}
`
)

func TestBashCompletions(t *testing.T) {
	c := initializeWithRootCmd()
	cmdEcho.AddCommand(cmdTimes)
	c.AddCommand(cmdEcho, cmdPrint, cmdDeprecated)

	// custom completion function
	c.BashCompletionFunction = bash_completion_func

	// required flag
	c.MarkFlagRequired("introot")

	// valid nouns
	validArgs := []string{"pods", "nodes", "services", "replicationControllers"}
	c.ValidArgs = validArgs

	// filename
	var flagval string
	c.Flags().StringVar(&flagval, "filename", "", "Enter a filename")
	c.MarkFlagFilename("filename", "json", "yaml", "yml")

	// filename extensions
	var flagvalExt string
	c.Flags().StringVar(&flagvalExt, "filename-ext", "", "Enter a filename (extension limited)")
	c.MarkFlagFilename("filename-ext")

	out := new(bytes.Buffer)
	c.GenBashCompletion(out)
	str := out.String()

	check(t, str, "_cobra-test")
	check(t, str, "_cobra-test_echo")
	check(t, str, "_cobra-test_echo_times")
	check(t, str, "_cobra-test_print")

	// check for required flags
	check(t, str, `must_have_one_flag+=("--introot=")`)
	// check for custom completion function
	check(t, str, `COMPREPLY=( "hello" )`)
	// check for required nouns
	check(t, str, `must_have_one_noun+=("pods")`)
	// check for filename extension flags
	check(t, str, `flags_completion+=("_filedir")`)
	// check for filename extension flags
	check(t, str, `flags_completion+=("__handle_filename_extension_flag json|yaml|yml")`)

	checkOmit(t, str, cmdDeprecated.Name())
}
