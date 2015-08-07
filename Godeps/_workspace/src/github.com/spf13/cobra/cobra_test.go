package cobra

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

var _ = fmt.Println
var _ = os.Stderr

var tp, te, tt, t1, tr []string
var rootPersPre, echoPre, echoPersPre, timesPersPre []string
var flagb1, flagb2, flagb3, flagbr, flagbp bool
var flags1, flags2a, flags2b, flags3 string
var flagi1, flagi2, flagi3, flagir int
var globalFlag1 bool
var flagEcho, rootcalled bool
var versionUsed int

const strtwoParentHelp = "help message for parent flag strtwo"
const strtwoChildHelp = "help message for child flag strtwo"

var cmdPrint = &Command{
	Use:   "print [string to print]",
	Short: "Print anything to the screen",
	Long:  `an absolutely utterly useless command for testing.`,
	Run: func(cmd *Command, args []string) {
		tp = args
	},
}

var cmdEcho = &Command{
	Use:     "echo [string to echo]",
	Aliases: []string{"say"},
	Short:   "Echo anything to the screen",
	Long:    `an utterly useless command for testing.`,
	Example: "Just run cobra-test echo",
	PersistentPreRun: func(cmd *Command, args []string) {
		echoPersPre = args
	},
	PreRun: func(cmd *Command, args []string) {
		echoPre = args
	},
	Run: func(cmd *Command, args []string) {
		te = args
	},
}

var cmdEchoSub = &Command{
	Use:   "echosub [string to print]",
	Short: "second sub command for echo",
	Long:  `an absolutely utterly useless command for testing gendocs!.`,
	Run: func(cmd *Command, args []string) {
	},
}

var cmdDeprecated = &Command{
	Use:        "deprecated [can't do anything here]",
	Short:      "A command which is deprecated",
	Long:       `an absolutely utterly useless command for testing deprecation!.`,
	Deprecated: "Please use echo instead",
	Run: func(cmd *Command, args []string) {
	},
}

var cmdTimes = &Command{
	Use:   "times [# times] [string to echo]",
	Short: "Echo anything to the screen more times",
	Long:  `a slightly useless command for testing.`,
	PersistentPreRun: func(cmd *Command, args []string) {
		timesPersPre = args
	},
	Run: func(cmd *Command, args []string) {
		tt = args
	},
}

var cmdRootNoRun = &Command{
	Use:   "cobra-test",
	Short: "The root can run it's own function",
	Long:  "The root description for help",
	PersistentPreRun: func(cmd *Command, args []string) {
		rootPersPre = args
	},
}

var cmdRootSameName = &Command{
	Use:   "print",
	Short: "Root with the same name as a subcommand",
	Long:  "The root description for help",
}

var cmdRootWithRun = &Command{
	Use:   "cobra-test",
	Short: "The root can run it's own function",
	Long:  "The root description for help",
	Run: func(cmd *Command, args []string) {
		tr = args
		rootcalled = true
	},
}

var cmdSubNoRun = &Command{
	Use:   "subnorun",
	Short: "A subcommand without a Run function",
	Long:  "A long output about a subcommand without a Run function",
}

var cmdVersion1 = &Command{
	Use:   "version",
	Short: "Print the version number",
	Long:  `First version of the version command`,
	Run: func(cmd *Command, args []string) {
		versionUsed = 1
	},
}

var cmdVersion2 = &Command{
	Use:   "version",
	Short: "Print the version number",
	Long:  `Second version of the version command`,
	Run: func(cmd *Command, args []string) {
		versionUsed = 2
	},
}

func flagInit() {
	cmdEcho.ResetFlags()
	cmdPrint.ResetFlags()
	cmdTimes.ResetFlags()
	cmdRootNoRun.ResetFlags()
	cmdRootSameName.ResetFlags()
	cmdRootWithRun.ResetFlags()
	cmdSubNoRun.ResetFlags()
	cmdRootNoRun.PersistentFlags().StringVarP(&flags2a, "strtwo", "t", "two", strtwoParentHelp)
	cmdEcho.Flags().IntVarP(&flagi1, "intone", "i", 123, "help message for flag intone")
	cmdTimes.Flags().IntVarP(&flagi2, "inttwo", "j", 234, "help message for flag inttwo")
	cmdPrint.Flags().IntVarP(&flagi3, "intthree", "i", 345, "help message for flag intthree")
	cmdEcho.PersistentFlags().StringVarP(&flags1, "strone", "s", "one", "help message for flag strone")
	cmdEcho.PersistentFlags().BoolVarP(&flagbp, "persistentbool", "p", false, "help message for flag persistentbool")
	cmdTimes.PersistentFlags().StringVarP(&flags2b, "strtwo", "t", "2", strtwoChildHelp)
	cmdPrint.PersistentFlags().StringVarP(&flags3, "strthree", "s", "three", "help message for flag strthree")
	cmdEcho.Flags().BoolVarP(&flagb1, "boolone", "b", true, "help message for flag boolone")
	cmdTimes.Flags().BoolVarP(&flagb2, "booltwo", "c", false, "help message for flag booltwo")
	cmdPrint.Flags().BoolVarP(&flagb3, "boolthree", "b", true, "help message for flag boolthree")
	cmdVersion1.ResetFlags()
	cmdVersion2.ResetFlags()
}

func commandInit() {
	cmdEcho.ResetCommands()
	cmdPrint.ResetCommands()
	cmdTimes.ResetCommands()
	cmdRootNoRun.ResetCommands()
	cmdRootSameName.ResetCommands()
	cmdRootWithRun.ResetCommands()
	cmdSubNoRun.ResetCommands()
}

func initialize() *Command {
	tt, tp, te = nil, nil, nil
	rootPersPre, echoPre, echoPersPre, timesPersPre = nil, nil, nil, nil

	var c = cmdRootNoRun
	flagInit()
	commandInit()
	return c
}

func initializeWithSameName() *Command {
	tt, tp, te = nil, nil, nil
	rootPersPre, echoPre, echoPersPre, timesPersPre = nil, nil, nil, nil
	var c = cmdRootSameName
	flagInit()
	commandInit()
	return c
}

func initializeWithRootCmd() *Command {
	cmdRootWithRun.ResetCommands()
	tt, tp, te, tr, rootcalled = nil, nil, nil, nil, false
	flagInit()
	cmdRootWithRun.Flags().BoolVarP(&flagbr, "boolroot", "b", false, "help message for flag boolroot")
	cmdRootWithRun.Flags().IntVarP(&flagir, "introot", "i", 321, "help message for flag introot")
	commandInit()
	return cmdRootWithRun
}

type resulter struct {
	Error   error
	Output  string
	Command *Command
}

func fullSetupTest(input string) resulter {
	c := initializeWithRootCmd()

	return fullTester(c, input)
}

func noRRSetupTest(input string) resulter {
	c := initialize()

	return fullTester(c, input)
}

func rootOnlySetupTest(input string) resulter {
	c := initializeWithRootCmd()

	return simpleTester(c, input)
}

func simpleTester(c *Command, input string) resulter {
	buf := new(bytes.Buffer)
	// Testing flag with invalid input
	c.SetOutput(buf)
	c.SetArgs(strings.Split(input, " "))

	err := c.Execute()
	output := buf.String()

	return resulter{err, output, c}
}

func fullTester(c *Command, input string) resulter {
	buf := new(bytes.Buffer)
	// Testing flag with invalid input
	c.SetOutput(buf)
	cmdEcho.AddCommand(cmdTimes)
	c.AddCommand(cmdPrint, cmdEcho, cmdSubNoRun, cmdDeprecated)
	c.SetArgs(strings.Split(input, " "))

	err := c.Execute()
	output := buf.String()

	return resulter{err, output, c}
}

func logErr(t *testing.T, found, expected string) {
	out := new(bytes.Buffer)

	_, _, line, ok := runtime.Caller(2)
	if ok {
		fmt.Fprintf(out, "Line: %d ", line)
	}
	fmt.Fprintf(out, "Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", expected, found)
	t.Errorf(out.String())
}

func checkResultContains(t *testing.T, x resulter, check string) {
	if !strings.Contains(x.Output, check) {
		logErr(t, x.Output, check)
	}
}

func checkResultOmits(t *testing.T, x resulter, check string) {
	if strings.Contains(x.Output, check) {
		logErr(t, x.Output, check)
	}
}

func checkOutputContains(t *testing.T, c *Command, check string) {
	buf := new(bytes.Buffer)
	c.SetOutput(buf)
	c.Execute()

	if !strings.Contains(buf.String(), check) {
		logErr(t, buf.String(), check)
	}
}

func TestSingleCommand(t *testing.T) {
	noRRSetupTest("print one two")

	if te != nil || tt != nil {
		t.Error("Wrong command called")
	}
	if tp == nil {
		t.Error("Wrong command called")
	}
	if strings.Join(tp, " ") != "one two" {
		t.Error("Command didn't parse correctly")
	}
}

func TestChildCommand(t *testing.T) {
	noRRSetupTest("echo times one two")

	if te != nil || tp != nil {
		t.Error("Wrong command called")
	}
	if tt == nil {
		t.Error("Wrong command called")
	}
	if strings.Join(tt, " ") != "one two" {
		t.Error("Command didn't parse correctly")
	}
}

func TestCommandAlias(t *testing.T) {
	noRRSetupTest("say times one two")

	if te != nil || tp != nil {
		t.Error("Wrong command called")
	}
	if tt == nil {
		t.Error("Wrong command called")
	}
	if strings.Join(tt, " ") != "one two" {
		t.Error("Command didn't parse correctly")
	}
}

func TestPrefixMatching(t *testing.T) {
	EnablePrefixMatching = true
	noRRSetupTest("ech times one two")

	if te != nil || tp != nil {
		t.Error("Wrong command called")
	}
	if tt == nil {
		t.Error("Wrong command called")
	}
	if strings.Join(tt, " ") != "one two" {
		t.Error("Command didn't parse correctly")
	}

	EnablePrefixMatching = false
}

func TestNoPrefixMatching(t *testing.T) {
	EnablePrefixMatching = false

	noRRSetupTest("ech times one two")

	if !(tt == nil && te == nil && tp == nil) {
		t.Error("Wrong command called")
	}
}

func TestAliasPrefixMatching(t *testing.T) {
	EnablePrefixMatching = true
	noRRSetupTest("sa times one two")

	if te != nil || tp != nil {
		t.Error("Wrong command called")
	}
	if tt == nil {
		t.Error("Wrong command called")
	}
	if strings.Join(tt, " ") != "one two" {
		t.Error("Command didn't parse correctly")
	}
	EnablePrefixMatching = false
}

func TestChildSameName(t *testing.T) {
	c := initializeWithSameName()
	c.AddCommand(cmdPrint, cmdEcho)
	c.SetArgs(strings.Split("print one two", " "))
	c.Execute()

	if te != nil || tt != nil {
		t.Error("Wrong command called")
	}
	if tp == nil {
		t.Error("Wrong command called")
	}
	if strings.Join(tp, " ") != "one two" {
		t.Error("Command didn't parse correctly")
	}
}

func TestGrandChildSameName(t *testing.T) {
	c := initializeWithSameName()
	cmdTimes.AddCommand(cmdPrint)
	c.AddCommand(cmdTimes)
	c.SetArgs(strings.Split("times print one two", " "))
	c.Execute()

	if te != nil || tt != nil {
		t.Error("Wrong command called")
	}
	if tp == nil {
		t.Error("Wrong command called")
	}
	if strings.Join(tp, " ") != "one two" {
		t.Error("Command didn't parse correctly")
	}
}

func TestFlagLong(t *testing.T) {
	noRRSetupTest("echo --intone=13 something here")

	if strings.Join(te, " ") != "something here" {
		t.Errorf("flags didn't leave proper args remaining..%s given", te)
	}
	if flagi1 != 13 {
		t.Errorf("int flag didn't get correct value, had %d", flagi1)
	}
	if flagi2 != 234 {
		t.Errorf("default flag value changed, 234 expected, %d given", flagi2)
	}
}

func TestFlagShort(t *testing.T) {
	noRRSetupTest("echo -i13 something here")

	if strings.Join(te, " ") != "something here" {
		t.Errorf("flags didn't leave proper args remaining..%s given", te)
	}
	if flagi1 != 13 {
		t.Errorf("int flag didn't get correct value, had %d", flagi1)
	}
	if flagi2 != 234 {
		t.Errorf("default flag value changed, 234 expected, %d given", flagi2)
	}

	noRRSetupTest("echo -i 13 something here")

	if strings.Join(te, " ") != "something here" {
		t.Errorf("flags didn't leave proper args remaining..%s given", te)
	}
	if flagi1 != 13 {
		t.Errorf("int flag didn't get correct value, had %d", flagi1)
	}
	if flagi2 != 234 {
		t.Errorf("default flag value changed, 234 expected, %d given", flagi2)
	}

	noRRSetupTest("print -i99 one two")

	if strings.Join(tp, " ") != "one two" {
		t.Errorf("flags didn't leave proper args remaining..%s given", tp)
	}
	if flagi3 != 99 {
		t.Errorf("int flag didn't get correct value, had %d", flagi3)
	}
	if flagi1 != 123 {
		t.Errorf("default flag value changed on different command with same shortname, 234 expected, %d given", flagi2)
	}
}

func TestChildCommandFlags(t *testing.T) {
	noRRSetupTest("echo times -j 99 one two")

	if strings.Join(tt, " ") != "one two" {
		t.Errorf("flags didn't leave proper args remaining..%s given", tt)
	}

	// Testing with flag that shouldn't be persistent
	r := noRRSetupTest("echo times -j 99 -i77 one two")

	if r.Error == nil {
		t.Errorf("invalid flag should generate error")
	}

	if !strings.Contains(r.Output, "unknown shorthand") {
		t.Errorf("Wrong error message displayed, \n %s", r.Output)
	}

	if flagi2 != 99 {
		t.Errorf("flag value should be 99, %d given", flagi2)
	}

	if flagi1 != 123 {
		t.Errorf("unset flag should have default value, expecting 123, given %d", flagi1)
	}

	// Testing with flag only existing on child
	r = noRRSetupTest("echo -j 99 -i77 one two")

	if r.Error == nil {
		t.Errorf("invalid flag should generate error")
	}

	if !strings.Contains(r.Output, "unknown shorthand flag") {
		t.Errorf("Wrong error message displayed, \n %s", r.Output)
	}

	// Testing with persistent flag overwritten by child
	noRRSetupTest("echo times --strtwo=child one two")

	if flags2b != "child" {
		t.Errorf("flag value should be child, %s given", flags2b)
	}

	if flags2a != "two" {
		t.Errorf("unset flag should have default value, expecting two, given %s", flags2a)
	}

	// Testing flag with invalid input
	r = noRRSetupTest("echo -i10E")

	if r.Error == nil {
		t.Errorf("invalid input should generate error")
	}

	if !strings.Contains(r.Output, "invalid argument \"10E\" for i10E") {
		t.Errorf("Wrong error message displayed, \n %s", r.Output)
	}
}

func TestTrailingCommandFlags(t *testing.T) {
	x := fullSetupTest("echo two -x")

	if x.Error == nil {
		t.Errorf("invalid flag should generate error")
	}
}

func TestInvalidSubcommandFlags(t *testing.T) {
	cmd := initializeWithRootCmd()
	cmd.AddCommand(cmdTimes)

	result := simpleTester(cmd, "times --inttwo=2 --badflag=bar")

	checkResultContains(t, result, "unknown flag: --badflag")

	if strings.Contains(result.Output, "unknown flag: --inttwo") {
		t.Errorf("invalid --badflag flag shouldn't fail on 'unknown' --inttwo flag")
	}

}

func TestSubcommandArgEvaluation(t *testing.T) {
	cmd := initializeWithRootCmd()

	first := &Command{
		Use: "first",
		Run: func(cmd *Command, args []string) {
		},
	}
	cmd.AddCommand(first)

	second := &Command{
		Use: "second",
		Run: func(cmd *Command, args []string) {
			fmt.Fprintf(cmd.Out(), "%v", args)
		},
	}
	first.AddCommand(second)

	result := simpleTester(cmd, "first second first third")

	expectedOutput := fmt.Sprintf("%v", []string{"first third"})
	if result.Output != expectedOutput {
		t.Errorf("exptected %v, got %v", expectedOutput, result.Output)
	}
}

func TestPersistentFlags(t *testing.T) {
	fullSetupTest("echo -s something -p more here")

	// persistentFlag should act like normal flag on it's own command
	if strings.Join(te, " ") != "more here" {
		t.Errorf("flags didn't leave proper args remaining..%s given", te)
	}
	if flags1 != "something" {
		t.Errorf("string flag didn't get correct value, had %v", flags1)
	}
	if !flagbp {
		t.Errorf("persistent bool flag not parsed correctly. Expected true, had %v", flagbp)
	}

	// persistentFlag should act like normal flag on it's own command
	fullSetupTest("echo times -s again -c -p test here")

	if strings.Join(tt, " ") != "test here" {
		t.Errorf("flags didn't leave proper args remaining..%s given", tt)
	}

	if flags1 != "again" {
		t.Errorf("string flag didn't get correct value, had %v", flags1)
	}

	if !flagb2 {
		t.Errorf("local flag not parsed correctly. Expected true, had %v", flagb2)
	}
	if !flagbp {
		t.Errorf("persistent bool flag not parsed correctly. Expected true, had %v", flagbp)
	}
}

func TestHelpCommand(t *testing.T) {
	x := fullSetupTest("help")
	checkResultContains(t, x, cmdRootWithRun.Long)

	x = fullSetupTest("help echo")
	checkResultContains(t, x, cmdEcho.Long)

	x = fullSetupTest("help echo times")
	checkResultContains(t, x, cmdTimes.Long)
}

func TestChildCommandHelp(t *testing.T) {
	c := noRRSetupTest("print --help")
	checkResultContains(t, c, strtwoParentHelp)
	r := noRRSetupTest("echo times --help")
	checkResultContains(t, r, strtwoChildHelp)
}

func TestNonRunChildHelp(t *testing.T) {
	x := noRRSetupTest("subnorun")
	checkResultContains(t, x, cmdSubNoRun.Long)
}

func TestRunnableRootCommand(t *testing.T) {
	fullSetupTest("")

	if rootcalled != true {
		t.Errorf("Root Function was not called")
	}
}

func TestRunnableRootCommandNilInput(t *testing.T) {
	empty_arg := make([]string, 0)
	c := initializeWithRootCmd()

	buf := new(bytes.Buffer)
	// Testing flag with invalid input
	c.SetOutput(buf)
	cmdEcho.AddCommand(cmdTimes)
	c.AddCommand(cmdPrint, cmdEcho)
	c.SetArgs(empty_arg)

	c.Execute()

	if rootcalled != true {
		t.Errorf("Root Function was not called")
	}
}

func TestRunnableRootCommandEmptyInput(t *testing.T) {
	args := make([]string, 3)
	args[0] = ""
	args[1] = "--introot=12"
	args[2] = ""
	c := initializeWithRootCmd()

	buf := new(bytes.Buffer)
	// Testing flag with invalid input
	c.SetOutput(buf)
	cmdEcho.AddCommand(cmdTimes)
	c.AddCommand(cmdPrint, cmdEcho)
	c.SetArgs(args)

	c.Execute()

	if rootcalled != true {
		t.Errorf("Root Function was not called.\n\nOutput was:\n\n%s\n", buf)
	}
}

func TestInvalidSubcommandWhenArgsAllowed(t *testing.T) {
	fullSetupTest("echo invalid-sub")

	if te[0] != "invalid-sub" {
		t.Errorf("Subcommand didn't work...")
	}
}

func TestRootFlags(t *testing.T) {
	fullSetupTest("-i 17 -b")

	if flagbr != true {
		t.Errorf("flag value should be true, %v given", flagbr)
	}

	if flagir != 17 {
		t.Errorf("flag value should be 17, %d given", flagir)
	}
}

func TestRootHelp(t *testing.T) {
	x := fullSetupTest("--help")

	checkResultContains(t, x, "Available Commands:")
	checkResultContains(t, x, "for more information about a command")

	if strings.Contains(x.Output, "unknown flag: --help") {
		t.Errorf("--help shouldn't trigger an error, Got: \n %s", x.Output)
	}

	if strings.Contains(x.Output, cmdEcho.Use) {
		t.Errorf("--help shouldn't display subcommand's usage, Got: \n %s", x.Output)
	}

	x = fullSetupTest("echo --help")

	if strings.Contains(x.Output, cmdTimes.Use) {
		t.Errorf("--help shouldn't display subsubcommand's usage, Got: \n %s", x.Output)
	}

	checkResultContains(t, x, "Available Commands:")
	checkResultContains(t, x, "for more information about a command")

	if strings.Contains(x.Output, "unknown flag: --help") {
		t.Errorf("--help shouldn't trigger an error, Got: \n %s", x.Output)
	}

}

func TestFlagAccess(t *testing.T) {
	initialize()

	local := cmdTimes.LocalFlags()
	inherited := cmdTimes.InheritedFlags()

	for _, f := range []string{"inttwo", "strtwo", "booltwo"} {
		if local.Lookup(f) == nil {
			t.Errorf("LocalFlags expected to contain %s, Got: nil", f)
		}
	}
	if inherited.Lookup("strone") == nil {
		t.Errorf("InheritedFlags expected to contain strone, Got: nil")
	}
	if inherited.Lookup("strtwo") != nil {
		t.Errorf("InheritedFlags shouldn not contain overwritten flag strtwo")

	}
}

func TestNoNRunnableRootCommandNilInput(t *testing.T) {
	args := make([]string, 0)
	c := initialize()

	buf := new(bytes.Buffer)
	// Testing flag with invalid input
	c.SetOutput(buf)
	cmdEcho.AddCommand(cmdTimes)
	c.AddCommand(cmdPrint, cmdEcho)
	c.SetArgs(args)

	c.Execute()

	if !strings.Contains(buf.String(), cmdRootNoRun.Long) {
		t.Errorf("Expected to get help output, Got: \n %s", buf)
	}
}

func TestRootNoCommandHelp(t *testing.T) {
	x := rootOnlySetupTest("--help")

	checkResultOmits(t, x, "Available Commands:")
	checkResultOmits(t, x, "for more information about a command")

	if strings.Contains(x.Output, "unknown flag: --help") {
		t.Errorf("--help shouldn't trigger an error, Got: \n %s", x.Output)
	}

	x = rootOnlySetupTest("echo --help")

	checkResultOmits(t, x, "Available Commands:")
	checkResultOmits(t, x, "for more information about a command")

	if strings.Contains(x.Output, "unknown flag: --help") {
		t.Errorf("--help shouldn't trigger an error, Got: \n %s", x.Output)
	}
}

func TestRootUnknownCommand(t *testing.T) {
	r := noRRSetupTest("bogus")
	s := "Error: unknown command \"bogus\" for \"cobra-test\"\nRun 'cobra-test --help' for usage.\n"

	if r.Output != s {
		t.Errorf("Unexpected response.\nExpecting to be:\n %q\nGot:\n %q\n", s, r.Output)
	}

	r = noRRSetupTest("--strtwo=a bogus")
	if r.Output != s {
		t.Errorf("Unexpected response.\nExpecting to be:\n %q\nGot:\n %q\n", s, r.Output)
	}
}

func TestFlagsBeforeCommand(t *testing.T) {
	// short without space
	x := fullSetupTest("-i10 echo")
	if x.Error != nil {
		t.Errorf("Valid Input shouldn't have errors, got:\n %q", x.Error)
	}

	// short (int) with equals
	// It appears that pflags doesn't support this...
	// Commenting out until support can be added

	//x = noRRSetupTest("echo -i=10")
	//if x.Error != nil {
	//t.Errorf("Valid Input shouldn't have errors, got:\n %s", x.Error)
	//}

	// long with equals
	x = noRRSetupTest("--intone=123 echo one two")
	if x.Error != nil {
		t.Errorf("Valid Input shouldn't have errors, got:\n %s", x.Error)
	}

	// With parsing error properly reported
	x = fullSetupTest("-i10E echo")
	if !strings.Contains(x.Output, "invalid argument \"10E\" for i10E") {
		t.Errorf("Wrong error message displayed, \n %s", x.Output)
	}

	//With quotes
	x = fullSetupTest("-s=\"walking\" echo")
	if x.Error != nil {
		t.Errorf("Valid Input shouldn't have errors, got:\n %q", x.Error)
	}

	//With quotes and space
	x = fullSetupTest("-s=\"walking fast\" echo")
	if x.Error != nil {
		t.Errorf("Valid Input shouldn't have errors, got:\n %q", x.Error)
	}

	//With inner quote
	x = fullSetupTest("-s=\"walking \\\"Inner Quote\\\" fast\" echo")
	if x.Error != nil {
		t.Errorf("Valid Input shouldn't have errors, got:\n %q", x.Error)
	}

	//With quotes and space
	x = fullSetupTest("-s=\"walking \\\"Inner Quote\\\" fast\" echo")
	if x.Error != nil {
		t.Errorf("Valid Input shouldn't have errors, got:\n %q", x.Error)
	}

}

func TestRemoveCommand(t *testing.T) {
	versionUsed = 0
	c := initializeWithRootCmd()
	c.AddCommand(cmdVersion1)
	c.RemoveCommand(cmdVersion1)
	x := fullTester(c, "version")
	if x.Error == nil {
		t.Errorf("Removed command should not have been called\n")
		return
	}
}

func TestCommandWithoutSubcommands(t *testing.T) {
	c := initializeWithRootCmd()

	x := simpleTester(c, "")
	if x.Error != nil {
		t.Errorf("Calling command without subcommands should not have error: %v", x.Error)
		return
	}
}

func TestCommandWithoutSubcommandsWithArg(t *testing.T) {
	c := initializeWithRootCmd()
	expectedArgs := []string{"arg"}

	x := simpleTester(c, "arg")
	if x.Error != nil {
		t.Errorf("Calling command without subcommands but with arg should not have error: %v", x.Error)
		return
	}
	if !reflect.DeepEqual(expectedArgs, tr) {
		t.Errorf("Calling command without subcommands but with arg has wrong args: expected: %v, actual: %v", expectedArgs, tr)
		return
	}
}

func TestReplaceCommandWithRemove(t *testing.T) {
	versionUsed = 0
	c := initializeWithRootCmd()
	c.AddCommand(cmdVersion1)
	c.RemoveCommand(cmdVersion1)
	c.AddCommand(cmdVersion2)
	x := fullTester(c, "version")
	if x.Error != nil {
		t.Errorf("Valid Input shouldn't have errors, got:\n %q", x.Error)
		return
	}
	if versionUsed == 1 {
		t.Errorf("Removed command shouldn't be called\n")
	}
	if versionUsed != 2 {
		t.Errorf("Replacing command should have been called but didn't\n")
	}
}

func TestDeprecatedSub(t *testing.T) {
	c := fullSetupTest("deprecated")

	checkResultContains(t, c, cmdDeprecated.Deprecated)
}

func TestPreRun(t *testing.T) {
	noRRSetupTest("echo one two")
	if echoPre == nil || echoPersPre == nil {
		t.Error("PreRun or PersistentPreRun not called")
	}
	if rootPersPre != nil || timesPersPre != nil {
		t.Error("Wrong *Pre functions called!")
	}

	noRRSetupTest("echo times one two")
	if timesPersPre == nil {
		t.Error("PreRun or PersistentPreRun not called")
	}
	if echoPre != nil || echoPersPre != nil || rootPersPre != nil {
		t.Error("Wrong *Pre functions called!")
	}

	noRRSetupTest("print one two")
	if rootPersPre == nil {
		t.Error("Parent PersistentPreRun not called but should not have been")
	}
	if echoPre != nil || echoPersPre != nil || timesPersPre != nil {
		t.Error("Wrong *Pre functions called!")
	}
}

// Check if cmdEchoSub gets PersistentPreRun from rootCmd even if is added last
func TestPeristentPreRunPropagation(t *testing.T) {
	rootCmd := initialize()

	// First add the cmdEchoSub to cmdPrint
	cmdPrint.AddCommand(cmdEchoSub)
	// Now add cmdPrint to rootCmd
	rootCmd.AddCommand(cmdPrint)

	rootCmd.SetArgs(strings.Split("print echosub lala", " "))
	rootCmd.Execute()

	if rootPersPre == nil || len(rootPersPre) == 0 || rootPersPre[0] != "lala" {
		t.Error("RootCmd PersistentPreRun not called but should have been")
	}
}

func TestGlobalNormFuncPropagation(t *testing.T) {
	normFunc := func(f *pflag.FlagSet, name string) pflag.NormalizedName {
		return pflag.NormalizedName(name)
	}

	rootCmd := initialize()
	rootCmd.SetGlobalNormalizationFunc(normFunc)
	if reflect.ValueOf(normFunc) != reflect.ValueOf(rootCmd.GlobalNormalizationFunc()) {
		t.Error("rootCmd seems to have a wrong normalization function")
	}

	// First add the cmdEchoSub to cmdPrint
	cmdPrint.AddCommand(cmdEchoSub)
	if cmdPrint.GlobalNormalizationFunc() != nil && cmdEchoSub.GlobalNormalizationFunc() != nil {
		t.Error("cmdPrint and cmdEchoSub should had no normalization functions")
	}

	// Now add cmdPrint to rootCmd
	rootCmd.AddCommand(cmdPrint)
	if reflect.ValueOf(cmdPrint.GlobalNormalizationFunc()).Pointer() != reflect.ValueOf(rootCmd.GlobalNormalizationFunc()).Pointer() ||
		reflect.ValueOf(cmdEchoSub.GlobalNormalizationFunc()).Pointer() != reflect.ValueOf(rootCmd.GlobalNormalizationFunc()).Pointer() {
		t.Error("cmdPrint and cmdEchoSub should had the normalization function of rootCmd")
	}
}
