package cobra

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

var _ = fmt.Println

var tp, te, tt, t1 []string
var flagb1, flagb2, flagb3, flagbr bool
var flags1, flags2, flags3 string
var flagi1, flagi2, flagi3, flagir int
var globalFlag1 bool
var flagEcho, rootcalled bool

var cmdPrint = &Command{
	Use:   "print [string to print]",
	Short: "Print anything to the screen",
	Long:  `an utterly useless command for testing.`,
	Run: func(cmd *Command, args []string) {
		tp = args
	},
}

var cmdEcho = &Command{
	Use:     "echo [string to echo]",
	Aliases: []string{"say"},
	Short:   "Echo anything to the screen",
	Long:    `an utterly useless command for testing.`,
	Run: func(cmd *Command, args []string) {
		te = args
	},
}

var cmdTimes = &Command{
	Use:   "times [# times] [string to echo]",
	Short: "Echo anything to the screen more times",
	Long:  `an slightly useless command for testing.`,
	Run: func(cmd *Command, args []string) {
		tt = args
	},
}

var cmdRootNoRun = &Command{
	Use:   "cobra-test",
	Short: "The root can run it's own function",
	Long:  "The root description for help",
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
		rootcalled = true
	},
}

func flagInit() {
	cmdEcho.ResetFlags()
	cmdPrint.ResetFlags()
	cmdTimes.ResetFlags()
	cmdRootNoRun.ResetFlags()
	cmdRootSameName.ResetFlags()
	cmdRootWithRun.ResetFlags()
	cmdEcho.Flags().IntVarP(&flagi1, "intone", "i", 123, "help message for flag intone")
	cmdTimes.Flags().IntVarP(&flagi2, "inttwo", "j", 234, "help message for flag inttwo")
	cmdPrint.Flags().IntVarP(&flagi3, "intthree", "i", 345, "help message for flag intthree")
	cmdEcho.PersistentFlags().StringVarP(&flags1, "strone", "s", "one", "help message for flag strone")
	cmdTimes.PersistentFlags().StringVarP(&flags2, "strtwo", "t", "two", "help message for flag strtwo")
	cmdPrint.PersistentFlags().StringVarP(&flags3, "strthree", "s", "three", "help message for flag strthree")
	cmdEcho.Flags().BoolVarP(&flagb1, "boolone", "b", true, "help message for flag boolone")
	cmdTimes.Flags().BoolVarP(&flagb2, "booltwo", "c", false, "help message for flag booltwo")
	cmdPrint.Flags().BoolVarP(&flagb3, "boolthree", "b", true, "help message for flag boolthree")
}

func commandInit() {
	cmdEcho.ResetCommands()
	cmdPrint.ResetCommands()
	cmdTimes.ResetCommands()
	cmdRootNoRun.ResetCommands()
	cmdRootSameName.ResetCommands()
	cmdRootWithRun.ResetCommands()
}

func initialize() *Command {
	tt, tp, te = nil, nil, nil
	var c = cmdRootNoRun
	flagInit()
	commandInit()
	return c
}

func initializeWithSameName() *Command {
	tt, tp, te = nil, nil, nil
	var c = cmdRootSameName
	flagInit()
	commandInit()
	return c
}

func initializeWithRootCmd() *Command {
	cmdRootWithRun.ResetCommands()
	tt, tp, te, rootcalled = nil, nil, nil, false
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
	c.AddCommand(cmdPrint, cmdEcho)
	c.SetArgs(strings.Split(input, " "))

	err := c.Execute()
	output := buf.String()

	return resulter{err, output, c}
}

func checkResultContains(t *testing.T, x resulter, check string) {
	if !strings.Contains(x.Output, check) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", check, x.Output)
	}
}

func checkResultOmits(t *testing.T, x resulter, check string) {
	if strings.Contains(x.Output, check) {
		t.Errorf("Unexpected response.\nExpecting to omit: \n %q\nGot:\n %q\n", check, x.Output)
	}
}

func checkOutputContains(t *testing.T, c *Command, check string) {
	buf := new(bytes.Buffer)
	c.SetOutput(buf)
	c.Execute()

	if !strings.Contains(buf.String(), check) {
		t.Errorf("Unexpected response.\nExpecting to contain: \n %q\nGot:\n %q\n", check, buf.String())
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

	if !strings.Contains(r.Output, "intone=123") {
		t.Errorf("Wrong error message displayed, \n %s", r.Output)
	}

	// Testing flag with invalid input
	r = noRRSetupTest("echo -i10E")

	if r.Error == nil {
		t.Errorf("invalid input should generate error")
	}

	if !strings.Contains(r.Output, "invalid argument \"10E\" for -i10E") {
		t.Errorf("Wrong error message displayed, \n %s", r.Output)
	}
}

func TestTrailingCommandFlags(t *testing.T) {
	x := fullSetupTest("echo two -x")

	if x.Error == nil {
		t.Errorf("invalid flag should generate error")
	}
}

func TestPersistentFlags(t *testing.T) {
	fullSetupTest("echo -s something more here")

	// persistentFlag should act like normal flag on it's own command
	if strings.Join(te, " ") != "more here" {
		t.Errorf("flags didn't leave proper args remaining..%s given", te)
	}

	// persistentFlag should act like normal flag on it's own command
	if flags1 != "something" {
		t.Errorf("string flag didn't get correct value, had %v", flags1)
	}

	fullSetupTest("echo times -s again -c test here")

	if strings.Join(tt, " ") != "test here" {
		t.Errorf("flags didn't leave proper args remaining..%s given", tt)
	}

	if flags1 != "again" {
		t.Errorf("string flag didn't get correct value, had %v", flags1)
	}

	if flagb2 != true {
		t.Errorf("local flag not parsed correctly. Expected false, had %v", flagb2)
	}
}

func TestHelpCommand(t *testing.T) {
	c := fullSetupTest("help echo")
	checkResultContains(t, c, cmdEcho.Long)

	r := fullSetupTest("help echo times")
	checkResultContains(t, r, cmdTimes.Long)
}

func TestRunnableRootCommand(t *testing.T) {
	fullSetupTest("")

	if rootcalled != true {
		t.Errorf("Root Function was not called")
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
	if !strings.Contains(x.Output, "invalid argument \"10E\" for -i10E") {
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
