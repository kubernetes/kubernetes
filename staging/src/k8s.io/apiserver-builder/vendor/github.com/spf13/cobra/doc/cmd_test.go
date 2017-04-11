package doc

import (
	"bytes"
	"fmt"
	"runtime"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

var flagb1, flagb2, flagb3, flagbr, flagbp bool
var flags1, flags2a, flags2b, flags3 string
var flagi1, flagi2, flagi3, flagir int

const strtwoParentHelp = "help message for parent flag strtwo"
const strtwoChildHelp = "help message for child flag strtwo"

var cmdEcho = &cobra.Command{
	Use:     "echo [string to echo]",
	Aliases: []string{"say"},
	Short:   "Echo anything to the screen",
	Long:    `an utterly useless command for testing.`,
	Example: "Just run cobra-test echo",
}

var cmdEchoSub = &cobra.Command{
	Use:   "echosub [string to print]",
	Short: "second sub command for echo",
	Long:  `an absolutely utterly useless command for testing gendocs!.`,
	Run:   func(cmd *cobra.Command, args []string) {},
}

var cmdDeprecated = &cobra.Command{
	Use:        "deprecated [can't do anything here]",
	Short:      "A command which is deprecated",
	Long:       `an absolutely utterly useless command for testing deprecation!.`,
	Deprecated: "Please use echo instead",
}

var cmdTimes = &cobra.Command{
	Use:              "times [# times] [string to echo]",
	SuggestFor:       []string{"counts"},
	Short:            "Echo anything to the screen more times",
	Long:             `a slightly useless command for testing.`,
	PersistentPreRun: func(cmd *cobra.Command, args []string) {},
	Run:              func(cmd *cobra.Command, args []string) {},
}

var cmdPrint = &cobra.Command{
	Use:   "print [string to print]",
	Short: "Print anything to the screen",
	Long:  `an absolutely utterly useless command for testing.`,
}

var cmdRootNoRun = &cobra.Command{
	Use:   "cobra-test",
	Short: "The root can run its own function",
	Long:  "The root description for help",
}

var cmdRootSameName = &cobra.Command{
	Use:   "print",
	Short: "Root with the same name as a subcommand",
	Long:  "The root description for help",
}

var cmdRootWithRun = &cobra.Command{
	Use:   "cobra-test",
	Short: "The root can run its own function",
	Long:  "The root description for help",
}

var cmdSubNoRun = &cobra.Command{
	Use:   "subnorun",
	Short: "A subcommand without a Run function",
	Long:  "A long output about a subcommand without a Run function",
}

var cmdVersion1 = &cobra.Command{
	Use:   "version",
	Short: "Print the version number",
	Long:  `First version of the version command`,
}

var cmdVersion2 = &cobra.Command{
	Use:   "version",
	Short: "Print the version number",
	Long:  `Second version of the version command`,
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

func initializeWithRootCmd() *cobra.Command {
	cmdRootWithRun.ResetCommands()
	flagInit()
	cmdRootWithRun.Flags().BoolVarP(&flagbr, "boolroot", "b", false, "help message for flag boolroot")
	cmdRootWithRun.Flags().IntVarP(&flagir, "introot", "i", 321, "help message for flag introot")
	return cmdRootWithRun
}

func checkStringContains(t *testing.T, found, expected string) {
	if !strings.Contains(found, expected) {
		logErr(t, found, expected)
	}
}

func checkStringOmits(t *testing.T, found, expected string) {
	if strings.Contains(found, expected) {
		logErr(t, found, expected)
	}
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
