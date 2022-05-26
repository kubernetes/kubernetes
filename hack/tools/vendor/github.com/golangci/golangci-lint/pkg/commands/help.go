package commands

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/fatih/color"
	"github.com/spf13/cobra"

	"github.com/golangci/golangci-lint/pkg/exitcodes"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
	"github.com/golangci/golangci-lint/pkg/logutils"
)

func (e *Executor) initHelp() {
	helpCmd := &cobra.Command{
		Use:   "help",
		Short: "Help",
		Run: func(cmd *cobra.Command, args []string) {
			if len(args) != 0 {
				e.log.Fatalf("Usage: golangci-lint help")
			}
			if err := cmd.Help(); err != nil {
				e.log.Fatalf("Can't run help: %s", err)
			}
		},
	}
	e.rootCmd.SetHelpCommand(helpCmd)

	lintersHelpCmd := &cobra.Command{
		Use:   "linters",
		Short: "Help about linters",
		Run:   e.executeLintersHelp,
	}
	helpCmd.AddCommand(lintersHelpCmd)
}

func printLinterConfigs(lcs []*linter.Config) {
	sort.Slice(lcs, func(i, j int) bool {
		return strings.Compare(lcs[i].Name(), lcs[j].Name()) < 0
	})
	for _, lc := range lcs {
		altNamesStr := ""
		if len(lc.AlternativeNames) != 0 {
			altNamesStr = fmt.Sprintf(" (%s)", strings.Join(lc.AlternativeNames, ", "))
		}

		// If the linter description spans multiple lines, truncate everything following the first newline
		linterDescription := lc.Linter.Desc()
		firstNewline := strings.IndexRune(linterDescription, '\n')
		if firstNewline > 0 {
			linterDescription = linterDescription[:firstNewline]
		}

		deprecatedMark := ""
		if lc.IsDeprecated() {
			deprecatedMark = " [" + color.RedString("deprecated") + "]"
		}

		fmt.Fprintf(logutils.StdOut, "%s%s%s: %s [fast: %t, auto-fix: %t]\n", color.YellowString(lc.Name()),
			altNamesStr, deprecatedMark, linterDescription, !lc.IsSlowLinter(), lc.CanAutoFix)
	}
}

func (e *Executor) executeLintersHelp(_ *cobra.Command, args []string) {
	if len(args) != 0 {
		e.log.Fatalf("Usage: golangci-lint help linters")
	}

	var enabledLCs, disabledLCs []*linter.Config
	for _, lc := range e.DBManager.GetAllSupportedLinterConfigs() {
		if lc.EnabledByDefault {
			enabledLCs = append(enabledLCs, lc)
		} else {
			disabledLCs = append(disabledLCs, lc)
		}
	}

	color.Green("Enabled by default linters:\n")
	printLinterConfigs(enabledLCs)
	color.Red("\nDisabled by default linters:\n")
	printLinterConfigs(disabledLCs)

	color.Green("\nLinters presets:")
	for _, p := range e.DBManager.AllPresets() {
		linters := e.DBManager.GetAllLinterConfigsForPreset(p)
		linterNames := make([]string, 0, len(linters))
		for _, lc := range linters {
			linterNames = append(linterNames, lc.Name())
		}
		sort.Strings(linterNames)
		fmt.Fprintf(logutils.StdOut, "%s: %s\n", color.YellowString(p), strings.Join(linterNames, ", "))
	}

	os.Exit(exitcodes.Success)
}
