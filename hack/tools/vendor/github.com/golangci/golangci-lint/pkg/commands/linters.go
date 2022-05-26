package commands

import (
	"log"
	"os"

	"github.com/fatih/color"
	"github.com/spf13/cobra"

	"github.com/golangci/golangci-lint/pkg/exitcodes"
	"github.com/golangci/golangci-lint/pkg/lint/linter"
)

func (e *Executor) initLinters() {
	e.lintersCmd = &cobra.Command{
		Use:   "linters",
		Short: "List current linters configuration",
		Run:   e.executeLinters,
	}
	e.rootCmd.AddCommand(e.lintersCmd)
	e.initRunConfiguration(e.lintersCmd)
}

// executeLinters runs the 'linters' CLI command, which displays the supported linters.
func (e *Executor) executeLinters(_ *cobra.Command, args []string) {
	if len(args) != 0 {
		e.log.Fatalf("Usage: golangci-lint linters")
	}

	enabledLintersMap, err := e.EnabledLintersSet.GetEnabledLintersMap()
	if err != nil {
		log.Fatalf("Can't get enabled linters: %s", err)
	}

	color.Green("Enabled by your configuration linters:\n")
	enabledLinters := make([]*linter.Config, 0, len(enabledLintersMap))
	for _, linter := range enabledLintersMap {
		enabledLinters = append(enabledLinters, linter)
	}
	printLinterConfigs(enabledLinters)

	var disabledLCs []*linter.Config
	for _, lc := range e.DBManager.GetAllSupportedLinterConfigs() {
		if enabledLintersMap[lc.Name()] == nil {
			disabledLCs = append(disabledLCs, lc)
		}
	}

	color.Red("\nDisabled by your configuration linters:\n")
	printLinterConfigs(disabledLCs)

	os.Exit(exitcodes.Success)
}
