// PowerShell completions are based on the amazing work from clap:
// https://github.com/clap-rs/clap/blob/3294d18efe5f264d12c9035f404c7d189d4824e1/src/completions/powershell.rs
//
// The generated scripts require PowerShell v5.0+ (which comes Windows 10, but
// can be downloaded separately for windows 7 or 8.1).

package cobra

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/spf13/pflag"
)

var powerShellCompletionTemplate = `using namespace System.Management.Automation
using namespace System.Management.Automation.Language
Register-ArgumentCompleter -Native -CommandName '%s' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $commandElements = $commandAst.CommandElements
    $command = @(
        '%s'
        for ($i = 1; $i -lt $commandElements.Count; $i++) {
            $element = $commandElements[$i]
            if ($element -isnot [StringConstantExpressionAst] -or
                $element.StringConstantType -ne [StringConstantType]::BareWord -or
                $element.Value.StartsWith('-')) {
                break
            }
            $element.Value
        }
    ) -join ';'
    $completions = @(switch ($command) {%s
    })
    $completions.Where{ $_.CompletionText -like "$wordToComplete*" } |
        Sort-Object -Property ListItemText
}`

func generatePowerShellSubcommandCases(out io.Writer, cmd *Command, previousCommandName string) {
	var cmdName string
	if previousCommandName == "" {
		cmdName = cmd.Name()
	} else {
		cmdName = fmt.Sprintf("%s;%s", previousCommandName, cmd.Name())
	}

	fmt.Fprintf(out, "\n        '%s' {", cmdName)

	cmd.Flags().VisitAll(func(flag *pflag.Flag) {
		if nonCompletableFlag(flag) {
			return
		}
		usage := escapeStringForPowerShell(flag.Usage)
		if len(flag.Shorthand) > 0 {
			fmt.Fprintf(out, "\n            [CompletionResult]::new('-%s', '%s', [CompletionResultType]::ParameterName, '%s')", flag.Shorthand, flag.Shorthand, usage)
		}
		fmt.Fprintf(out, "\n            [CompletionResult]::new('--%s', '%s', [CompletionResultType]::ParameterName, '%s')", flag.Name, flag.Name, usage)
	})

	for _, subCmd := range cmd.Commands() {
		usage := escapeStringForPowerShell(subCmd.Short)
		fmt.Fprintf(out, "\n            [CompletionResult]::new('%s', '%s', [CompletionResultType]::ParameterValue, '%s')", subCmd.Name(), subCmd.Name(), usage)
	}

	fmt.Fprint(out, "\n            break\n        }")

	for _, subCmd := range cmd.Commands() {
		generatePowerShellSubcommandCases(out, subCmd, cmdName)
	}
}

func escapeStringForPowerShell(s string) string {
	return strings.Replace(s, "'", "''", -1)
}

// GenPowerShellCompletion generates PowerShell completion file and writes to the passed writer.
func (c *Command) GenPowerShellCompletion(w io.Writer) error {
	buf := new(bytes.Buffer)

	var subCommandCases bytes.Buffer
	generatePowerShellSubcommandCases(&subCommandCases, c, "")
	fmt.Fprintf(buf, powerShellCompletionTemplate, c.Name(), c.Name(), subCommandCases.String())

	_, err := buf.WriteTo(w)
	return err
}

// GenPowerShellCompletionFile generates PowerShell completion file.
func (c *Command) GenPowerShellCompletionFile(filename string) error {
	outFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer outFile.Close()

	return c.GenPowerShellCompletion(outFile)
}
