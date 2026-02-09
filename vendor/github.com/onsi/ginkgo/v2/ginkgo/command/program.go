package command

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/onsi/ginkgo/v2/formatter"
	"github.com/onsi/ginkgo/v2/types"
)

type Program struct {
	Name               string
	Heading            string
	Commands           []Command
	DefaultCommand     Command
	DeprecatedCommands []DeprecatedCommand

	//For testing - leave as nil in production
	OutWriter io.Writer
	ErrWriter io.Writer
	Exiter    func(code int)
}

type DeprecatedCommand struct {
	Name        string
	Deprecation types.Deprecation
}

func (p Program) RunAndExit(osArgs []string) {
	var command Command
	deprecationTracker := types.NewDeprecationTracker()
	if p.Exiter == nil {
		p.Exiter = os.Exit
	}
	if p.OutWriter == nil {
		p.OutWriter = formatter.ColorableStdOut
	}
	if p.ErrWriter == nil {
		p.ErrWriter = formatter.ColorableStdErr
	}

	defer func() {
		exitCode := 0

		if r := recover(); r != nil {
			details, ok := r.(AbortDetails)
			if !ok {
				panic(r)
			}

			if details.Error != nil {
				fmt.Fprintln(p.ErrWriter, formatter.F("{{red}}{{bold}}%s %s{{/}} {{red}}failed{{/}}", p.Name, command.Name))
				fmt.Fprintln(p.ErrWriter, formatter.Fi(1, details.Error.Error()))
			}
			if details.EmitUsage {
				if details.Error != nil {
					fmt.Fprintln(p.ErrWriter, "")
				}
				command.EmitUsage(p.ErrWriter)
			}
			exitCode = details.ExitCode
		}

		command.Flags.ValidateDeprecations(deprecationTracker)
		if deprecationTracker.DidTrackDeprecations() {
			fmt.Fprintln(p.ErrWriter, deprecationTracker.DeprecationsReport())
		}
		p.Exiter(exitCode)
	}()

	args, additionalArgs := []string{}, []string{}

	foundDelimiter := false
	for _, arg := range osArgs[1:] {
		if !foundDelimiter {
			if arg == "--" {
				foundDelimiter = true
				continue
			}
		}

		if foundDelimiter {
			additionalArgs = append(additionalArgs, arg)
		} else {
			args = append(args, arg)
		}
	}

	command = p.DefaultCommand
	if len(args) > 0 {
		p.handleHelpRequestsAndExit(p.OutWriter, args)
		if command.Name == args[0] {
			args = args[1:]
		} else {
			for _, deprecatedCommand := range p.DeprecatedCommands {
				if deprecatedCommand.Name == args[0] {
					deprecationTracker.TrackDeprecation(deprecatedCommand.Deprecation)
					return
				}
			}
			for _, tryCommand := range p.Commands {
				if tryCommand.Name == args[0] {
					command, args = tryCommand, args[1:]
					break
				}
			}
		}
	}

	command.Run(args, additionalArgs)
}

func (p Program) handleHelpRequestsAndExit(writer io.Writer, args []string) {
	if len(args) == 0 {
		return
	}

	matchesHelpFlag := func(args ...string) bool {
		for _, arg := range args {
			if arg == "--help" || arg == "-help" || arg == "-h" || arg == "--h" {
				return true
			}
		}
		return false
	}
	if len(args) == 1 {
		if args[0] == "help" || matchesHelpFlag(args[0]) {
			p.EmitUsage(writer)
			Abort(AbortDetails{})
		}
	} else {
		var name string
		if args[0] == "help" || matchesHelpFlag(args[0]) {
			name = args[1]
		} else if matchesHelpFlag(args[1:]...) {
			name = args[0]
		} else {
			return
		}

		if p.DefaultCommand.Name == name || p.Name == name {
			p.DefaultCommand.EmitUsage(writer)
			Abort(AbortDetails{})
		}
		for _, command := range p.Commands {
			if command.Name == name {
				command.EmitUsage(writer)
				Abort(AbortDetails{})
			}
		}

		fmt.Fprintln(writer, formatter.F("{{red}}Unknown Command: {{bold}}%s{{/}}", name))
		fmt.Fprintln(writer, "")
		p.EmitUsage(writer)
		Abort(AbortDetails{ExitCode: 1})
	}
}

func (p Program) EmitUsage(writer io.Writer) {
	fmt.Fprintln(writer, formatter.F(p.Heading))
	fmt.Fprintln(writer, formatter.F("{{gray}}%s{{/}}", strings.Repeat("-", len(p.Heading))))
	fmt.Fprintln(writer, formatter.F("For usage information for a command, run {{bold}}%s help COMMAND{{/}}.", p.Name))
	fmt.Fprintln(writer, formatter.F("For usage information for the default command, run {{bold}}%s help %s{{/}} or {{bold}}%s help %s{{/}}.", p.Name, p.Name, p.Name, p.DefaultCommand.Name))
	fmt.Fprintln(writer, "")
	fmt.Fprintln(writer, formatter.F("The following commands are available:"))

	fmt.Fprintln(writer, formatter.Fi(1, "{{bold}}%s{{/}} or %s {{bold}}%s{{/}} - {{gray}}%s{{/}}", p.Name, p.Name, p.DefaultCommand.Name, p.DefaultCommand.Usage))
	if p.DefaultCommand.ShortDoc != "" {
		fmt.Fprintln(writer, formatter.Fi(2, p.DefaultCommand.ShortDoc))
	}

	for _, command := range p.Commands {
		fmt.Fprintln(writer, formatter.Fi(1, "{{bold}}%s{{/}} - {{gray}}%s{{/}}", command.Name, command.Usage))
		if command.ShortDoc != "" {
			fmt.Fprintln(writer, formatter.Fi(2, command.ShortDoc))
		}
	}
}
