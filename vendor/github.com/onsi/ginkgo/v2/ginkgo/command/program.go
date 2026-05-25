package command

import (
	"bufio"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
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

type completionOptions = struct {
	Complete bool
	Install  bool
}

func (p *Program) BuildCompletionCommand() Command {
	opts := completionOptions{}
	flags, err := types.NewGinkgoFlagSet(
		types.GinkgoFlags{
			{Name: "complete", KeyPath: "Complete", Usage: "Generate completion for arguments after --"},
			{Name: "install", KeyPath: "Install", Usage: "Install shell completion script into $XDG_DATA_HOME, ~/.local/share"},
		},
		&opts,
		types.GinkgoFlagSections{},
	)
	if err != nil {
		panic(err)
	}
	return Command{
		Name:     "completion",
		Usage:    "ginkgo completion <FLAGS> <SHELL> [-- <COMPLETE>]",
		Flags:    flags,
		ShortDoc: "Generate shell completion",
		Documentation: `To use install completion script for your shell (bash, fish, zsh).
Or load completion code by: {{bold}}source <(ginkgo completion <SHELL>){{/}}.`,
		Command: func(args []string, completeArgs []string) {
			p.handleCompletionAndExit(args, completeArgs, opts)
		},
	}
}

func (p Program) generateShellCompletionScript(shell string) (scriptPath string, script string) {
	switch shell {
	case "bash":
		scriptPath = fmt.Sprintf("bash-completion/completions/%s", p.Name)
		script = fmt.Sprintf(`__%s_complete_bash() {
  mapfile -t COMPREPLY < <("${COMP_WORDS[0]}" completion --complete bash -- "${COMP_WORDS[@]:1:COMP_CWORD}")
}
complete -o bashdefault -o default -F __%[1]s_complete_bash %[1]s
`, p.Name)

	case "fish":
		scriptPath = fmt.Sprintf("fish/vendor_completions.d/%s.fish", p.Name)
		script = fmt.Sprintf(`function __fish_%[1]s_complete
  set -l args (commandline -opc) (commandline -ct)
  set -e args[1]
  %[1]s completion --complete fish -- $args
end
complete -c %[1]s -a "(__fish_%[1]s_complete)"
`, p.Name)

	case "zsh":
		scriptPath = fmt.Sprintf("zsh/site-functions/_%s", p.Name)
		script = fmt.Sprintf(`#compdef %[1]s
_%[1]s() {
  local -a completions
  completions=(${(f)"$("${words[1]}" completion --complete zsh -- "${words[@]:1:$((CURRENT-1))}")"})
  if (( ${#completions[@]} )); then
    _describe 'completions' completions
  else
    _default
  fi
}
compdef _%[1]s %[1]s
if [ "$funcstack[1]" = "_%[1]s" ]; then
  _%[1]s
fi
`, p.Name)

	case "":
		AbortWithUsage("Shell is not specified")
	default:
		AbortWith("Shell %q is not supported yet. Choose: bash, fish, zsh", shell)
	}

	return scriptPath, script
}

func (p Program) handleCompletionAndExit(args, completeArgs []string, opts completionOptions) {
	writer := p.OutWriter
	if writer == nil {
		writer = os.Stdout
	}
	buffer := bufio.NewWriter(writer)
	defer buffer.Flush()

	var shell string
	if len(args) > 0 {
		shell = args[0]
	}

	if !opts.Complete {
		scriptPath, script := p.generateShellCompletionScript(shell)
		if opts.Install {
			dataHomeDir := os.Getenv("XDG_DATA_HOME")
			if dataHomeDir == "" {
				userHomeDir, err := os.UserHomeDir()
				AbortIfError("Failed to find home", err)
				dataHomeDir = filepath.Join(userHomeDir, ".local/share")
			}
			scriptPath = filepath.Join(dataHomeDir, scriptPath)
			fmt.Fprintf(buffer, "Installing completion script: %v\n", scriptPath)
			err := os.WriteFile(scriptPath, []byte(script), 0644)
			AbortIfError("Failed to install completion script", err)
		} else {
			buffer.Write([]byte(script))
		}
		Abort(AbortDetails{})
	}

	var lastArg string
	var result map[string]string
	if len(completeArgs) > 0 {
		lastArg = completeArgs[len(completeArgs)-1]
	}

	if delim := slices.Index(completeArgs, "--"); delim >= 0 && delim != len(completeArgs)-1 {
		// No completion for pass-through arguments after "--"
	} else if len(lastArg) > 0 && lastArg[0] == '-' {
		// Complete flags
		cmd := &p.DefaultCommand
		for i := range p.Commands {
			if p.Commands[i].Name == completeArgs[0] {
				cmd = &p.Commands[i]
				break
			}
		}
		result = cmd.Flags.Completion(lastArg)
	} else if len(completeArgs) <= 1 {
		// Complete commands
		result = make(map[string]string, len(p.Commands)+1)
		for _, cmd := range append(p.Commands, p.DefaultCommand) {
			if strings.HasPrefix(cmd.Name, lastArg) {
				result[cmd.Name] = cmd.Usage
			}
		}
	}

	width := 0
	for suggest := range result {
		width = max(width, len(suggest))
	}

	for _, suggest := range slices.Sorted(maps.Keys(result)) {
		usage := result[suggest]
		switch {
		case shell == "bash" && usage != "" && len(result) > 1:
			fmt.Fprintf(buffer, "%*s (%s)\n", -width-2, suggest, usage)
		case shell == "fish":
			fmt.Fprintf(buffer, "%s\t%s\n", suggest, usage)
		case shell == "zsh":
			fmt.Fprintf(buffer, "%s:%s\n", suggest, usage)
		default:
			fmt.Fprintln(buffer, suggest)
		}
	}

	Abort(AbortDetails{})
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
