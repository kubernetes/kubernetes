package cobra

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/pflag"
)

const (
	// ShellCompRequestCmd is the name of the hidden command that is used to request
	// completion results from the program.  It is used by the shell completion scripts.
	ShellCompRequestCmd = "__complete"
	// ShellCompNoDescRequestCmd is the name of the hidden command that is used to request
	// completion results without their description.  It is used by the shell completion scripts.
	ShellCompNoDescRequestCmd = "__completeNoDesc"
)

// Global map of flag completion functions.
var flagCompletionFunctions = map[*pflag.Flag]func(cmd *Command, args []string, toComplete string) ([]string, ShellCompDirective){}

// ShellCompDirective is a bit map representing the different behaviors the shell
// can be instructed to have once completions have been provided.
type ShellCompDirective int

const (
	// ShellCompDirectiveError indicates an error occurred and completions should be ignored.
	ShellCompDirectiveError ShellCompDirective = 1 << iota

	// ShellCompDirectiveNoSpace indicates that the shell should not add a space
	// after the completion even if there is a single completion provided.
	ShellCompDirectiveNoSpace

	// ShellCompDirectiveNoFileComp indicates that the shell should not provide
	// file completion even when no completion is provided.
	// This currently does not work for zsh or bash < 4
	ShellCompDirectiveNoFileComp

	// ShellCompDirectiveFilterFileExt indicates that the provided completions
	// should be used as file extension filters.
	// For flags, using Command.MarkFlagFilename() and Command.MarkPersistentFlagFilename()
	// is a shortcut to using this directive explicitly.  The BashCompFilenameExt
	// annotation can also be used to obtain the same behavior for flags.
	ShellCompDirectiveFilterFileExt

	// ShellCompDirectiveFilterDirs indicates that only directory names should
	// be provided in file completion.  To request directory names within another
	// directory, the returned completions should specify the directory within
	// which to search.  The BashCompSubdirsInDir annotation can be used to
	// obtain the same behavior but only for flags.
	ShellCompDirectiveFilterDirs

	// ===========================================================================

	// All directives using iota should be above this one.
	// For internal use.
	shellCompDirectiveMaxValue

	// ShellCompDirectiveDefault indicates to let the shell perform its default
	// behavior after completions have been provided.
	// This one must be last to avoid messing up the iota count.
	ShellCompDirectiveDefault ShellCompDirective = 0
)

// RegisterFlagCompletionFunc should be called to register a function to provide completion for a flag.
func (c *Command) RegisterFlagCompletionFunc(flagName string, f func(cmd *Command, args []string, toComplete string) ([]string, ShellCompDirective)) error {
	flag := c.Flag(flagName)
	if flag == nil {
		return fmt.Errorf("RegisterFlagCompletionFunc: flag '%s' does not exist", flagName)
	}
	if _, exists := flagCompletionFunctions[flag]; exists {
		return fmt.Errorf("RegisterFlagCompletionFunc: flag '%s' already registered", flagName)
	}
	flagCompletionFunctions[flag] = f
	return nil
}

// Returns a string listing the different directive enabled in the specified parameter
func (d ShellCompDirective) string() string {
	var directives []string
	if d&ShellCompDirectiveError != 0 {
		directives = append(directives, "ShellCompDirectiveError")
	}
	if d&ShellCompDirectiveNoSpace != 0 {
		directives = append(directives, "ShellCompDirectiveNoSpace")
	}
	if d&ShellCompDirectiveNoFileComp != 0 {
		directives = append(directives, "ShellCompDirectiveNoFileComp")
	}
	if d&ShellCompDirectiveFilterFileExt != 0 {
		directives = append(directives, "ShellCompDirectiveFilterFileExt")
	}
	if d&ShellCompDirectiveFilterDirs != 0 {
		directives = append(directives, "ShellCompDirectiveFilterDirs")
	}
	if len(directives) == 0 {
		directives = append(directives, "ShellCompDirectiveDefault")
	}

	if d >= shellCompDirectiveMaxValue {
		return fmt.Sprintf("ERROR: unexpected ShellCompDirective value: %d", d)
	}
	return strings.Join(directives, ", ")
}

// Adds a special hidden command that can be used to request custom completions.
func (c *Command) initCompleteCmd(args []string) {
	completeCmd := &Command{
		Use:                   fmt.Sprintf("%s [command-line]", ShellCompRequestCmd),
		Aliases:               []string{ShellCompNoDescRequestCmd},
		DisableFlagsInUseLine: true,
		Hidden:                true,
		DisableFlagParsing:    true,
		Args:                  MinimumNArgs(1),
		Short:                 "Request shell completion choices for the specified command-line",
		Long: fmt.Sprintf("%[2]s is a special command that is used by the shell completion logic\n%[1]s",
			"to request completion choices for the specified command-line.", ShellCompRequestCmd),
		Run: func(cmd *Command, args []string) {
			finalCmd, completions, directive, err := cmd.getCompletions(args)
			if err != nil {
				CompErrorln(err.Error())
				// Keep going for multiple reasons:
				// 1- There could be some valid completions even though there was an error
				// 2- Even without completions, we need to print the directive
			}

			noDescriptions := (cmd.CalledAs() == ShellCompNoDescRequestCmd)
			for _, comp := range completions {
				if noDescriptions {
					// Remove any description that may be included following a tab character.
					comp = strings.Split(comp, "\t")[0]
				}

				// Make sure we only write the first line to the output.
				// This is needed if a description contains a linebreak.
				// Otherwise the shell scripts will interpret the other lines as new flags
				// and could therefore provide a wrong completion.
				comp = strings.Split(comp, "\n")[0]

				// Finally trim the completion.  This is especially important to get rid
				// of a trailing tab when there are no description following it.
				// For example, a sub-command without a description should not be completed
				// with a tab at the end (or else zsh will show a -- following it
				// although there is no description).
				comp = strings.TrimSpace(comp)

				// Print each possible completion to stdout for the completion script to consume.
				fmt.Fprintln(finalCmd.OutOrStdout(), comp)
			}

			if directive >= shellCompDirectiveMaxValue {
				directive = ShellCompDirectiveDefault
			}

			// As the last printout, print the completion directive for the completion script to parse.
			// The directive integer must be that last character following a single colon (:).
			// The completion script expects :<directive>
			fmt.Fprintf(finalCmd.OutOrStdout(), ":%d\n", directive)

			// Print some helpful info to stderr for the user to understand.
			// Output from stderr must be ignored by the completion script.
			fmt.Fprintf(finalCmd.ErrOrStderr(), "Completion ended with directive: %s\n", directive.string())
		},
	}
	c.AddCommand(completeCmd)
	subCmd, _, err := c.Find(args)
	if err != nil || subCmd.Name() != ShellCompRequestCmd {
		// Only create this special command if it is actually being called.
		// This reduces possible side-effects of creating such a command;
		// for example, having this command would cause problems to a
		// cobra program that only consists of the root command, since this
		// command would cause the root command to suddenly have a subcommand.
		c.RemoveCommand(completeCmd)
	}
}

func (c *Command) getCompletions(args []string) (*Command, []string, ShellCompDirective, error) {
	// The last argument, which is not completely typed by the user,
	// should not be part of the list of arguments
	toComplete := args[len(args)-1]
	trimmedArgs := args[:len(args)-1]

	var finalCmd *Command
	var finalArgs []string
	var err error
	// Find the real command for which completion must be performed
	// check if we need to traverse here to parse local flags on parent commands
	if c.Root().TraverseChildren {
		finalCmd, finalArgs, err = c.Root().Traverse(trimmedArgs)
	} else {
		finalCmd, finalArgs, err = c.Root().Find(trimmedArgs)
	}
	if err != nil {
		// Unable to find the real command. E.g., <program> someInvalidCmd <TAB>
		return c, []string{}, ShellCompDirectiveDefault, fmt.Errorf("Unable to find a command for arguments: %v", trimmedArgs)
	}

	// Check if we are doing flag value completion before parsing the flags.
	// This is important because if we are completing a flag value, we need to also
	// remove the flag name argument from the list of finalArgs or else the parsing
	// could fail due to an invalid value (incomplete) for the flag.
	flag, finalArgs, toComplete, err := checkIfFlagCompletion(finalCmd, finalArgs, toComplete)
	if err != nil {
		// Error while attempting to parse flags
		return finalCmd, []string{}, ShellCompDirectiveDefault, err
	}

	// Parse the flags early so we can check if required flags are set
	if err = finalCmd.ParseFlags(finalArgs); err != nil {
		return finalCmd, []string{}, ShellCompDirectiveDefault, fmt.Errorf("Error while parsing flags from args %v: %s", finalArgs, err.Error())
	}

	if flag != nil {
		// Check if we are completing a flag value subject to annotations
		if validExts, present := flag.Annotations[BashCompFilenameExt]; present {
			if len(validExts) != 0 {
				// File completion filtered by extensions
				return finalCmd, validExts, ShellCompDirectiveFilterFileExt, nil
			}

			// The annotation requests simple file completion.  There is no reason to do
			// that since it is the default behavior anyway.  Let's ignore this annotation
			// in case the program also registered a completion function for this flag.
			// Even though it is a mistake on the program's side, let's be nice when we can.
		}

		if subDir, present := flag.Annotations[BashCompSubdirsInDir]; present {
			if len(subDir) == 1 {
				// Directory completion from within a directory
				return finalCmd, subDir, ShellCompDirectiveFilterDirs, nil
			}
			// Directory completion
			return finalCmd, []string{}, ShellCompDirectiveFilterDirs, nil
		}
	}

	// When doing completion of a flag name, as soon as an argument starts with
	// a '-' we know it is a flag.  We cannot use isFlagArg() here as it requires
	// the flag name to be complete
	if flag == nil && len(toComplete) > 0 && toComplete[0] == '-' && !strings.Contains(toComplete, "=") {
		var completions []string

		// First check for required flags
		completions = completeRequireFlags(finalCmd, toComplete)

		// If we have not found any required flags, only then can we show regular flags
		if len(completions) == 0 {
			doCompleteFlags := func(flag *pflag.Flag) {
				if !flag.Changed ||
					strings.Contains(flag.Value.Type(), "Slice") ||
					strings.Contains(flag.Value.Type(), "Array") {
					// If the flag is not already present, or if it can be specified multiple times (Array or Slice)
					// we suggest it as a completion
					completions = append(completions, getFlagNameCompletions(flag, toComplete)...)
				}
			}

			// We cannot use finalCmd.Flags() because we may not have called ParsedFlags() for commands
			// that have set DisableFlagParsing; it is ParseFlags() that merges the inherited and
			// non-inherited flags.
			finalCmd.InheritedFlags().VisitAll(func(flag *pflag.Flag) {
				doCompleteFlags(flag)
			})
			finalCmd.NonInheritedFlags().VisitAll(func(flag *pflag.Flag) {
				doCompleteFlags(flag)
			})
		}

		directive := ShellCompDirectiveNoFileComp
		if len(completions) == 1 && strings.HasSuffix(completions[0], "=") {
			// If there is a single completion, the shell usually adds a space
			// after the completion.  We don't want that if the flag ends with an =
			directive = ShellCompDirectiveNoSpace
		}
		return finalCmd, completions, directive, nil
	}

	// We only remove the flags from the arguments if DisableFlagParsing is not set.
	// This is important for commands which have requested to do their own flag completion.
	if !finalCmd.DisableFlagParsing {
		finalArgs = finalCmd.Flags().Args()
	}

	var completions []string
	directive := ShellCompDirectiveDefault
	if flag == nil {
		foundLocalNonPersistentFlag := false
		// If TraverseChildren is true on the root command we don't check for
		// local flags because we can use a local flag on a parent command
		if !finalCmd.Root().TraverseChildren {
			// Check if there are any local, non-persistent flags on the command-line
			localNonPersistentFlags := finalCmd.LocalNonPersistentFlags()
			finalCmd.NonInheritedFlags().VisitAll(func(flag *pflag.Flag) {
				if localNonPersistentFlags.Lookup(flag.Name) != nil && flag.Changed {
					foundLocalNonPersistentFlag = true
				}
			})
		}

		// Complete subcommand names, including the help command
		if len(finalArgs) == 0 && !foundLocalNonPersistentFlag {
			// We only complete sub-commands if:
			// - there are no arguments on the command-line and
			// - there are no local, non-peristent flag on the command-line or TraverseChildren is true
			for _, subCmd := range finalCmd.Commands() {
				if subCmd.IsAvailableCommand() || subCmd == finalCmd.helpCommand {
					if strings.HasPrefix(subCmd.Name(), toComplete) {
						completions = append(completions, fmt.Sprintf("%s\t%s", subCmd.Name(), subCmd.Short))
					}
					directive = ShellCompDirectiveNoFileComp
				}
			}
		}

		// Complete required flags even without the '-' prefix
		completions = append(completions, completeRequireFlags(finalCmd, toComplete)...)

		// Always complete ValidArgs, even if we are completing a subcommand name.
		// This is for commands that have both subcommands and ValidArgs.
		if len(finalCmd.ValidArgs) > 0 {
			if len(finalArgs) == 0 {
				// ValidArgs are only for the first argument
				for _, validArg := range finalCmd.ValidArgs {
					if strings.HasPrefix(validArg, toComplete) {
						completions = append(completions, validArg)
					}
				}
				directive = ShellCompDirectiveNoFileComp

				// If no completions were found within commands or ValidArgs,
				// see if there are any ArgAliases that should be completed.
				if len(completions) == 0 {
					for _, argAlias := range finalCmd.ArgAliases {
						if strings.HasPrefix(argAlias, toComplete) {
							completions = append(completions, argAlias)
						}
					}
				}
			}

			// If there are ValidArgs specified (even if they don't match), we stop completion.
			// Only one of ValidArgs or ValidArgsFunction can be used for a single command.
			return finalCmd, completions, directive, nil
		}

		// Let the logic continue so as to add any ValidArgsFunction completions,
		// even if we already found sub-commands.
		// This is for commands that have subcommands but also specify a ValidArgsFunction.
	}

	// Find the completion function for the flag or command
	var completionFn func(cmd *Command, args []string, toComplete string) ([]string, ShellCompDirective)
	if flag != nil {
		completionFn = flagCompletionFunctions[flag]
	} else {
		completionFn = finalCmd.ValidArgsFunction
	}
	if completionFn != nil {
		// Go custom completion defined for this flag or command.
		// Call the registered completion function to get the completions.
		var comps []string
		comps, directive = completionFn(finalCmd, finalArgs, toComplete)
		completions = append(completions, comps...)
	}

	return finalCmd, completions, directive, nil
}

func getFlagNameCompletions(flag *pflag.Flag, toComplete string) []string {
	if nonCompletableFlag(flag) {
		return []string{}
	}

	var completions []string
	flagName := "--" + flag.Name
	if strings.HasPrefix(flagName, toComplete) {
		// Flag without the =
		completions = append(completions, fmt.Sprintf("%s\t%s", flagName, flag.Usage))

		// Why suggest both long forms: --flag and --flag= ?
		// This forces the user to *always* have to type either an = or a space after the flag name.
		// Let's be nice and avoid making users have to do that.
		// Since boolean flags and shortname flags don't show the = form, let's go that route and never show it.
		// The = form will still work, we just won't suggest it.
		// This also makes the list of suggested flags shorter as we avoid all the = forms.
		//
		// if len(flag.NoOptDefVal) == 0 {
		// 	// Flag requires a value, so it can be suffixed with =
		// 	flagName += "="
		// 	completions = append(completions, fmt.Sprintf("%s\t%s", flagName, flag.Usage))
		// }
	}

	flagName = "-" + flag.Shorthand
	if len(flag.Shorthand) > 0 && strings.HasPrefix(flagName, toComplete) {
		completions = append(completions, fmt.Sprintf("%s\t%s", flagName, flag.Usage))
	}

	return completions
}

func completeRequireFlags(finalCmd *Command, toComplete string) []string {
	var completions []string

	doCompleteRequiredFlags := func(flag *pflag.Flag) {
		if _, present := flag.Annotations[BashCompOneRequiredFlag]; present {
			if !flag.Changed {
				// If the flag is not already present, we suggest it as a completion
				completions = append(completions, getFlagNameCompletions(flag, toComplete)...)
			}
		}
	}

	// We cannot use finalCmd.Flags() because we may not have called ParsedFlags() for commands
	// that have set DisableFlagParsing; it is ParseFlags() that merges the inherited and
	// non-inherited flags.
	finalCmd.InheritedFlags().VisitAll(func(flag *pflag.Flag) {
		doCompleteRequiredFlags(flag)
	})
	finalCmd.NonInheritedFlags().VisitAll(func(flag *pflag.Flag) {
		doCompleteRequiredFlags(flag)
	})

	return completions
}

func checkIfFlagCompletion(finalCmd *Command, args []string, lastArg string) (*pflag.Flag, []string, string, error) {
	if finalCmd.DisableFlagParsing {
		// We only do flag completion if we are allowed to parse flags
		// This is important for commands which have requested to do their own flag completion.
		return nil, args, lastArg, nil
	}

	var flagName string
	trimmedArgs := args
	flagWithEqual := false

	// When doing completion of a flag name, as soon as an argument starts with
	// a '-' we know it is a flag.  We cannot use isFlagArg() here as that function
	// requires the flag name to be complete
	if len(lastArg) > 0 && lastArg[0] == '-' {
		if index := strings.Index(lastArg, "="); index >= 0 {
			// Flag with an =
			flagName = strings.TrimLeft(lastArg[:index], "-")
			lastArg = lastArg[index+1:]
			flagWithEqual = true
		} else {
			// Normal flag completion
			return nil, args, lastArg, nil
		}
	}

	if len(flagName) == 0 {
		if len(args) > 0 {
			prevArg := args[len(args)-1]
			if isFlagArg(prevArg) {
				// Only consider the case where the flag does not contain an =.
				// If the flag contains an = it means it has already been fully processed,
				// so we don't need to deal with it here.
				if index := strings.Index(prevArg, "="); index < 0 {
					flagName = strings.TrimLeft(prevArg, "-")

					// Remove the uncompleted flag or else there could be an error created
					// for an invalid value for that flag
					trimmedArgs = args[:len(args)-1]
				}
			}
		}
	}

	if len(flagName) == 0 {
		// Not doing flag completion
		return nil, trimmedArgs, lastArg, nil
	}

	flag := findFlag(finalCmd, flagName)
	if flag == nil {
		// Flag not supported by this command, nothing to complete
		err := fmt.Errorf("Subcommand '%s' does not support flag '%s'", finalCmd.Name(), flagName)
		return nil, nil, "", err
	}

	if !flagWithEqual {
		if len(flag.NoOptDefVal) != 0 {
			// We had assumed dealing with a two-word flag but the flag is a boolean flag.
			// In that case, there is no value following it, so we are not really doing flag completion.
			// Reset everything to do noun completion.
			trimmedArgs = args
			flag = nil
		}
	}

	return flag, trimmedArgs, lastArg, nil
}

func findFlag(cmd *Command, name string) *pflag.Flag {
	flagSet := cmd.Flags()
	if len(name) == 1 {
		// First convert the short flag into a long flag
		// as the cmd.Flag() search only accepts long flags
		if short := flagSet.ShorthandLookup(name); short != nil {
			name = short.Name
		} else {
			set := cmd.InheritedFlags()
			if short = set.ShorthandLookup(name); short != nil {
				name = short.Name
			} else {
				return nil
			}
		}
	}
	return cmd.Flag(name)
}

// CompDebug prints the specified string to the same file as where the
// completion script prints its logs.
// Note that completion printouts should never be on stdout as they would
// be wrongly interpreted as actual completion choices by the completion script.
func CompDebug(msg string, printToStdErr bool) {
	msg = fmt.Sprintf("[Debug] %s", msg)

	// Such logs are only printed when the user has set the environment
	// variable BASH_COMP_DEBUG_FILE to the path of some file to be used.
	if path := os.Getenv("BASH_COMP_DEBUG_FILE"); path != "" {
		f, err := os.OpenFile(path,
			os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err == nil {
			defer f.Close()
			WriteStringAndCheck(f, msg)
		}
	}

	if printToStdErr {
		// Must print to stderr for this not to be read by the completion script.
		fmt.Fprint(os.Stderr, msg)
	}
}

// CompDebugln prints the specified string with a newline at the end
// to the same file as where the completion script prints its logs.
// Such logs are only printed when the user has set the environment
// variable BASH_COMP_DEBUG_FILE to the path of some file to be used.
func CompDebugln(msg string, printToStdErr bool) {
	CompDebug(fmt.Sprintf("%s\n", msg), printToStdErr)
}

// CompError prints the specified completion message to stderr.
func CompError(msg string) {
	msg = fmt.Sprintf("[Error] %s", msg)
	CompDebug(msg, true)
}

// CompErrorln prints the specified completion message to stderr with a newline at the end.
func CompErrorln(msg string) {
	CompError(fmt.Sprintf("%s\n", msg))
}
