package cobra

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
)

func genFishComp(buf io.StringWriter, name string, includeDesc bool) {
	// Variables should not contain a '-' or ':' character
	nameForVar := name
	nameForVar = strings.Replace(nameForVar, "-", "_", -1)
	nameForVar = strings.Replace(nameForVar, ":", "_", -1)

	compCmd := ShellCompRequestCmd
	if !includeDesc {
		compCmd = ShellCompNoDescRequestCmd
	}
	WriteStringAndCheck(buf, fmt.Sprintf("# fish completion for %-36s -*- shell-script -*-\n", name))
	WriteStringAndCheck(buf, fmt.Sprintf(`
function __%[1]s_debug
    set file "$BASH_COMP_DEBUG_FILE"
    if test -n "$file"
        echo "$argv" >> $file
    end
end

function __%[1]s_perform_completion
    __%[1]s_debug "Starting __%[1]s_perform_completion with: $argv"

    set args (string split -- " " "$argv")
    set lastArg "$args[-1]"

    __%[1]s_debug "args: $args"
    __%[1]s_debug "last arg: $lastArg"

    set emptyArg ""
    if test -z "$lastArg"
        __%[1]s_debug "Setting emptyArg"
        set emptyArg \"\"
    end
    __%[1]s_debug "emptyArg: $emptyArg"

    if not type -q "$args[1]"
        # This can happen when "complete --do-complete %[2]s" is called when running this script.
        __%[1]s_debug "Cannot find $args[1]. No completions."
        return
    end

    set requestComp "$args[1] %[3]s $args[2..-1] $emptyArg"
    __%[1]s_debug "Calling $requestComp"

    set results (eval $requestComp 2> /dev/null)
    set comps $results[1..-2]
    set directiveLine $results[-1]

    # For Fish, when completing a flag with an = (e.g., <program> -n=<TAB>)
    # completions must be prefixed with the flag
    set flagPrefix (string match -r -- '-.*=' "$lastArg")

    __%[1]s_debug "Comps: $comps"
    __%[1]s_debug "DirectiveLine: $directiveLine"
    __%[1]s_debug "flagPrefix: $flagPrefix"

    for comp in $comps
        printf "%%s%%s\n" "$flagPrefix" "$comp"
    end

    printf "%%s\n" "$directiveLine"
end

# This function does three things:
# 1- Obtain the completions and store them in the global __%[1]s_comp_results
# 2- Set the __%[1]s_comp_do_file_comp flag if file completion should be performed
#    and unset it otherwise
# 3- Return true if the completion results are not empty
function __%[1]s_prepare_completions
    # Start fresh
    set --erase __%[1]s_comp_do_file_comp
    set --erase __%[1]s_comp_results

    # Check if the command-line is already provided.  This is useful for testing.
    if not set --query __%[1]s_comp_commandLine
        # Use the -c flag to allow for completion in the middle of the line
        set __%[1]s_comp_commandLine (commandline -c)
    end
    __%[1]s_debug "commandLine is: $__%[1]s_comp_commandLine"

    set results (__%[1]s_perform_completion "$__%[1]s_comp_commandLine")
    set --erase __%[1]s_comp_commandLine
    __%[1]s_debug "Completion results: $results"

    if test -z "$results"
        __%[1]s_debug "No completion, probably due to a failure"
        # Might as well do file completion, in case it helps
        set --global __%[1]s_comp_do_file_comp 1
        return 1
    end

    set directive (string sub --start 2 $results[-1])
    set --global __%[1]s_comp_results $results[1..-2]

    __%[1]s_debug "Completions are: $__%[1]s_comp_results"
    __%[1]s_debug "Directive is: $directive"

    set shellCompDirectiveError %[4]d
    set shellCompDirectiveNoSpace %[5]d
    set shellCompDirectiveNoFileComp %[6]d
    set shellCompDirectiveFilterFileExt %[7]d
    set shellCompDirectiveFilterDirs %[8]d

    if test -z "$directive"
        set directive 0
    end

    set compErr (math (math --scale 0 $directive / $shellCompDirectiveError) %% 2)
    if test $compErr -eq 1
        __%[1]s_debug "Received error directive: aborting."
        # Might as well do file completion, in case it helps
        set --global __%[1]s_comp_do_file_comp 1
        return 1
    end

    set filefilter (math (math --scale 0 $directive / $shellCompDirectiveFilterFileExt) %% 2)
    set dirfilter (math (math --scale 0 $directive / $shellCompDirectiveFilterDirs) %% 2)
    if test $filefilter -eq 1; or test $dirfilter -eq 1
        __%[1]s_debug "File extension filtering or directory filtering not supported"
        # Do full file completion instead
        set --global __%[1]s_comp_do_file_comp 1
        return 1
    end

    set nospace (math (math --scale 0 $directive / $shellCompDirectiveNoSpace) %% 2)
    set nofiles (math (math --scale 0 $directive / $shellCompDirectiveNoFileComp) %% 2)

    __%[1]s_debug "nospace: $nospace, nofiles: $nofiles"

    # Important not to quote the variable for count to work
    set numComps (count $__%[1]s_comp_results)
    __%[1]s_debug "numComps: $numComps"

    if test $numComps -eq 1; and test $nospace -ne 0
        # To support the "nospace" directive we trick the shell
        # by outputting an extra, longer completion.
        __%[1]s_debug "Adding second completion to perform nospace directive"
        set --append __%[1]s_comp_results $__%[1]s_comp_results[1].
    end

    if test $numComps -eq 0; and test $nofiles -eq 0
        __%[1]s_debug "Requesting file completion"
        set --global __%[1]s_comp_do_file_comp 1
    end

    # If we don't want file completion, we must return true even if there
    # are no completions found.  This is because fish will perform the last
    # completion command, even if its condition is false, if no other
    # completion command was triggered
    return (not set --query __%[1]s_comp_do_file_comp)
end

# Since Fish completions are only loaded once the user triggers them, we trigger them ourselves
# so we can properly delete any completions provided by another script.
# The space after the the program name is essential to trigger completion for the program
# and not completion of the program name itself.
complete --do-complete "%[2]s " > /dev/null 2>&1
# Using '> /dev/null 2>&1' since '&>' is not supported in older versions of fish.

# Remove any pre-existing completions for the program since we will be handling all of them.
complete -c %[2]s -e

# The order in which the below two lines are defined is very important so that __%[1]s_prepare_completions
# is called first.  It is __%[1]s_prepare_completions that sets up the __%[1]s_comp_do_file_comp variable.
#
# This completion will be run second as complete commands are added FILO.
# It triggers file completion choices when __%[1]s_comp_do_file_comp is set.
complete -c %[2]s -n 'set --query __%[1]s_comp_do_file_comp'

# This completion will be run first as complete commands are added FILO.
# The call to __%[1]s_prepare_completions will setup both __%[1]s_comp_results and __%[1]s_comp_do_file_comp.
# It provides the program's completion choices.
complete -c %[2]s -n '__%[1]s_prepare_completions' -f -a '$__%[1]s_comp_results'

`, nameForVar, name, compCmd,
		ShellCompDirectiveError, ShellCompDirectiveNoSpace, ShellCompDirectiveNoFileComp,
		ShellCompDirectiveFilterFileExt, ShellCompDirectiveFilterDirs))
}

// GenFishCompletion generates fish completion file and writes to the passed writer.
func (c *Command) GenFishCompletion(w io.Writer, includeDesc bool) error {
	buf := new(bytes.Buffer)
	genFishComp(buf, c.Name(), includeDesc)
	_, err := buf.WriteTo(w)
	return err
}

// GenFishCompletionFile generates fish completion file.
func (c *Command) GenFishCompletionFile(filename string, includeDesc bool) error {
	outFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer outFile.Close()

	return c.GenFishCompletion(outFile, includeDesc)
}
