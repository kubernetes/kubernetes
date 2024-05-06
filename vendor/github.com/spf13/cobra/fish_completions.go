// Copyright 2013-2023 The Cobra Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
	nameForVar = strings.ReplaceAll(nameForVar, "-", "_")
	nameForVar = strings.ReplaceAll(nameForVar, ":", "_")

	compCmd := ShellCompRequestCmd
	if !includeDesc {
		compCmd = ShellCompNoDescRequestCmd
	}
	WriteStringAndCheck(buf, fmt.Sprintf("# fish completion for %-36s -*- shell-script -*-\n", name))
	WriteStringAndCheck(buf, fmt.Sprintf(`
function __%[1]s_debug
    set -l file "$BASH_COMP_DEBUG_FILE"
    if test -n "$file"
        echo "$argv" >> $file
    end
end

function __%[1]s_perform_completion
    __%[1]s_debug "Starting __%[1]s_perform_completion"

    # Extract all args except the last one
    set -l args (commandline -opc)
    # Extract the last arg and escape it in case it is a space
    set -l lastArg (string escape -- (commandline -ct))

    __%[1]s_debug "args: $args"
    __%[1]s_debug "last arg: $lastArg"

    # Disable ActiveHelp which is not supported for fish shell
    set -l requestComp "%[10]s=0 $args[1] %[3]s $args[2..-1] $lastArg"

    __%[1]s_debug "Calling $requestComp"
    set -l results (eval $requestComp 2> /dev/null)

    # Some programs may output extra empty lines after the directive.
    # Let's ignore them or else it will break completion.
    # Ref: https://github.com/spf13/cobra/issues/1279
    for line in $results[-1..1]
        if test (string trim -- $line) = ""
            # Found an empty line, remove it
            set results $results[1..-2]
        else
            # Found non-empty line, we have our proper output
            break
        end
    end

    set -l comps $results[1..-2]
    set -l directiveLine $results[-1]

    # For Fish, when completing a flag with an = (e.g., <program> -n=<TAB>)
    # completions must be prefixed with the flag
    set -l flagPrefix (string match -r -- '-.*=' "$lastArg")

    __%[1]s_debug "Comps: $comps"
    __%[1]s_debug "DirectiveLine: $directiveLine"
    __%[1]s_debug "flagPrefix: $flagPrefix"

    for comp in $comps
        printf "%%s%%s\n" "$flagPrefix" "$comp"
    end

    printf "%%s\n" "$directiveLine"
end

# this function limits calls to __%[1]s_perform_completion, by caching the result behind $__%[1]s_perform_completion_once_result
function __%[1]s_perform_completion_once
    __%[1]s_debug "Starting __%[1]s_perform_completion_once"

    if test -n "$__%[1]s_perform_completion_once_result"
        __%[1]s_debug "Seems like a valid result already exists, skipping __%[1]s_perform_completion"
        return 0
    end

    set --global __%[1]s_perform_completion_once_result (__%[1]s_perform_completion)
    if test -z "$__%[1]s_perform_completion_once_result"
        __%[1]s_debug "No completions, probably due to a failure"
        return 1
    end

    __%[1]s_debug "Performed completions and set __%[1]s_perform_completion_once_result"
    return 0
end

# this function is used to clear the $__%[1]s_perform_completion_once_result variable after completions are run
function __%[1]s_clear_perform_completion_once_result
    __%[1]s_debug ""
    __%[1]s_debug "========= clearing previously set __%[1]s_perform_completion_once_result variable =========="
    set --erase __%[1]s_perform_completion_once_result
    __%[1]s_debug "Succesfully erased the variable __%[1]s_perform_completion_once_result"
end

function __%[1]s_requires_order_preservation
    __%[1]s_debug ""
    __%[1]s_debug "========= checking if order preservation is required =========="

    __%[1]s_perform_completion_once
    if test -z "$__%[1]s_perform_completion_once_result"
        __%[1]s_debug "Error determining if order preservation is required"
        return 1
    end

    set -l directive (string sub --start 2 $__%[1]s_perform_completion_once_result[-1])
    __%[1]s_debug "Directive is: $directive"

    set -l shellCompDirectiveKeepOrder %[9]d
    set -l keeporder (math (math --scale 0 $directive / $shellCompDirectiveKeepOrder) %% 2)
    __%[1]s_debug "Keeporder is: $keeporder"

    if test $keeporder -ne 0
        __%[1]s_debug "This does require order preservation"
        return 0
    end

    __%[1]s_debug "This doesn't require order preservation"
    return 1
end


# This function does two things:
# - Obtain the completions and store them in the global __%[1]s_comp_results
# - Return false if file completion should be performed
function __%[1]s_prepare_completions
    __%[1]s_debug ""
    __%[1]s_debug "========= starting completion logic =========="

    # Start fresh
    set --erase __%[1]s_comp_results

    __%[1]s_perform_completion_once
    __%[1]s_debug "Completion results: $__%[1]s_perform_completion_once_result"

    if test -z "$__%[1]s_perform_completion_once_result"
        __%[1]s_debug "No completion, probably due to a failure"
        # Might as well do file completion, in case it helps
        return 1
    end

    set -l directive (string sub --start 2 $__%[1]s_perform_completion_once_result[-1])
    set --global __%[1]s_comp_results $__%[1]s_perform_completion_once_result[1..-2]

    __%[1]s_debug "Completions are: $__%[1]s_comp_results"
    __%[1]s_debug "Directive is: $directive"

    set -l shellCompDirectiveError %[4]d
    set -l shellCompDirectiveNoSpace %[5]d
    set -l shellCompDirectiveNoFileComp %[6]d
    set -l shellCompDirectiveFilterFileExt %[7]d
    set -l shellCompDirectiveFilterDirs %[8]d

    if test -z "$directive"
        set directive 0
    end

    set -l compErr (math (math --scale 0 $directive / $shellCompDirectiveError) %% 2)
    if test $compErr -eq 1
        __%[1]s_debug "Received error directive: aborting."
        # Might as well do file completion, in case it helps
        return 1
    end

    set -l filefilter (math (math --scale 0 $directive / $shellCompDirectiveFilterFileExt) %% 2)
    set -l dirfilter (math (math --scale 0 $directive / $shellCompDirectiveFilterDirs) %% 2)
    if test $filefilter -eq 1; or test $dirfilter -eq 1
        __%[1]s_debug "File extension filtering or directory filtering not supported"
        # Do full file completion instead
        return 1
    end

    set -l nospace (math (math --scale 0 $directive / $shellCompDirectiveNoSpace) %% 2)
    set -l nofiles (math (math --scale 0 $directive / $shellCompDirectiveNoFileComp) %% 2)

    __%[1]s_debug "nospace: $nospace, nofiles: $nofiles"

    # If we want to prevent a space, or if file completion is NOT disabled,
    # we need to count the number of valid completions.
    # To do so, we will filter on prefix as the completions we have received
    # may not already be filtered so as to allow fish to match on different
    # criteria than the prefix.
    if test $nospace -ne 0; or test $nofiles -eq 0
        set -l prefix (commandline -t | string escape --style=regex)
        __%[1]s_debug "prefix: $prefix"

        set -l completions (string match -r -- "^$prefix.*" $__%[1]s_comp_results)
        set --global __%[1]s_comp_results $completions
        __%[1]s_debug "Filtered completions are: $__%[1]s_comp_results"

        # Important not to quote the variable for count to work
        set -l numComps (count $__%[1]s_comp_results)
        __%[1]s_debug "numComps: $numComps"

        if test $numComps -eq 1; and test $nospace -ne 0
            # We must first split on \t to get rid of the descriptions to be
            # able to check what the actual completion will be.
            # We don't need descriptions anyway since there is only a single
            # real completion which the shell will expand immediately.
            set -l split (string split --max 1 \t $__%[1]s_comp_results[1])

            # Fish won't add a space if the completion ends with any
            # of the following characters: @=/:.,
            set -l lastChar (string sub -s -1 -- $split)
            if not string match -r -q "[@=/:.,]" -- "$lastChar"
                # In other cases, to support the "nospace" directive we trick the shell
                # by outputting an extra, longer completion.
                __%[1]s_debug "Adding second completion to perform nospace directive"
                set --global __%[1]s_comp_results $split[1] $split[1].
                __%[1]s_debug "Completions are now: $__%[1]s_comp_results"
            end
        end

        if test $numComps -eq 0; and test $nofiles -eq 0
            # To be consistent with bash and zsh, we only trigger file
            # completion when there are no other completions
            __%[1]s_debug "Requesting file completion"
            return 1
        end
    end

    return 0
end

# Since Fish completions are only loaded once the user triggers them, we trigger them ourselves
# so we can properly delete any completions provided by another script.
# Only do this if the program can be found, or else fish may print some errors; besides,
# the existing completions will only be loaded if the program can be found.
if type -q "%[2]s"
    # The space after the program name is essential to trigger completion for the program
    # and not completion of the program name itself.
    # Also, we use '> /dev/null 2>&1' since '&>' is not supported in older versions of fish.
    complete --do-complete "%[2]s " > /dev/null 2>&1
end

# Remove any pre-existing completions for the program since we will be handling all of them.
complete -c %[2]s -e

# this will get called after the two calls below and clear the $__%[1]s_perform_completion_once_result global
complete -c %[2]s -n '__%[1]s_clear_perform_completion_once_result'
# The call to __%[1]s_prepare_completions will setup __%[1]s_comp_results
# which provides the program's completion choices.
# If this doesn't require order preservation, we don't use the -k flag
complete -c %[2]s -n 'not __%[1]s_requires_order_preservation && __%[1]s_prepare_completions' -f -a '$__%[1]s_comp_results'
# otherwise we use the -k flag
complete -k -c %[2]s -n '__%[1]s_requires_order_preservation && __%[1]s_prepare_completions' -f -a '$__%[1]s_comp_results'
`, nameForVar, name, compCmd,
		ShellCompDirectiveError, ShellCompDirectiveNoSpace, ShellCompDirectiveNoFileComp,
		ShellCompDirectiveFilterFileExt, ShellCompDirectiveFilterDirs, ShellCompDirectiveKeepOrder, activeHelpEnvVar(name)))
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
