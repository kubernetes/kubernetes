package cobra

import (
	"bytes"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/spf13/pflag"
)

const (
	BashCompFilenameExt     = "cobra_annotation_bash_completion_filename_extentions"
	BashCompOneRequiredFlag = "cobra_annotation_bash_completion_one_required_flag"
	BashCompSubdirsInDir    = "cobra_annotation_bash_completion_subdirs_in_dir"
)

func preamble(out *bytes.Buffer) {
	fmt.Fprintf(out, `#!/bin/bash

__debug()
{
    if [[ -n ${BASH_COMP_DEBUG_FILE} ]]; then
        echo "$*" >> "${BASH_COMP_DEBUG_FILE}"
    fi
}

# Homebrew on Macs have version 1.3 of bash-completion which doesn't include
# _init_completion. This is a very minimal version of that function.
__my_init_completion()
{
    COMPREPLY=()
    _get_comp_words_by_ref cur prev words cword
}

__index_of_word()
{
    local w word=$1
    shift
    index=0
    for w in "$@"; do
        [[ $w = "$word" ]] && return
        index=$((index+1))
    done
    index=-1
}

__contains_word()
{
    local w word=$1; shift
    for w in "$@"; do
        [[ $w = "$word" ]] && return
    done
    return 1
}

__handle_reply()
{
    __debug "${FUNCNAME}"
    case $cur in
        -*)
            if [[ $(type -t compopt) = "builtin" ]]; then
                compopt -o nospace
            fi
            local allflags
            if [ ${#must_have_one_flag[@]} -ne 0 ]; then
                allflags=("${must_have_one_flag[@]}")
            else
                allflags=("${flags[*]} ${two_word_flags[*]}")
            fi
            COMPREPLY=( $(compgen -W "${allflags[*]}" -- "$cur") )
            if [[ $(type -t compopt) = "builtin" ]]; then
                [[ $COMPREPLY == *= ]] || compopt +o nospace
            fi
            return 0;
            ;;
    esac

    # check if we are handling a flag with special work handling
    local index
    __index_of_word "${prev}" "${flags_with_completion[@]}"
    if [[ ${index} -ge 0 ]]; then
        ${flags_completion[${index}]}
        return
    fi

    # we are parsing a flag and don't have a special handler, no completion
    if [[ ${cur} != "${words[cword]}" ]]; then
        return
    fi

    local completions
    if [[ ${#must_have_one_flag[@]} -ne 0 ]]; then
        completions=("${must_have_one_flag[@]}")
    elif [[ ${#must_have_one_noun[@]} -ne 0 ]]; then
        completions=("${must_have_one_noun[@]}")
    else
        completions=("${commands[@]}")
    fi
    COMPREPLY=( $(compgen -W "${completions[*]}" -- "$cur") )

    if [[ ${#COMPREPLY[@]} -eq 0 ]]; then
        declare -F __custom_func >/dev/null && __custom_func
    fi
}

# The arguments should be in the form "ext1|ext2|extn"
__handle_filename_extension_flag()
{
    local ext="$1"
    _filedir "@(${ext})"
}

__handle_subdirs_in_dir_flag()
{
    local dir="$1"
    pushd "${dir}" >/dev/null 2>&1 && _filedir -d && popd >/dev/null 2>&1
}

__handle_flag()
{
    __debug "${FUNCNAME}: c is $c words[c] is ${words[c]}"

    # if a command required a flag, and we found it, unset must_have_one_flag()
    local flagname=${words[c]}
    # if the word contained an =
    if [[ ${words[c]} == *"="* ]]; then
        flagname=${flagname%%=*} # strip everything after the =
        flagname="${flagname}=" # but put the = back
    fi
    __debug "${FUNCNAME}: looking for ${flagname}"
    if __contains_word "${flagname}" "${must_have_one_flag[@]}"; then
        must_have_one_flag=()
    fi

    # skip the argument to a two word flag
    if __contains_word "${words[c]}" "${two_word_flags[@]}"; then
        c=$((c+1))
        # if we are looking for a flags value, don't show commands
        if [[ $c -eq $cword ]]; then
            commands=()
        fi
    fi

    # skip the flag itself
    c=$((c+1))

}

__handle_noun()
{
    __debug "${FUNCNAME}: c is $c words[c] is ${words[c]}"

    if __contains_word "${words[c]}" "${must_have_one_noun[@]}"; then
        must_have_one_noun=()
    fi

    nouns+=("${words[c]}")
    c=$((c+1))
}

__handle_command()
{
    __debug "${FUNCNAME}: c is $c words[c] is ${words[c]}"

    local next_command
    if [[ -n ${last_command} ]]; then
        next_command="_${last_command}_${words[c]}"
    else
        next_command="_${words[c]}"
    fi
    c=$((c+1))
    __debug "${FUNCNAME}: looking for ${next_command}"
    declare -F $next_command >/dev/null && $next_command
}

__handle_word()
{
    if [[ $c -ge $cword ]]; then
        __handle_reply
        return
    fi
    __debug "${FUNCNAME}: c is $c words[c] is ${words[c]}"
    if [[ "${words[c]}" == -* ]]; then
        __handle_flag
    elif __contains_word "${words[c]}" "${commands[@]}"; then
        __handle_command
    else
        __handle_noun
    fi
    __handle_word
}

`)
}

func postscript(out *bytes.Buffer, name string) {
	fmt.Fprintf(out, "__start_%s()\n", name)
	fmt.Fprintf(out, `{
    local cur prev words cword
    if declare -F _init_completion >/dev/null 2>&1; then
        _init_completion -s || return
    else
        __my_init_completion || return
    fi

    local c=0
    local flags=()
    local two_word_flags=()
    local flags_with_completion=()
    local flags_completion=()
    local commands=("%s")
    local must_have_one_flag=()
    local must_have_one_noun=()
    local last_command
    local nouns=()

    __handle_word
}

`, name)
	fmt.Fprintf(out, `if [[ $(type -t compopt) = "builtin" ]]; then
    complete -F __start_%s %s
else
    complete -o nospace -F __start_%s %s
fi

`, name, name, name, name)
	fmt.Fprintf(out, "# ex: ts=4 sw=4 et filetype=sh\n")
}

func writeCommands(cmd *Command, out *bytes.Buffer) {
	fmt.Fprintf(out, "    commands=()\n")
	for _, c := range cmd.Commands() {
		if !c.IsAvailableCommand() || c == cmd.helpCommand {
			continue
		}
		fmt.Fprintf(out, "    commands+=(%q)\n", c.Name())
	}
	fmt.Fprintf(out, "\n")
}

func writeFlagHandler(name string, annotations map[string][]string, out *bytes.Buffer) {
	for key, value := range annotations {
		switch key {
		case BashCompFilenameExt:
			fmt.Fprintf(out, "    flags_with_completion+=(%q)\n", name)

			if len(value) > 0 {
				ext := "__handle_filename_extension_flag " + strings.Join(value, "|")
				fmt.Fprintf(out, "    flags_completion+=(%q)\n", ext)
			} else {
				ext := "_filedir"
				fmt.Fprintf(out, "    flags_completion+=(%q)\n", ext)
			}
		case BashCompSubdirsInDir:
			fmt.Fprintf(out, "    flags_with_completion+=(%q)\n", name)

			if len(value) == 1 {
				ext := "__handle_subdirs_in_dir_flag " + value[0]
				fmt.Fprintf(out, "    flags_completion+=(%q)\n", ext)
			} else {
				ext := "_filedir -d"
				fmt.Fprintf(out, "    flags_completion+=(%q)\n", ext)
			}
		}
	}
}

func writeShortFlag(flag *pflag.Flag, out *bytes.Buffer) {
	b := (flag.Value.Type() == "bool")
	name := flag.Shorthand
	format := "    "
	if !b {
		format += "two_word_"
	}
	format += "flags+=(\"-%s\")\n"
	fmt.Fprintf(out, format, name)
	writeFlagHandler("-"+name, flag.Annotations, out)
}

func writeFlag(flag *pflag.Flag, out *bytes.Buffer) {
	b := (flag.Value.Type() == "bool")
	name := flag.Name
	format := "    flags+=(\"--%s"
	if !b {
		format += "="
	}
	format += "\")\n"
	fmt.Fprintf(out, format, name)
	writeFlagHandler("--"+name, flag.Annotations, out)
}

func writeFlags(cmd *Command, out *bytes.Buffer) {
	fmt.Fprintf(out, `    flags=()
    two_word_flags=()
    flags_with_completion=()
    flags_completion=()

`)
	cmd.NonInheritedFlags().VisitAll(func(flag *pflag.Flag) {
		writeFlag(flag, out)
		if len(flag.Shorthand) > 0 {
			writeShortFlag(flag, out)
		}
	})
	cmd.InheritedFlags().VisitAll(func(flag *pflag.Flag) {
		writeFlag(flag, out)
		if len(flag.Shorthand) > 0 {
			writeShortFlag(flag, out)
		}
	})

	fmt.Fprintf(out, "\n")
}

func writeRequiredFlag(cmd *Command, out *bytes.Buffer) {
	fmt.Fprintf(out, "    must_have_one_flag=()\n")
	flags := cmd.NonInheritedFlags()
	flags.VisitAll(func(flag *pflag.Flag) {
		for key := range flag.Annotations {
			switch key {
			case BashCompOneRequiredFlag:
				format := "    must_have_one_flag+=(\"--%s"
				b := (flag.Value.Type() == "bool")
				if !b {
					format += "="
				}
				format += "\")\n"
				fmt.Fprintf(out, format, flag.Name)

				if len(flag.Shorthand) > 0 {
					fmt.Fprintf(out, "    must_have_one_flag+=(\"-%s\")\n", flag.Shorthand)
				}
			}
		}
	})
}

func writeRequiredNoun(cmd *Command, out *bytes.Buffer) {
	fmt.Fprintf(out, "    must_have_one_noun=()\n")
	sort.Sort(sort.StringSlice(cmd.ValidArgs))
	for _, value := range cmd.ValidArgs {
		fmt.Fprintf(out, "    must_have_one_noun+=(%q)\n", value)
	}
}

func gen(cmd *Command, out *bytes.Buffer) {
	for _, c := range cmd.Commands() {
		if !c.IsAvailableCommand() || c == cmd.helpCommand {
			continue
		}
		gen(c, out)
	}
	commandName := cmd.CommandPath()
	commandName = strings.Replace(commandName, " ", "_", -1)
	fmt.Fprintf(out, "_%s()\n{\n", commandName)
	fmt.Fprintf(out, "    last_command=%q\n", commandName)
	writeCommands(cmd, out)
	writeFlags(cmd, out)
	writeRequiredFlag(cmd, out)
	writeRequiredNoun(cmd, out)
	fmt.Fprintf(out, "}\n\n")
}

func (cmd *Command) GenBashCompletion(out *bytes.Buffer) {
	preamble(out)
	if len(cmd.BashCompletionFunction) > 0 {
		fmt.Fprintf(out, "%s\n", cmd.BashCompletionFunction)
	}
	gen(cmd, out)
	postscript(out, cmd.Name())
}

func (cmd *Command) GenBashCompletionFile(filename string) error {
	out := new(bytes.Buffer)

	cmd.GenBashCompletion(out)

	outFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer outFile.Close()

	_, err = outFile.Write(out.Bytes())
	if err != nil {
		return err
	}
	return nil
}

// MarkFlagRequired adds the BashCompOneRequiredFlag annotation to the named flag, if it exists.
func (cmd *Command) MarkFlagRequired(name string) error {
	return MarkFlagRequired(cmd.Flags(), name)
}

// MarkPersistentFlagRequired adds the BashCompOneRequiredFlag annotation to the named persistent flag, if it exists.
func (cmd *Command) MarkPersistentFlagRequired(name string) error {
	return MarkFlagRequired(cmd.PersistentFlags(), name)
}

// MarkFlagRequired adds the BashCompOneRequiredFlag annotation to the named flag in the flag set, if it exists.
func MarkFlagRequired(flags *pflag.FlagSet, name string) error {
	return flags.SetAnnotation(name, BashCompOneRequiredFlag, []string{"true"})
}

// MarkFlagFilename adds the BashCompFilenameExt annotation to the named flag, if it exists.
// Generated bash autocompletion will select filenames for the flag, limiting to named extensions if provided.
func (cmd *Command) MarkFlagFilename(name string, extensions ...string) error {
	return MarkFlagFilename(cmd.Flags(), name, extensions...)
}

// MarkPersistentFlagFilename adds the BashCompFilenameExt annotation to the named persistent flag, if it exists.
// Generated bash autocompletion will select filenames for the flag, limiting to named extensions if provided.
func (cmd *Command) MarkPersistentFlagFilename(name string, extensions ...string) error {
	return MarkFlagFilename(cmd.PersistentFlags(), name, extensions...)
}

// MarkFlagFilename adds the BashCompFilenameExt annotation to the named flag in the flag set, if it exists.
// Generated bash autocompletion will select filenames for the flag, limiting to named extensions if provided.
func MarkFlagFilename(flags *pflag.FlagSet, name string, extensions ...string) error {
	return flags.SetAnnotation(name, BashCompFilenameExt, extensions)
}
