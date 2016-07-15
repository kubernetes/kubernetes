package cobra

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/spf13/pflag"
)

const (
	BashCompFilenameExt     = "cobra_annotation_bash_completion_filename_extentions"
	BashCompCustom          = "cobra_annotation_bash_completion_custom"
	BashCompOneRequiredFlag = "cobra_annotation_bash_completion_one_required_flag"
	BashCompSubdirsInDir    = "cobra_annotation_bash_completion_subdirs_in_dir"
)

func preamble(out io.Writer, name string) error {
	_, err := fmt.Fprintf(out, "# bash completion for %-36s -*- shell-script -*-\n", name)
	if err != nil {
		return err
	}
	_, err = fmt.Fprint(out, `
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
    _get_comp_words_by_ref "$@" cur prev words cword
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
    __debug "${FUNCNAME[0]}"
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
                [[ "${COMPREPLY[0]}" == *= ]] || compopt +o nospace
            fi

            # complete after --flag=abc
            if [[ $cur == *=* ]]; then
                if [[ $(type -t compopt) = "builtin" ]]; then
                    compopt +o nospace
                fi

                local index flag
                flag="${cur%%=*}"
                __index_of_word "${flag}" "${flags_with_completion[@]}"
                if [[ ${index} -ge 0 ]]; then
                    COMPREPLY=()
                    PREFIX=""
                    cur="${cur#*=}"
                    ${flags_completion[${index}]}
                    if [ -n "${ZSH_VERSION}" ]; then
                        # zfs completion needs --flag= prefix
                        eval "COMPREPLY=( \"\${COMPREPLY[@]/#/${flag}=}\" )"
                    fi
                fi
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
    completions=("${commands[@]}")
    if [[ ${#must_have_one_noun[@]} -ne 0 ]]; then
        completions=("${must_have_one_noun[@]}")
    fi
    if [[ ${#must_have_one_flag[@]} -ne 0 ]]; then
        completions+=("${must_have_one_flag[@]}")
    fi
    COMPREPLY=( $(compgen -W "${completions[*]}" -- "$cur") )

    if [[ ${#COMPREPLY[@]} -eq 0 && ${#noun_aliases[@]} -gt 0 && ${#must_have_one_noun[@]} -ne 0 ]]; then
        COMPREPLY=( $(compgen -W "${noun_aliases[*]}" -- "$cur") )
    fi

    if [[ ${#COMPREPLY[@]} -eq 0 ]]; then
        declare -F __custom_func >/dev/null && __custom_func
    fi

    __ltrim_colon_completions "$cur"
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
    __debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"

    # if a command required a flag, and we found it, unset must_have_one_flag()
    local flagname=${words[c]}
    local flagvalue
    # if the word contained an =
    if [[ ${words[c]} == *"="* ]]; then
        flagvalue=${flagname#*=} # take in as flagvalue after the =
        flagname=${flagname%%=*} # strip everything after the =
        flagname="${flagname}=" # but put the = back
    fi
    __debug "${FUNCNAME[0]}: looking for ${flagname}"
    if __contains_word "${flagname}" "${must_have_one_flag[@]}"; then
        must_have_one_flag=()
    fi

    # if you set a flag which only applies to this command, don't show subcommands
    if __contains_word "${flagname}" "${local_nonpersistent_flags[@]}"; then
      commands=()
    fi

    # keep flag value with flagname as flaghash
    if [ -n "${flagvalue}" ] ; then
        flaghash[${flagname}]=${flagvalue}
    elif [ -n "${words[ $((c+1)) ]}" ] ; then
        flaghash[${flagname}]=${words[ $((c+1)) ]}
    else
        flaghash[${flagname}]="true" # pad "true" for bool flag
    fi

    # skip the argument to a two word flag
    if __contains_word "${words[c]}" "${two_word_flags[@]}"; then
        c=$((c+1))
        # if we are looking for a flags value, don't show commands
        if [[ $c -eq $cword ]]; then
            commands=()
        fi
    fi

    c=$((c+1))

}

__handle_noun()
{
    __debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"

    if __contains_word "${words[c]}" "${must_have_one_noun[@]}"; then
        must_have_one_noun=()
    elif __contains_word "${words[c]}" "${noun_aliases[@]}"; then
        must_have_one_noun=()
    fi

    nouns+=("${words[c]}")
    c=$((c+1))
}

__handle_command()
{
    __debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"

    local next_command
    if [[ -n ${last_command} ]]; then
        next_command="_${last_command}_${words[c]//:/__}"
    else
        if [[ $c -eq 0 ]]; then
            next_command="_$(basename "${words[c]//:/__}")"
        else
            next_command="_${words[c]//:/__}"
        fi
    fi
    c=$((c+1))
    __debug "${FUNCNAME[0]}: looking for ${next_command}"
    declare -F $next_command >/dev/null && $next_command
}

__handle_word()
{
    if [[ $c -ge $cword ]]; then
        __handle_reply
        return
    fi
    __debug "${FUNCNAME[0]}: c is $c words[c] is ${words[c]}"
    if [[ "${words[c]}" == -* ]]; then
        __handle_flag
    elif __contains_word "${words[c]}" "${commands[@]}"; then
        __handle_command
    elif [[ $c -eq 0 ]] && __contains_word "$(basename "${words[c]}")" "${commands[@]}"; then
        __handle_command
    else
        __handle_noun
    fi
    __handle_word
}

`)
	return err
}

func postscript(w io.Writer, name string) error {
	name = strings.Replace(name, ":", "__", -1)
	_, err := fmt.Fprintf(w, "__start_%s()\n", name)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, `{
    local cur prev words cword
    declare -A flaghash 2>/dev/null || :
    if declare -F _init_completion >/dev/null 2>&1; then
        _init_completion -s || return
    else
        __my_init_completion -n "=" || return
    fi

    local c=0
    local flags=()
    local two_word_flags=()
    local local_nonpersistent_flags=()
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
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, `if [[ $(type -t compopt) = "builtin" ]]; then
    complete -o default -F __start_%s %s
else
    complete -o default -o nospace -F __start_%s %s
fi

`, name, name, name, name)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "# ex: ts=4 sw=4 et filetype=sh\n")
	return err
}

func writeCommands(cmd *Command, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "    commands=()\n"); err != nil {
		return err
	}
	for _, c := range cmd.Commands() {
		if !c.IsAvailableCommand() || c == cmd.helpCommand {
			continue
		}
		if _, err := fmt.Fprintf(w, "    commands+=(%q)\n", c.Name()); err != nil {
			return err
		}
	}
	_, err := fmt.Fprintf(w, "\n")
	return err
}

func writeFlagHandler(name string, annotations map[string][]string, w io.Writer) error {
	for key, value := range annotations {
		switch key {
		case BashCompFilenameExt:
			_, err := fmt.Fprintf(w, "    flags_with_completion+=(%q)\n", name)
			if err != nil {
				return err
			}

			if len(value) > 0 {
				ext := "__handle_filename_extension_flag " + strings.Join(value, "|")
				_, err = fmt.Fprintf(w, "    flags_completion+=(%q)\n", ext)
			} else {
				ext := "_filedir"
				_, err = fmt.Fprintf(w, "    flags_completion+=(%q)\n", ext)
			}
			if err != nil {
				return err
			}
		case BashCompCustom:
			_, err := fmt.Fprintf(w, "    flags_with_completion+=(%q)\n", name)
			if err != nil {
				return err
			}
			if len(value) > 0 {
				handlers := strings.Join(value, "; ")
				_, err = fmt.Fprintf(w, "    flags_completion+=(%q)\n", handlers)
			} else {
				_, err = fmt.Fprintf(w, "    flags_completion+=(:)\n")
			}
			if err != nil {
				return err
			}
		case BashCompSubdirsInDir:
			_, err := fmt.Fprintf(w, "    flags_with_completion+=(%q)\n", name)

			if len(value) == 1 {
				ext := "__handle_subdirs_in_dir_flag " + value[0]
				_, err = fmt.Fprintf(w, "    flags_completion+=(%q)\n", ext)
			} else {
				ext := "_filedir -d"
				_, err = fmt.Fprintf(w, "    flags_completion+=(%q)\n", ext)
			}
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func writeShortFlag(flag *pflag.Flag, w io.Writer) error {
	b := (len(flag.NoOptDefVal) > 0)
	name := flag.Shorthand
	format := "    "
	if !b {
		format += "two_word_"
	}
	format += "flags+=(\"-%s\")\n"
	if _, err := fmt.Fprintf(w, format, name); err != nil {
		return err
	}
	return writeFlagHandler("-"+name, flag.Annotations, w)
}

func writeFlag(flag *pflag.Flag, w io.Writer) error {
	b := (len(flag.NoOptDefVal) > 0)
	name := flag.Name
	format := "    flags+=(\"--%s"
	if !b {
		format += "="
	}
	format += "\")\n"
	if _, err := fmt.Fprintf(w, format, name); err != nil {
		return err
	}
	return writeFlagHandler("--"+name, flag.Annotations, w)
}

func writeLocalNonPersistentFlag(flag *pflag.Flag, w io.Writer) error {
	b := (len(flag.NoOptDefVal) > 0)
	name := flag.Name
	format := "    local_nonpersistent_flags+=(\"--%s"
	if !b {
		format += "="
	}
	format += "\")\n"
	if _, err := fmt.Fprintf(w, format, name); err != nil {
		return err
	}
	return nil
}

func writeFlags(cmd *Command, w io.Writer) error {
	_, err := fmt.Fprintf(w, `    flags=()
    two_word_flags=()
    local_nonpersistent_flags=()
    flags_with_completion=()
    flags_completion=()

`)
	if err != nil {
		return err
	}
	localNonPersistentFlags := cmd.LocalNonPersistentFlags()
	var visitErr error
	cmd.NonInheritedFlags().VisitAll(func(flag *pflag.Flag) {
		if nonCompletableFlag(flag) {
			return
		}
		if err := writeFlag(flag, w); err != nil {
			visitErr = err
			return
		}
		if len(flag.Shorthand) > 0 {
			if err := writeShortFlag(flag, w); err != nil {
				visitErr = err
				return
			}
		}
		if localNonPersistentFlags.Lookup(flag.Name) != nil {
			if err := writeLocalNonPersistentFlag(flag, w); err != nil {
				visitErr = err
				return
			}
		}
	})
	if visitErr != nil {
		return visitErr
	}
	cmd.InheritedFlags().VisitAll(func(flag *pflag.Flag) {
		if nonCompletableFlag(flag) {
			return
		}
		if err := writeFlag(flag, w); err != nil {
			visitErr = err
			return
		}
		if len(flag.Shorthand) > 0 {
			if err := writeShortFlag(flag, w); err != nil {
				visitErr = err
				return
			}
		}
	})
	if visitErr != nil {
		return visitErr
	}

	_, err = fmt.Fprintf(w, "\n")
	return err
}

func writeRequiredFlag(cmd *Command, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "    must_have_one_flag=()\n"); err != nil {
		return err
	}
	flags := cmd.NonInheritedFlags()
	var visitErr error
	flags.VisitAll(func(flag *pflag.Flag) {
		if nonCompletableFlag(flag) {
			return
		}
		for key := range flag.Annotations {
			switch key {
			case BashCompOneRequiredFlag:
				format := "    must_have_one_flag+=(\"--%s"
				b := (flag.Value.Type() == "bool")
				if !b {
					format += "="
				}
				format += "\")\n"
				if _, err := fmt.Fprintf(w, format, flag.Name); err != nil {
					visitErr = err
					return
				}

				if len(flag.Shorthand) > 0 {
					if _, err := fmt.Fprintf(w, "    must_have_one_flag+=(\"-%s\")\n", flag.Shorthand); err != nil {
						visitErr = err
						return
					}
				}
			}
		}
	})
	return visitErr
}

func writeRequiredNouns(cmd *Command, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "    must_have_one_noun=()\n"); err != nil {
		return err
	}
	sort.Sort(sort.StringSlice(cmd.ValidArgs))
	for _, value := range cmd.ValidArgs {
		if _, err := fmt.Fprintf(w, "    must_have_one_noun+=(%q)\n", value); err != nil {
			return err
		}
	}
	return nil
}

func writeArgAliases(cmd *Command, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "    noun_aliases=()\n"); err != nil {
		return err
	}
	sort.Sort(sort.StringSlice(cmd.ArgAliases))
	for _, value := range cmd.ArgAliases {
		if _, err := fmt.Fprintf(w, "    noun_aliases+=(%q)\n", value); err != nil {
			return err
		}
	}
	return nil
}

func gen(cmd *Command, w io.Writer) error {
	for _, c := range cmd.Commands() {
		if !c.IsAvailableCommand() || c == cmd.helpCommand {
			continue
		}
		if err := gen(c, w); err != nil {
			return err
		}
	}
	commandName := cmd.CommandPath()
	commandName = strings.Replace(commandName, " ", "_", -1)
	commandName = strings.Replace(commandName, ":", "__", -1)
	if _, err := fmt.Fprintf(w, "_%s()\n{\n", commandName); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "    last_command=%q\n", commandName); err != nil {
		return err
	}
	if err := writeCommands(cmd, w); err != nil {
		return err
	}
	if err := writeFlags(cmd, w); err != nil {
		return err
	}
	if err := writeRequiredFlag(cmd, w); err != nil {
		return err
	}
	if err := writeRequiredNouns(cmd, w); err != nil {
		return err
	}
	if err := writeArgAliases(cmd, w); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "}\n\n"); err != nil {
		return err
	}
	return nil
}

func (cmd *Command) GenBashCompletion(w io.Writer) error {
	if err := preamble(w, cmd.Name()); err != nil {
		return err
	}
	if len(cmd.BashCompletionFunction) > 0 {
		if _, err := fmt.Fprintf(w, "%s\n", cmd.BashCompletionFunction); err != nil {
			return err
		}
	}
	if err := gen(cmd, w); err != nil {
		return err
	}
	return postscript(w, cmd.Name())
}

func nonCompletableFlag(flag *pflag.Flag) bool {
	return flag.Hidden || len(flag.Deprecated) > 0
}

func (cmd *Command) GenBashCompletionFile(filename string) error {
	outFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer outFile.Close()

	return cmd.GenBashCompletion(outFile)
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

// MarkFlagCustom adds the BashCompCustom annotation to the named flag, if it exists.
// Generated bash autocompletion will call the bash function f for the flag.
func (cmd *Command) MarkFlagCustom(name string, f string) error {
	return MarkFlagCustom(cmd.Flags(), name, f)
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

// MarkFlagCustom adds the BashCompCustom annotation to the named flag in the flag set, if it exists.
// Generated bash autocompletion will call the bash function f for the flag.
func MarkFlagCustom(flags *pflag.FlagSet, name string, f string) error {
	return flags.SetAnnotation(name, BashCompCustom, []string{f})
}
