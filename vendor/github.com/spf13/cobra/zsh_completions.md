## Generating Zsh Completion for your cobra.Command

Cobra supports native Zsh completion generated from the root `cobra.Command`.
The generated completion script should be put somewhere in your `$fpath` named
`_<YOUR COMMAND>`.

### What's Supported

* Completion for all non-hidden subcommands using their `.Short` description.
* Completion for all non-hidden flags using the following rules:
  * Filename completion works by marking the flag with `cmd.MarkFlagFilename...`
    family of commands.
  * The requirement for argument to the flag is decided by the `.NoOptDefVal`
    flag value - if it's empty then completion will expect an argument.
  * Flags of one of the various `*Array` and `*Slice` types supports multiple
    specifications (with or without argument depending on the specific type).
* Completion of positional arguments using the following rules:
  * Argument position for all options below starts at `1`. If argument position
    `0` is requested it will raise an error.
  * Use `command.MarkZshCompPositionalArgumentFile` to complete filenames. Glob
    patterns (e.g. `"*.log"`) are optional - if not specified it will offer to
    complete all file types.
  * Use `command.MarkZshCompPositionalArgumentWords` to offer specific words for
    completion. At least one word is required.
  * It's possible to specify completion for some arguments and leave some
    unspecified (e.g. offer words for second argument but nothing for first
    argument). This will cause no completion for first argument but words
    completion for second argument.
  * If no argument completion was specified for 1st argument (but optionally was
    specified for 2nd) and the command has `ValidArgs` it will be used as
    completion options for 1st argument.
  * Argument completions only offered for commands with no subcommands.

### What's not yet Supported

* Custom completion scripts are not supported yet (We should probably create zsh
  specific one, doesn't make sense to re-use the bash one as the functions will
  be different).
* Whatever other feature you're looking for and doesn't exist :)
