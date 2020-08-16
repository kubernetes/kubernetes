## Generating Fish Completions for your own cobra.Command

Cobra supports native Fish completions generated from the root `cobra.Command`.  You can use the `command.GenFishCompletion()` or `command.GenFishCompletionFile()` functions. You must provide these functions with a parameter indicating if the completions should be annotated with a description; Cobra will provide the description automatically based on usage information.  You can choose to make this option configurable by your users.

### Limitations

* Custom completions implemented using the `ValidArgsFunction` and `RegisterFlagCompletionFunc()` are supported automatically but the ones implemented in Bash scripting are not.
