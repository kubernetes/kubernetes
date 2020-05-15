# Generating Bash Completions For Your Own cobra.Command

If you are using the generator you can create a completion command by running

```bash
cobra add completion
```

Update the help text show how to install the bash_completion Linux show here [Kubectl docs show mac options](https://kubernetes.io/docs/tasks/tools/install-kubectl/#enabling-shell-autocompletion)

Writing the shell script to stdout allows the most flexible use.

```go
// completionCmd represents the completion command
var completionCmd = &cobra.Command{
	Use:   "completion",
	Short: "Generates bash completion scripts",
	Long: `To load completion run

. <(bitbucket completion)

To configure your bash shell to load completions for each session add to your bashrc

# ~/.bashrc or ~/.profile
. <(bitbucket completion)
`,
	Run: func(cmd *cobra.Command, args []string) {
		rootCmd.GenBashCompletion(os.Stdout);
	},
}
```

**Note:** The cobra generator may include messages printed to stdout for example if the config file is loaded, this will break the auto complete script


## Example from kubectl

Generating bash completions from a cobra command is incredibly easy. An actual program which does so for the kubernetes kubectl binary is as follows:

```go
package main

import (
	"io/ioutil"
	"os"

	"k8s.io/kubernetes/pkg/kubectl/cmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func main() {
	kubectl := cmd.NewKubectlCommand(util.NewFactory(nil), os.Stdin, ioutil.Discard, ioutil.Discard)
	kubectl.GenBashCompletionFile("out.sh")
}
```

`out.sh` will get you completions of subcommands and flags. Copy it to `/etc/bash_completion.d/` as described [here](https://debian-administration.org/article/316/An_introduction_to_bash_completion_part_1) and reset your terminal to use autocompletion. If you make additional annotations to your code, you can get even more intelligent and flexible behavior.

## Have the completions code complete your 'nouns'

### Static completion of nouns

This method allows you to provide a pre-defined list of completion choices for your nouns using the `validArgs` field.
For example, if you want `kubectl get [tab][tab]` to show a list of valid "nouns" you have to set them. Simplified code from `kubectl get` looks like:

```go
validArgs []string = { "pod", "node", "service", "replicationcontroller" }

cmd := &cobra.Command{
	Use:     "get [(-o|--output=)json|yaml|template|...] (RESOURCE [NAME] | RESOURCE/NAME ...)",
	Short:   "Display one or many resources",
	Long:    get_long,
	Example: get_example,
	Run: func(cmd *cobra.Command, args []string) {
		err := RunGet(f, out, cmd, args)
		util.CheckErr(err)
	},
	ValidArgs: validArgs,
}
```

Notice we put the "ValidArgs" on the "get" subcommand. Doing so will give results like

```bash
# kubectl get [tab][tab]
node                 pod                    replicationcontroller  service
```

### Plural form and shortcuts for nouns

If your nouns have a number of aliases, you can define them alongside `ValidArgs` using `ArgAliases`:

```go
argAliases []string = { "pods", "nodes", "services", "svc", "replicationcontrollers", "rc" }

cmd := &cobra.Command{
    ...
	ValidArgs:  validArgs,
	ArgAliases: argAliases
}
```

The aliases are not shown to the user on tab completion, but they are accepted as valid nouns by
the completion algorithm if entered manually, e.g. in:

```bash
# kubectl get rc [tab][tab]
backend        frontend       database 
```

Note that without declaring `rc` as an alias, the completion algorithm would show the list of nouns
in this example again instead of the replication controllers.

### Dynamic completion of nouns

In some cases it is not possible to provide a list of possible completions in advance.  Instead, the list of completions must be determined at execution-time.  Cobra provides two ways of defining such dynamic completion of nouns. Note that both these methods can be used along-side each other as long as they are not both used for the same command.

**Note**: *Custom Completions written in Go* will automatically work for other shell-completion scripts (e.g., Fish shell), while *Custom Completions written in Bash* will only work for Bash shell-completion.  It is therefore recommended to use *Custom Completions written in Go*.

#### 1. Custom completions of nouns written in Go

In a similar fashion as for static completions, you can use the `ValidArgsFunction` field to provide a Go function that Cobra will execute when it needs the list of completion choices for the nouns of a command.  Note that either `ValidArgs` or `ValidArgsFunction` can be used for a single cobra command, but not both.
Simplified code from `helm status` looks like:

```go
cmd := &cobra.Command{
	Use:   "status RELEASE_NAME",
	Short: "Display the status of the named release",
	Long:  status_long,
	RunE: func(cmd *cobra.Command, args []string) {
		RunGet(args[0])
	},
	ValidArgsFunction: func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		if len(args) != 0 {
			return nil, cobra.ShellCompDirectiveNoFileComp
		}
		return getReleasesFromCluster(toComplete), cobra.ShellCompDirectiveNoFileComp
	},
}
```
Where `getReleasesFromCluster()` is a Go function that obtains the list of current Helm releases running on the Kubernetes cluster.
Notice we put the `ValidArgsFunction` on the `status` subcommand. Let's assume the Helm releases on the cluster are: `harbor`, `notary`, `rook` and `thanos` then this dynamic completion will give results like

```bash
# helm status [tab][tab]
harbor notary rook thanos
```
You may have noticed the use of `cobra.ShellCompDirective`.  These directives are bit fields allowing to control some shell completion behaviors for your particular completion.  You can combine them with the bit-or operator such as `cobra.ShellCompDirectiveNoSpace | cobra.ShellCompDirectiveNoFileComp`
```go
// Indicates an error occurred and completions should be ignored.
ShellCompDirectiveError
// Indicates that the shell should not add a space after the completion,
// even if there is a single completion provided.
ShellCompDirectiveNoSpace
// Indicates that the shell should not provide file completion even when
// no completion is provided.
// This currently does not work for zsh or bash < 4
ShellCompDirectiveNoFileComp
// Indicates that the shell will perform its default behavior after completions
// have been provided (this implies !ShellCompDirectiveNoSpace && !ShellCompDirectiveNoFileComp).
ShellCompDirectiveDefault
```

When using the `ValidArgsFunction`, Cobra will call your registered function after having parsed all flags and arguments provided in the command-line.  You therefore don't need to do this parsing yourself.  For example, when a user calls `helm status --namespace my-rook-ns [tab][tab]`, Cobra will call your registered `ValidArgsFunction` after having parsed the `--namespace` flag, as it would have done when calling the `RunE` function.

##### Debugging

Cobra achieves dynamic completions written in Go through the use of a hidden command called by the completion script.  To debug your Go completion code, you can call this hidden command directly:
```bash
# helm __complete status har<ENTER>
harbor
:4
Completion ended with directive: ShellCompDirectiveNoFileComp # This is on stderr
```
***Important:*** If the noun to complete is empty, you must pass an empty parameter to the `__complete` command:
```bash
# helm __complete status ""<ENTER>
harbor
notary
rook
thanos
:4
Completion ended with directive: ShellCompDirectiveNoFileComp # This is on stderr
```
Calling the `__complete` command directly allows you to run the Go debugger to troubleshoot your code.  You can also add printouts to your code; Cobra provides the following functions to use for printouts in Go completion code:
```go
// Prints to the completion script debug file (if BASH_COMP_DEBUG_FILE
// is set to a file path) and optionally prints to stderr.
cobra.CompDebug(msg string, printToStdErr bool) {
cobra.CompDebugln(msg string, printToStdErr bool)

// Prints to the completion script debug file (if BASH_COMP_DEBUG_FILE
// is set to a file path) and to stderr.
cobra.CompError(msg string)
cobra.CompErrorln(msg string)
```
***Important:*** You should **not** leave traces that print to stdout in your completion code as they will be interpreted as completion choices by the completion script.  Instead, use the cobra-provided debugging traces functions mentioned above.

#### 2. Custom completions of nouns written in Bash

This method allows you to inject bash functions into the completion script.  Those bash functions are responsible for providing the completion choices for your own completions.

Some more actual code that works in kubernetes:

```bash
const (
        bash_completion_func = `__kubectl_parse_get()
{
    local kubectl_output out
    if kubectl_output=$(kubectl get --no-headers "$1" 2>/dev/null); then
        out=($(echo "${kubectl_output}" | awk '{print $1}'))
        COMPREPLY=( $( compgen -W "${out[*]}" -- "$cur" ) )
    fi
}

__kubectl_get_resource()
{
    if [[ ${#nouns[@]} -eq 0 ]]; then
        return 1
    fi
    __kubectl_parse_get ${nouns[${#nouns[@]} -1]}
    if [[ $? -eq 0 ]]; then
        return 0
    fi
}

__kubectl_custom_func() {
    case ${last_command} in
        kubectl_get | kubectl_describe | kubectl_delete | kubectl_stop)
            __kubectl_get_resource
            return
            ;;
        *)
            ;;
    esac
}
`)
```

And then I set that in my command definition:

```go
cmds := &cobra.Command{
	Use:   "kubectl",
	Short: "kubectl controls the Kubernetes cluster manager",
	Long: `kubectl controls the Kubernetes cluster manager.

Find more information at https://github.com/GoogleCloudPlatform/kubernetes.`,
	Run: runHelp,
	BashCompletionFunction: bash_completion_func,
}
```

The `BashCompletionFunction` option is really only valid/useful on the root command. Doing the above will cause `__kubectl_custom_func()` (`__<command-use>_custom_func()`) to be called when the built in processor was unable to find a solution. In the case of kubernetes a valid command might look something like `kubectl get pod [mypod]`. If you type `kubectl get pod [tab][tab]` the `__kubectl_customc_func()` will run because the cobra.Command only understood "kubectl" and "get." `__kubectl_custom_func()` will see that the cobra.Command is "kubectl_get" and will thus call another helper `__kubectl_get_resource()`.  `__kubectl_get_resource` will look at the 'nouns' collected. In our example the only noun will be `pod`.  So it will call `__kubectl_parse_get pod`.  `__kubectl_parse_get` will actually call out to kubernetes and get any pods.  It will then set `COMPREPLY` to valid pods!

## Mark flags as required

Most of the time completions will only show subcommands. But if a flag is required to make a subcommand work, you probably want it to show up when the user types [tab][tab].  Marking a flag as 'Required' is incredibly easy.

```go
cmd.MarkFlagRequired("pod")
cmd.MarkFlagRequired("container")
```

and you'll get something like

```bash
# kubectl exec [tab][tab][tab]
-c            --container=  -p            --pod=  
```

# Specify valid filename extensions for flags that take a filename

In this example we use --filename= and expect to get a json or yaml file as the argument. To make this easier we annotate the --filename flag with valid filename extensions.

```go
	annotations := []string{"json", "yaml", "yml"}
	annotation := make(map[string][]string)
	annotation[cobra.BashCompFilenameExt] = annotations

	flag := &pflag.Flag{
		Name:        "filename",
		Shorthand:   "f",
		Usage:       usage,
		Value:       value,
		DefValue:    value.String(),
		Annotations: annotation,
	}
	cmd.Flags().AddFlag(flag)
```

Now when you run a command with this filename flag you'll get something like

```bash
# kubectl create -f 
test/                         example/                      rpmbuild/
hello.yml                     test.json
```

So while there are many other files in the CWD it only shows me subdirs and those with valid extensions.

# Specify custom flag completion

As for nouns, Cobra provides two ways of defining dynamic completion of flags.  Note that both these methods can be used along-side each other as long as they are not both used for the same flag.

**Note**: *Custom Completions written in Go* will automatically work for other shell-completion scripts (e.g., Fish shell), while *Custom Completions written in Bash* will only work for Bash shell-completion.  It is therefore recommended to use *Custom Completions written in Go*.

## 1. Custom completions of flags written in Go

To provide a Go function that Cobra will execute when it needs the list of completion choices for a flag, you must register the function in the following manner:

```go
flagName := "output"
cmd.RegisterFlagCompletionFunc(flagName, func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
	return []string{"json", "table", "yaml"}, cobra.ShellCompDirectiveDefault
})
```
Notice that calling `RegisterFlagCompletionFunc()` is done through the `command` with which the flag is associated.  In our example this dynamic completion will give results like so:

```bash
# helm status --output [tab][tab]
json table yaml
```

### Debugging

You can also easily debug your Go completion code for flags:
```bash
# helm __complete status --output ""
json
table
yaml
:4
Completion ended with directive: ShellCompDirectiveNoFileComp # This is on stderr
```
***Important:*** You should **not** leave traces that print to stdout in your completion code as they will be interpreted as completion choices by the completion script.  Instead, use the cobra-provided debugging traces functions mentioned in the above section.

## 2. Custom completions of flags written in Bash

Alternatively, you can use bash code for flag custom completion. Similar to the filename
completion and filtering using `cobra.BashCompFilenameExt`, you can specify
a custom flag completion bash function with `cobra.BashCompCustom`:

```go
	annotation := make(map[string][]string)
	annotation[cobra.BashCompCustom] = []string{"__kubectl_get_namespaces"}

	flag := &pflag.Flag{
		Name:        "namespace",
		Usage:       usage,
		Annotations: annotation,
	}
	cmd.Flags().AddFlag(flag)
```

In addition add the `__kubectl_get_namespaces` implementation in the `BashCompletionFunction`
value, e.g.:

```bash
__kubectl_get_namespaces()
{
    local template
    template="{{ range .items  }}{{ .metadata.name }} {{ end }}"
    local kubectl_out
    if kubectl_out=$(kubectl get -o template --template="${template}" namespace 2>/dev/null); then
        COMPREPLY=( $( compgen -W "${kubectl_out}[*]" -- "$cur" ) )
    fi
}
```
# Using bash aliases for commands

You can also configure the `bash aliases` for the commands and they will also support completions.

```bash
alias aliasname=origcommand
complete -o default -F __start_origcommand aliasname

# and now when you run `aliasname` completion will make
# suggestions as it did for `origcommand`.

$) aliasname <tab><tab>
completion     firstcommand   secondcommand
```
