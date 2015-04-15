# Generating Bash Completions For Your Own cobra.Command

Generating bash completions from a cobra command is incredibly easy. An actual program which does so for the kubernetes kubectl binary is as follows:

```go
package main

import (
        "io/ioutil"
        "os"

        "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd"
)

func main() {
        kubectl := cmd.NewFactory(nil).NewKubectlCommand(os.Stdin, ioutil.Discard, ioutil.Discard)
        kubectl.GenBashCompletionFile("out.sh")
}
```

That will get you completions of subcommands and flags. If you make additional annotations to your code, you can get even more intelligent and flexible behavior.

## Creating your own custom functions

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

__custom_func() {
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

The `BashCompletionFunction` option is really only valid/useful on the root command. Doing the above will cause `__custom_func()` to be called when the built in processor was unable to find a solution. In the case of kubernetes a valid command might look something like `kubectl get pod [mypod]`. If you type `kubectl get pod [tab][tab]` the `__customc_func()` will run because the cobra.Command only understood "kubectl" and "get." `__custom_func()` will see that the cobra.Command is "kubectl_get" and will thus call another helper `__kubectl_get_resource()`.  `__kubectl_get_resource` will look at the 'nouns' collected. In our example the only noun will be `pod`.  So it will call `__kubectl_parse_get pod`.  `__kubectl_parse_get` will actually call out to kubernetes and get any pods.  It will then set `COMPREPLY` to valid pods!

## Have the completions code complete your 'nouns'

In the above example "pod" was assumed to already be typed. But if you want `kubectl get [tab][tab]` to show a list of valid "nouns" you have to set them. Simplified code from `kubectl get` looks like:

```go
validArgs []string = { "pods", "nodes", "services", "replicationControllers" }

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
nodes                 pods                    replicationControllers  services
```

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

# Specify valid filename extentions for flags that take a filename

In this example we use --filename= and expect to get a json or yaml file as the argument. To make this easier we annotate the --filename flag with valid filename extensions.

```go
	annotations := make([]string, 3)
	annotations[0] = "json"
	annotations[1] = "yaml"
	annotations[2] = "yml"

	annotation := make(map[string][]string)
	annotation[cobra.BashCompFilenameExt] = annotations

	flag := &pflag.Flag{"filename", "f", usage, value, value.String(), false, annotation}
	cmd.Flags().AddFlag(flag)
```

Now when you run a command with this filename flag you'll get something like

```bash
# kubectl create -f 
test/                         example/                      rpmbuild/
hello.yml                     test.json
```

So while there are many other files in the CWD it only shows me subdirs and those with valid extensions.
