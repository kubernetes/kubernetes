/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cmd

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"syscall"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	utilflag "k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/auth"
	cmdconfig "k8s.io/kubernetes/pkg/kubectl/cmd/config"
	"k8s.io/kubernetes/pkg/kubectl/cmd/create"
	"k8s.io/kubernetes/pkg/kubectl/cmd/get"
	"k8s.io/kubernetes/pkg/kubectl/cmd/rollout"
	"k8s.io/kubernetes/pkg/kubectl/cmd/set"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/cmd/wait"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"

	"k8s.io/cli-runtime/pkg/genericclioptions"
)

const (
	bashCompletionFunc = `# call kubectl get $1,
__kubectl_override_flag_list=(--kubeconfig --cluster --user --context --namespace --server -n -s)
__kubectl_override_flags()
{
    local ${__kubectl_override_flag_list[*]##*-} two_word_of of var
    for w in "${words[@]}"; do
        if [ -n "${two_word_of}" ]; then
            eval "${two_word_of##*-}=\"${two_word_of}=\${w}\""
            two_word_of=
            continue
        fi
        for of in "${__kubectl_override_flag_list[@]}"; do
            case "${w}" in
                ${of}=*)
                    eval "${of##*-}=\"${w}\""
                    ;;
                ${of})
                    two_word_of="${of}"
                    ;;
            esac
        done
    done
    for var in "${__kubectl_override_flag_list[@]##*-}"; do
        if eval "test -n \"\$${var}\""; then
            eval "echo \${${var}}"
        fi
    done
}

__kubectl_config_get_contexts()
{
    __kubectl_parse_config "contexts"
}

__kubectl_config_get_clusters()
{
    __kubectl_parse_config "clusters"
}

__kubectl_config_get_users()
{
    __kubectl_parse_config "users"
}

# $1 has to be "contexts", "clusters" or "users"
__kubectl_parse_config()
{
    local template kubectl_out
    template="{{ range .$1  }}{{ .name }} {{ end }}"
    if kubectl_out=$(kubectl config $(__kubectl_override_flags) -o template --template="${template}" view 2>/dev/null); then
        COMPREPLY=( $( compgen -W "${kubectl_out[*]}" -- "$cur" ) )
    fi
}

# $1 is the name of resource (required)
# $2 is template string for kubectl get (optional)
__kubectl_parse_get()
{
    local template
    template="${2:-"{{ range .items  }}{{ .metadata.name }} {{ end }}"}"
    local kubectl_out
    if kubectl_out=$(kubectl get $(__kubectl_override_flags) -o template --template="${template}" "$1" 2>/dev/null); then
        COMPREPLY+=( $( compgen -W "${kubectl_out[*]}" -- "$cur" ) )
    fi
}

__kubectl_get_resource()
{
    if [[ ${#nouns[@]} -eq 0 ]]; then
      local kubectl_out
      if kubectl_out=$(kubectl api-resources $(__kubectl_override_flags) -o name --cached --request-timeout=5s --verbs=get 2>/dev/null); then
          COMPREPLY=( $( compgen -W "${kubectl_out[*]}" -- "$cur" ) )
          return 0
      fi
      return 1
    fi
    __kubectl_parse_get "${nouns[${#nouns[@]} -1]}"
}

__kubectl_get_resource_namespace()
{
    __kubectl_parse_get "namespace"
}

__kubectl_get_resource_pod()
{
    __kubectl_parse_get "pod"
}

__kubectl_get_resource_rc()
{
    __kubectl_parse_get "rc"
}

__kubectl_get_resource_node()
{
    __kubectl_parse_get "node"
}

__kubectl_get_resource_clusterrole()
{
    __kubectl_parse_get "clusterrole"
}

# $1 is the name of the pod we want to get the list of containers inside
__kubectl_get_containers()
{
    local template
    template="{{ range .spec.initContainers }}{{ .name }} {{end}}{{ range .spec.containers  }}{{ .name }} {{ end }}"
    __kubectl_debug "${FUNCNAME} nouns are ${nouns[*]}"

    local len="${#nouns[@]}"
    if [[ ${len} -ne 1 ]]; then
        return
    fi
    local last=${nouns[${len} -1]}
    local kubectl_out
    if kubectl_out=$(kubectl get $(__kubectl_override_flags) -o template --template="${template}" pods "${last}" 2>/dev/null); then
        COMPREPLY=( $( compgen -W "${kubectl_out[*]}" -- "$cur" ) )
    fi
}

# Require both a pod and a container to be specified
__kubectl_require_pod_and_container()
{
    if [[ ${#nouns[@]} -eq 0 ]]; then
        __kubectl_parse_get pods
        return 0
    fi;
    __kubectl_get_containers
    return 0
}

__kubectl_cp()
{
    if [[ $(type -t compopt) = "builtin" ]]; then
        compopt -o nospace
    fi

    case "$cur" in
        /*|[.~]*) # looks like a path
            return
            ;;
        *:*) # TODO: complete remote files in the pod
            return
            ;;
        */*) # complete <namespace>/<pod>
            local template namespace kubectl_out
            template="{{ range .items }}{{ .metadata.namespace }}/{{ .metadata.name }}: {{ end }}"
            namespace="${cur%%/*}"
            if kubectl_out=( $(kubectl get $(__kubectl_override_flags) --namespace "${namespace}" -o template --template="${template}" pods 2>/dev/null) ); then
                COMPREPLY=( $(compgen -W "${kubectl_out[*]}" -- "${cur}") )
            fi
            return
            ;;
        *) # complete namespaces, pods, and filedirs
            __kubectl_parse_get "namespace" "{{ range .items  }}{{ .metadata.name }}/ {{ end }}"
            __kubectl_parse_get "pod" "{{ range .items  }}{{ .metadata.name }}: {{ end }}"
            _filedir
            ;;
    esac
}

__custom_func() {
    case ${last_command} in
        kubectl_get | kubectl_describe | kubectl_delete | kubectl_label | kubectl_edit | kubectl_patch |\
        kubectl_annotate | kubectl_expose | kubectl_scale | kubectl_autoscale | kubectl_taint | kubectl_rollout_* |\
        kubectl_apply_edit-last-applied | kubectl_apply_view-last-applied)
            __kubectl_get_resource
            return
            ;;
        kubectl_logs | kubectl_attach)
            __kubectl_require_pod_and_container
            return
            ;;
        kubectl_exec | kubectl_port-forward | kubectl_top_pod)
            __kubectl_get_resource_pod
            return
            ;;
        kubectl_rolling-update)
            __kubectl_get_resource_rc
            return
            ;;
        kubectl_cordon | kubectl_uncordon | kubectl_drain | kubectl_top_node)
            __kubectl_get_resource_node
            return
            ;;
        kubectl_config_use-context | kubectl_config_rename-context)
            __kubectl_config_get_contexts
            return
            ;;
        kubectl_config_delete-cluster)
            __kubectl_config_get_clusters
            return
            ;;
        kubectl_cp)
            __kubectl_cp
            return
            ;;
        *)
            ;;
    esac
}
`
)

var (
	bash_completion_flags = map[string]string{
		"namespace": "__kubectl_get_resource_namespace",
		"context":   "__kubectl_config_get_contexts",
		"cluster":   "__kubectl_config_get_clusters",
		"user":      "__kubectl_config_get_users",
	}
)

func NewDefaultKubectlCommand() *cobra.Command {
	return NewDefaultKubectlCommandWithArgs(&defaultPluginHandler{}, os.Args, os.Stdin, os.Stdout, os.Stderr)
}

func NewDefaultKubectlCommandWithArgs(pluginHandler PluginHandler, args []string, in io.Reader, out, errout io.Writer) *cobra.Command {
	cmd := NewKubectlCommand(in, out, errout)

	if pluginHandler == nil {
		return cmd
	}

	if len(args) > 1 {
		cmdPathPieces := args[1:]

		// only look for suitable extension executables if
		// the specified command does not already exist
		if _, _, err := cmd.Find(cmdPathPieces); err != nil {
			if err := handleEndpointExtensions(pluginHandler, cmdPathPieces); err != nil {
				fmt.Fprintf(errout, "%v\n", err)
				os.Exit(1)
			}
		}
	}

	return cmd
}

// PluginHandler is capable of parsing command line arguments
// and performing executable filename lookups to search
// for valid plugin files, and execute found plugins.
type PluginHandler interface {
	// Lookup receives a potential filename and returns
	// a full or relative path to an executable, if one
	// exists at the given filename, or an error.
	Lookup(filename string) (string, error)
	// Execute receives an executable's filepath, a slice
	// of arguments, and a slice of environment variables
	// to relay to the executable.
	Execute(executablePath string, cmdArgs, environment []string) error
}

type defaultPluginHandler struct{}

// Lookup implements PluginHandler
func (h *defaultPluginHandler) Lookup(filename string) (string, error) {
	// if on Windows, append the "exe" extension
	// to the filename that we are looking up.
	if runtime.GOOS == "windows" {
		filename = filename + ".exe"
	}

	return exec.LookPath(filename)
}

// Execute implements PluginHandler
func (h *defaultPluginHandler) Execute(executablePath string, cmdArgs, environment []string) error {
	return syscall.Exec(executablePath, cmdArgs, environment)
}

func handleEndpointExtensions(pluginHandler PluginHandler, cmdArgs []string) error {
	remainingArgs := []string{} // all "non-flag" arguments

	for idx := range cmdArgs {
		if strings.HasPrefix(cmdArgs[idx], "-") {
			break
		}
		remainingArgs = append(remainingArgs, strings.Replace(cmdArgs[idx], "-", "_", -1))
	}

	foundBinaryPath := ""

	// attempt to find binary, starting at longest possible name with given cmdArgs
	for len(remainingArgs) > 0 {
		path, err := pluginHandler.Lookup(fmt.Sprintf("kubectl-%s", strings.Join(remainingArgs, "-")))
		if err != nil || len(path) == 0 {
			remainingArgs = remainingArgs[:len(remainingArgs)-1]
			continue
		}

		foundBinaryPath = path
		break
	}

	if len(foundBinaryPath) == 0 {
		return nil
	}

	// invoke cmd binary relaying the current environment and args given
	// remainingArgs will always have at least one element.
	// execve will make remainingArgs[0] the "binary name".
	if err := pluginHandler.Execute(foundBinaryPath, append([]string{foundBinaryPath}, cmdArgs[len(remainingArgs):]...), os.Environ()); err != nil {
		return err
	}

	return nil
}

// NewKubectlCommand creates the `kubectl` command and its nested children.
func NewKubectlCommand(in io.Reader, out, err io.Writer) *cobra.Command {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "kubectl",
		Short: i18n.T("kubectl controls the Kubernetes cluster manager"),
		Long: templates.LongDesc(`
      kubectl controls the Kubernetes cluster manager.

      Find more information at:
            https://kubernetes.io/docs/reference/kubectl/overview/`),
		Run: runHelp,
		BashCompletionFunction: bashCompletionFunc,
	}

	flags := cmds.PersistentFlags()
	flags.SetNormalizeFunc(utilflag.WarnWordSepNormalizeFunc) // Warn for "_" flags

	// Normalize all flags that are coming from other packages or pre-configurations
	// a.k.a. change all "_" to "-". e.g. glog package
	flags.SetNormalizeFunc(utilflag.WordSepNormalizeFunc)

	kubeConfigFlags := genericclioptions.NewConfigFlags()
	kubeConfigFlags.AddFlags(flags)
	matchVersionKubeConfigFlags := cmdutil.NewMatchVersionFlags(kubeConfigFlags)
	matchVersionKubeConfigFlags.AddFlags(cmds.PersistentFlags())

	cmds.PersistentFlags().AddGoFlagSet(flag.CommandLine)

	f := cmdutil.NewFactory(matchVersionKubeConfigFlags)

	// Sending in 'nil' for the getLanguageFn() results in using
	// the LANG environment variable.
	//
	// TODO: Consider adding a flag or file preference for setting
	// the language, instead of just loading from the LANG env. variable.
	i18n.LoadTranslations("kubectl", nil)

	// From this point and forward we get warnings on flags that contain "_" separators
	cmds.SetGlobalNormalizationFunc(utilflag.WarnWordSepNormalizeFunc)

	ioStreams := genericclioptions.IOStreams{In: in, Out: out, ErrOut: err}

	groups := templates.CommandGroups{
		{
			Message: "Basic Commands (Beginner):",
			Commands: []*cobra.Command{
				create.NewCmdCreate(f, ioStreams),
				NewCmdExposeService(f, ioStreams),
				NewCmdRun(f, ioStreams),
				set.NewCmdSet(f, ioStreams),
				deprecatedAlias("run-container", NewCmdRun(f, ioStreams)),
			},
		},
		{
			Message: "Basic Commands (Intermediate):",
			Commands: []*cobra.Command{
				NewCmdExplain("kubectl", f, ioStreams),
				get.NewCmdGet("kubectl", f, ioStreams),
				NewCmdEdit(f, ioStreams),
				NewCmdDelete(f, ioStreams),
			},
		},
		{
			Message: "Deploy Commands:",
			Commands: []*cobra.Command{
				rollout.NewCmdRollout(f, ioStreams),
				NewCmdRollingUpdate(f, ioStreams),
				NewCmdScale(f, ioStreams),
				NewCmdAutoscale(f, ioStreams),
			},
		},
		{
			Message: "Cluster Management Commands:",
			Commands: []*cobra.Command{
				NewCmdCertificate(f, ioStreams),
				NewCmdClusterInfo(f, ioStreams),
				NewCmdTop(f, ioStreams),
				NewCmdCordon(f, ioStreams),
				NewCmdUncordon(f, ioStreams),
				NewCmdDrain(f, ioStreams),
				NewCmdTaint(f, ioStreams),
			},
		},
		{
			Message: "Troubleshooting and Debugging Commands:",
			Commands: []*cobra.Command{
				NewCmdDescribe("kubectl", f, ioStreams),
				NewCmdLogs(f, ioStreams),
				NewCmdAttach(f, ioStreams),
				NewCmdExec(f, ioStreams),
				NewCmdPortForward(f, ioStreams),
				NewCmdProxy(f, ioStreams),
				NewCmdCp(f, ioStreams),
				auth.NewCmdAuth(f, ioStreams),
			},
		},
		{
			Message: "Advanced Commands:",
			Commands: []*cobra.Command{
				NewCmdApply("kubectl", f, ioStreams),
				NewCmdPatch(f, ioStreams),
				NewCmdReplace(f, ioStreams),
				wait.NewCmdWait(f, ioStreams),
				NewCmdConvert(f, ioStreams),
			},
		},
		{
			Message: "Settings Commands:",
			Commands: []*cobra.Command{
				NewCmdLabel(f, ioStreams),
				NewCmdAnnotate("kubectl", f, ioStreams),
				NewCmdCompletion(ioStreams.Out, ""),
			},
		},
	}
	groups.Add(cmds)

	filters := []string{"options"}

	// Hide the "alpha" subcommand if there are no alpha commands in this build.
	alpha := NewCmdAlpha(f, ioStreams)
	if !alpha.HasSubCommands() {
		filters = append(filters, alpha.Name())
	}

	templates.ActsAsRootCommand(cmds, filters, groups...)

	for name, completion := range bash_completion_flags {
		if cmds.Flag(name) != nil {
			if cmds.Flag(name).Annotations == nil {
				cmds.Flag(name).Annotations = map[string][]string{}
			}
			cmds.Flag(name).Annotations[cobra.BashCompCustom] = append(
				cmds.Flag(name).Annotations[cobra.BashCompCustom],
				completion,
			)
		}
	}

	cmds.AddCommand(alpha)
	cmds.AddCommand(cmdconfig.NewCmdConfig(f, clientcmd.NewDefaultPathOptions(), ioStreams))
	cmds.AddCommand(NewCmdPlugin(f, ioStreams))
	cmds.AddCommand(NewCmdVersion(f, ioStreams))
	cmds.AddCommand(NewCmdApiVersions(f, ioStreams))
	cmds.AddCommand(NewCmdApiResources(f, ioStreams))
	cmds.AddCommand(NewCmdOptions(ioStreams.Out))

	return cmds
}

func runHelp(cmd *cobra.Command, args []string) {
	cmd.Help()
}

func printDeprecationWarning(errOut io.Writer, command, alias string) {
	fmt.Fprintf(errOut, "%s is DEPRECATED and will be removed in a future version. Use %s instead.\n", alias, command)
}

// deprecatedAlias is intended to be used to create a "wrapper" command around
// an existing command. The wrapper works the same but prints a deprecation
// message before running. This command is identical functionality.
func deprecatedAlias(deprecatedVersion string, cmd *cobra.Command) *cobra.Command {
	// Have to be careful here because Cobra automatically extracts the name
	// of the command from the .Use field.
	originalName := cmd.Name()

	cmd.Use = deprecatedVersion
	cmd.Deprecated = fmt.Sprintf("use %q instead", originalName)
	cmd.Short = fmt.Sprintf("%s. This command is deprecated, use %q instead", cmd.Short, originalName)
	cmd.Hidden = true
	return cmd
}

var metadataAccessor = meta.NewAccessor()
