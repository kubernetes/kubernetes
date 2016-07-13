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
	"io"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	cmdconfig "k8s.io/kubernetes/pkg/kubectl/cmd/config"
	"k8s.io/kubernetes/pkg/kubectl/cmd/rollout"
	"k8s.io/kubernetes/pkg/kubectl/cmd/set"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/flag"

	"github.com/spf13/cobra"
)

const (
	bash_completion_func = `# call kubectl get $1,
__kubectl_override_flag_list=(kubeconfig cluster user context namespace server)
__kubectl_override_flags()
{
    local ${__kubectl_override_flag_list[*]} two_word_of of
    for w in "${words[@]}"; do
        if [ -n "${two_word_of}" ]; then
            eval "${two_word_of}=\"--${two_word_of}=\${w}\""
            two_word_of=
            continue
        fi
        for of in "${__kubectl_override_flag_list[@]}"; do
            case "${w}" in
                --${of}=*)
                    eval "${of}=\"--${of}=\${w}\""
                    ;;
                --${of})
                    two_word_of="${of}"
                    ;;
            esac
        done
        if [ "${w}" == "--all-namespaces" ]; then
            namespace="--all-namespaces"
        fi
    done
    for of in "${__kubectl_override_flag_list[@]}"; do
        if eval "test -n \"\$${of}\""; then
            eval "echo \${${of}}"
        fi
    done
}

__kubectl_get_namespaces()
{
    local template kubectl_out
    template="{{ range .items  }}{{ .metadata.name }} {{ end }}"
    if kubectl_out=$(kubectl get -o template --template="${template}" namespace 2>/dev/null); then
        COMPREPLY=( $( compgen -W "${kubectl_out[*]}" -- "$cur" ) )
    fi
}

__kubectl_parse_get()
{
    local template
    template="{{ range .items  }}{{ .metadata.name }} {{ end }}"
    local kubectl_out
    if kubectl_out=$(kubectl get $(__kubectl_override_flags) -o template --template="${template}" "$1" 2>/dev/null); then
        COMPREPLY=( $( compgen -W "${kubectl_out[*]}" -- "$cur" ) )
    fi
}

__kubectl_get_resource()
{
    if [[ ${#nouns[@]} -eq 0 ]]; then
        return 1
    fi
    __kubectl_parse_get "${nouns[${#nouns[@]} -1]}"
}

__kubectl_get_resource_pod()
{
    __kubectl_parse_get "pod"
}

__kubectl_get_resource_rc()
{
    __kubectl_parse_get "rc"
}

# $1 is the name of the pod we want to get the list of containers inside
__kubectl_get_containers()
{
    local template
    template="{{ range .spec.containers  }}{{ .name }} {{ end }}"
    __debug "${FUNCNAME} nouns are ${nouns[*]}"

    local len="${#nouns[@]}"
    if [[ ${len} -ne 1 ]]; then
        return
    fi
    local last=${nouns[${len} -1]}
    local kubectl_out
    if kubectl_out=$(kubectl get $(__kubectl_namespace_flag) -o template --template="${template}" pods "${last}" 2>/dev/null); then
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

__custom_func() {
    case ${last_command} in
        kubectl_get | kubectl_describe | kubectl_delete | kubectl_label | kubectl_stop | kubectl_edit | kubectl_patch |\
        kubectl_annotate | kubectl_expose)
            __kubectl_get_resource
            return
            ;;
        kubectl_logs)
            __kubectl_require_pod_and_container
            return
            ;;
        kubectl_exec)
            __kubectl_get_resource_pod
            return
            ;;
        kubectl_rolling-update)
            __kubectl_get_resource_rc
            return
            ;;
        *)
            ;;
    esac
}
`

	// If you add a resource to this list, please also take a look at pkg/kubectl/kubectl.go
	// and add a short forms entry in expandResourceShortcut() when appropriate.
	valid_resources = `Valid resource types include:
   * componentstatuses (aka 'cs')
   * configmaps (aka 'cm')
   * daemonsets (aka 'ds')
   * deployments (aka 'deploy')
   * events (aka 'ev')
   * endpoints (aka 'ep')
   * horizontalpodautoscalers (aka 'hpa')
   * ingress (aka 'ing')
   * jobs
   * limitranges (aka 'limits')
   * nodes (aka 'no')
   * namespaces (aka 'ns')
   * petsets (alpha feature, may be unstable)
   * pods (aka 'po')
   * persistentvolumes (aka 'pv')
   * persistentvolumeclaims (aka 'pvc')
   * quota
   * resourcequotas (aka 'quota')
   * replicasets (aka 'rs')
   * replicationcontrollers (aka 'rc')
   * secrets
   * serviceaccounts (aka 'sa')
   * services (aka 'svc')
`
	usage_template = `{{if gt .Aliases 0}}

Aliases:
  {{.NameAndAliases}}{{end}}{{if .HasExample}}

Examples:
{{ .Example }}{{end}}{{ if .HasAvailableSubCommands}}

Available Sub-commands:{{range .Commands}}{{if .IsAvailableCommand}}
  {{rpad .Name .NamePadding }} {{.Short}}{{end}}{{end}}{{end}}{{ if .HasLocalFlags}}

Flags:
{{.LocalFlags.FlagUsages | trimRightSpace}}{{end}}{{ if .HasInheritedFlags}}

Global Flags:
{{.InheritedFlags.FlagUsages | trimRightSpace}}{{end}}{{if .HasHelpSubCommands}}

Additional help topics:{{range .Commands}}{{if .IsHelpCommand}}
  {{rpad .CommandPath .CommandPathPadding}} {{.Short}}{{end}}{{end}}{{end}}

Usage:{{if .Runnable}}
  {{if .HasFlags}}{{appendIfNotPresent .UseLine "[flags]"}}{{else}}{{.UseLine}}{{end}}{{end}}{{ if .HasSubCommands }}
  {{ .CommandPath}} [command]

Use "{{.CommandPath}} [command] --help" for more information about a command.{{end}}
`
	help_template = `{{with or .Long .Short }}{{. | trim}}{{end}}{{if or .Runnable .HasSubCommands}}{{.UsageString}}{{end}}`
)

// NewKubectlCommand creates the `kubectl` command and its nested children.
func NewKubectlCommand(f *cmdutil.Factory, in io.Reader, out, err io.Writer) *cobra.Command {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "kubectl",
		Short: "kubectl controls the Kubernetes cluster manager",
		Long: `kubectl controls the Kubernetes cluster manager.

Find more information at https://github.com/kubernetes/kubernetes.`,
		Run: runHelp,
		BashCompletionFunction: bash_completion_func,
	}
	cmds.SetHelpTemplate(help_template)
	cmds.SetUsageTemplate(usage_template)

	f.BindFlags(cmds.PersistentFlags())
	f.BindExternalFlags(cmds.PersistentFlags())

	cmds.SetHelpCommand(NewCmdHelp(f, out))

	// From this point and forward we get warnings on flags that contain "_" separators
	cmds.SetGlobalNormalizationFunc(flag.WarnWordSepNormalizeFunc)

	cmds.AddCommand(NewCmdGet(f, out, err))
	cmds.AddCommand(set.NewCmdSet(f, out))
	cmds.AddCommand(NewCmdDescribe(f, out))
	cmds.AddCommand(NewCmdCreate(f, out))
	cmds.AddCommand(NewCmdReplace(f, out))
	cmds.AddCommand(NewCmdPatch(f, out))
	cmds.AddCommand(NewCmdDelete(f, out))
	cmds.AddCommand(NewCmdEdit(f, out, err))
	cmds.AddCommand(NewCmdApply(f, out))

	cmds.AddCommand(NewCmdNamespace(out))
	cmds.AddCommand(NewCmdLogs(f, out))
	cmds.AddCommand(NewCmdRollingUpdate(f, out))
	cmds.AddCommand(NewCmdScale(f, out))
	cmds.AddCommand(NewCmdCordon(f, out))
	cmds.AddCommand(NewCmdDrain(f, out))
	cmds.AddCommand(NewCmdUncordon(f, out))

	cmds.AddCommand(NewCmdAttach(f, in, out, err))
	cmds.AddCommand(NewCmdExec(f, in, out, err))
	cmds.AddCommand(NewCmdPortForward(f, out, err))
	cmds.AddCommand(NewCmdProxy(f, out))

	cmds.AddCommand(NewCmdRun(f, in, out, err))
	cmds.AddCommand(NewCmdStop(f, out))
	cmds.AddCommand(NewCmdExposeService(f, out))
	cmds.AddCommand(NewCmdAutoscale(f, out))
	cmds.AddCommand(rollout.NewCmdRollout(f, out))

	cmds.AddCommand(NewCmdLabel(f, out))
	cmds.AddCommand(NewCmdAnnotate(f, out))
	cmds.AddCommand(NewCmdTaint(f, out))

	cmds.AddCommand(cmdconfig.NewCmdConfig(clientcmd.NewDefaultPathOptions(), out))
	cmds.AddCommand(NewCmdClusterInfo(f, out))
	cmds.AddCommand(NewCmdApiVersions(f, out))
	cmds.AddCommand(NewCmdVersion(f, out))
	cmds.AddCommand(NewCmdExplain(f, out))
	cmds.AddCommand(NewCmdConvert(f, out))
	cmds.AddCommand(NewCmdCompletion(f, out))

	cmds.AddCommand(NewCmdTop(f, out))

	if cmds.Flag("namespace") != nil {
		if cmds.Flag("namespace").Annotations == nil {
			cmds.Flag("namespace").Annotations = map[string][]string{}
		}
		cmds.Flag("namespace").Annotations[cobra.BashCompCustom] = append(
			cmds.Flag("namespace").Annotations[cobra.BashCompCustom],
			"__kubectl_get_namespaces",
		)
	}

	return cmds
}

func runHelp(cmd *cobra.Command, args []string) {
	cmd.Help()
}

func printDeprecationWarning(command, alias string) {
	glog.Warningf("%s is DEPRECATED and will be removed in a future version. Use %s instead.", alias, command)
}
