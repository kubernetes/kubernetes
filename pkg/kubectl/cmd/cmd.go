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
	"fmt"
	"io"

	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/auth"
	cmdconfig "k8s.io/kubernetes/pkg/kubectl/cmd/config"
	"k8s.io/kubernetes/pkg/kubectl/cmd/rollout"
	"k8s.io/kubernetes/pkg/kubectl/cmd/set"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

const (
	bashCompletionFunc = `# call kubectl get $1,
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
                    eval "${of}=\"${w}\""
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

__custom_func() {
    case ${last_command} in
        kubectl_get | kubectl_describe | kubectl_delete | kubectl_label | kubectl_edit | kubectl_patch |\
        kubectl_annotate | kubectl_expose | kubectl_scale | kubectl_autoscale | kubectl_taint | kubectl_rollout_*)
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
        kubectl_config_use-context)
            __kubectl_config_get_contexts
            return
            ;;
        kubectl_config_delete-cluster)
            __kubectl_config_get_clusters
            return
            ;;
        *)
            ;;
    esac
}
`

	// If you add a resource to this list, please also take a look at pkg/kubectl/kubectl.go
	// and add a short forms entry in expandResourceShortcut() when appropriate.
	// TODO: This should be populated using the discovery information from apiserver.
	validResources = `Valid resource types include:

    * all
    * certificatesigningrequests (aka 'csr')
    * clusterrolebindings
    * clusterroles
    * clusters (valid only for federation apiservers)
    * componentstatuses (aka 'cs')
    * configmaps (aka 'cm')
    * controllerrevisions
    * cronjobs
    * customresourcedefinition (aka 'crd')
    * daemonsets (aka 'ds')
    * deployments (aka 'deploy')
    * endpoints (aka 'ep')
    * events (aka 'ev')
    * horizontalpodautoscalers (aka 'hpa')
    * ingresses (aka 'ing')
    * jobs
    * limitranges (aka 'limits')
    * namespaces (aka 'ns')
    * networkpolicies (aka 'netpol')
    * nodes (aka 'no')
    * persistentvolumeclaims (aka 'pvc')
    * persistentvolumes (aka 'pv')
    * poddisruptionbudgets (aka 'pdb')
    * podpreset
    * pods (aka 'po')
    * podsecuritypolicies (aka 'psp')
    * podtemplates
    * replicasets (aka 'rs')
    * replicationcontrollers (aka 'rc')
    * resourcequotas (aka 'quota')
    * rolebindings
    * roles
    * secrets
    * serviceaccounts (aka 'sa')
    * services (aka 'svc')
    * statefulsets
    * storageclasses

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

// NewKubectlCommand creates the `kubectl` command and its nested children.
func NewKubectlCommand(f cmdutil.Factory, in io.Reader, out, err io.Writer) *cobra.Command {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "kubectl",
		Short: i18n.T("kubectl controls the Kubernetes cluster manager"),
		Long: templates.LongDesc(`
      kubectl controls the Kubernetes cluster manager.

      Find more information at https://github.com/kubernetes/kubernetes.`),
		Run: runHelp,
		BashCompletionFunction: bashCompletionFunc,
	}

	f.BindFlags(cmds.PersistentFlags())
	f.BindExternalFlags(cmds.PersistentFlags())

	// Sending in 'nil' for the getLanguageFn() results in using
	// the LANG environment variable.
	//
	// TODO: Consider adding a flag or file preference for setting
	// the language, instead of just loading from the LANG env. variable.
	i18n.LoadTranslations("kubectl", nil)

	// From this point and forward we get warnings on flags that contain "_" separators
	cmds.SetGlobalNormalizationFunc(flag.WarnWordSepNormalizeFunc)

	groups := templates.CommandGroups{
		{
			Message: "Basic Commands (Beginner):",
			Commands: []*cobra.Command{
				NewCmdCreate(f, out, err),
				NewCmdExposeService(f, out),
				NewCmdRun(f, in, out, err),
				set.NewCmdSet(f, in, out, err),
				deprecatedAlias("run-container", NewCmdRun(f, in, out, err)),
			},
		},
		{
			Message: "Basic Commands (Intermediate):",
			Commands: []*cobra.Command{
				NewCmdGet(f, out, err),
				NewCmdExplain(f, out, err),
				NewCmdEdit(f, out, err),
				NewCmdDelete(f, out, err),
			},
		},
		{
			Message: "Deploy Commands:",
			Commands: []*cobra.Command{
				rollout.NewCmdRollout(f, out, err),
				NewCmdRollingUpdate(f, out),
				NewCmdScale(f, out),
				NewCmdAutoscale(f, out),
			},
		},
		{
			Message: "Cluster Management Commands:",
			Commands: []*cobra.Command{
				NewCmdCertificate(f, out),
				NewCmdClusterInfo(f, out),
				NewCmdTop(f, out, err),
				NewCmdCordon(f, out),
				NewCmdUncordon(f, out),
				NewCmdDrain(f, out, err),
				NewCmdTaint(f, out),
			},
		},
		{
			Message: "Troubleshooting and Debugging Commands:",
			Commands: []*cobra.Command{
				NewCmdDescribe(f, out, err),
				NewCmdLogs(f, out),
				NewCmdAttach(f, in, out, err),
				NewCmdExec(f, in, out, err),
				NewCmdPortForward(f, out, err),
				NewCmdProxy(f, out),
				NewCmdCp(f, out, err),
				auth.NewCmdAuth(f, out, err),
			},
		},
		{
			Message: "Advanced Commands:",
			Commands: []*cobra.Command{
				NewCmdApply("kubectl", f, out, err),
				NewCmdPatch(f, out),
				NewCmdReplace(f, out),
				NewCmdConvert(f, out),
			},
		},
		{
			Message: "Settings Commands:",
			Commands: []*cobra.Command{
				NewCmdLabel(f, out),
				NewCmdAnnotate(f, out),
				NewCmdCompletion(out, ""),
			},
		},
	}
	groups.Add(cmds)

	filters := []string{"options"}

	// Hide the "alpha" subcommand if there are no alpha commands in this build.
	alpha := NewCmdAlpha(f, in, out, err)
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
	cmds.AddCommand(cmdconfig.NewCmdConfig(clientcmd.NewDefaultPathOptions(), out, err))
	cmds.AddCommand(NewCmdPlugin(f, in, out, err))
	cmds.AddCommand(NewCmdVersion(f, out))
	cmds.AddCommand(NewCmdApiVersions(f, out))
	cmds.AddCommand(NewCmdOptions(out))

	return cmds
}

func runHelp(cmd *cobra.Command, args []string) {
	cmd.Help()
}

func printDeprecationWarning(command, alias string) {
	glog.Warningf("%s is DEPRECATED and will be removed in a future version. Use %s instead.", alias, command)
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

// deprecated is similar to deprecatedAlias, but it is used for deprecations
// that are not simple aliases; this command is actually a different
// (deprecated) codepath.
func deprecated(baseName, to string, parent, cmd *cobra.Command) string {
	cmd.Long = fmt.Sprintf("Deprecated: all functionality can be found in \"%s %s\"", baseName, to)
	cmd.Short = fmt.Sprintf("Deprecated: use %s", to)
	parent.AddCommand(cmd)
	return cmd.Name()
}
