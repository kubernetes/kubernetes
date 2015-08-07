/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	cmdconfig "k8s.io/kubernetes/pkg/kubectl/cmd/config"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util"

	"github.com/spf13/cobra"
)

const (
	bash_completion_func = `# call kubectl get $1,
__kubectl_parse_get()
{
    local template
    template="{{ range .items  }}{{ .metadata.name }} {{ end }}"
    local kubectl_out
    if kubectl_out=$(kubectl get -o template --template="${template}" "$1" 2>/dev/null); then
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
    if kubectl_out=$(kubectl get -o template --template="${template}" pods "${last}" 2>/dev/null); then
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
        kubectl_get | kubectl_describe | kubectl_delete | kubectl_label | kubectl_stop)
	    __kubectl_get_resource
            return
            ;;
	kubectl_logs)
	    __kubectl_require_pod_and_container
	    return
	    ;;
        *)
            ;;
    esac
}
`
	valid_resources = `Valid resource types include:
   * pods (aka 'po')
   * replicationcontrollers (aka 'rc')
   * services (aka 'svc')
   * events (aka 'ev')
   * nodes (aka 'no')
   * namespaces (aka 'ns')
   * secrets
   * persistentvolumes (aka 'pv')
   * persistentvolumeclaims (aka 'pvc')
   * limitranges (aka 'limits')
   * resourcequotas (aka 'quota')
`
)

// NewKubectlCommand creates the `kubectl` command and its nested children.
func NewKubectlCommand(f *cmdutil.Factory, in io.Reader, out, err io.Writer) *cobra.Command {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "kubectl",
		Short: "kubectl controls the Kubernetes cluster manager",
		Long: `kubectl controls the Kubernetes cluster manager.

Find more information at https://github.com/GoogleCloudPlatform/kubernetes.`,
		Run: runHelp,
		BashCompletionFunction: bash_completion_func,
	}

	f.BindFlags(cmds.PersistentFlags())

	// From this point and forward we get warnings on flags that contain "_" separators
	cmds.SetGlobalNormalizationFunc(util.WarnWordSepNormalizeFunc)

	cmds.AddCommand(NewCmdGet(f, out))
	cmds.AddCommand(NewCmdDescribe(f, out))
	cmds.AddCommand(NewCmdCreate(f, out))
	cmds.AddCommand(NewCmdReplace(f, out))
	cmds.AddCommand(NewCmdPatch(f, out))
	cmds.AddCommand(NewCmdDelete(f, out))

	cmds.AddCommand(NewCmdNamespace(out))
	cmds.AddCommand(NewCmdLog(f, out))
	cmds.AddCommand(NewCmdRollingUpdate(f, out))
	cmds.AddCommand(NewCmdScale(f, out))

	cmds.AddCommand(NewCmdAttach(f, in, out, err))
	cmds.AddCommand(NewCmdExec(f, in, out, err))
	cmds.AddCommand(NewCmdPortForward(f))
	cmds.AddCommand(NewCmdProxy(f, out))

	cmds.AddCommand(NewCmdRun(f, in, out, err))
	cmds.AddCommand(NewCmdStop(f, out))
	cmds.AddCommand(NewCmdExposeService(f, out))

	cmds.AddCommand(NewCmdLabel(f, out))
	cmds.AddCommand(NewCmdAnnotate(f, out))

	cmds.AddCommand(cmdconfig.NewCmdConfig(cmdconfig.NewDefaultPathOptions(), out))
	cmds.AddCommand(NewCmdClusterInfo(f, out))
	cmds.AddCommand(NewCmdApiVersions(f, out))
	cmds.AddCommand(NewCmdVersion(f, out))

	return cmds
}

func runHelp(cmd *cobra.Command, args []string) {
	cmd.Help()
}

func printDeprecationWarning(command, alias string) {
	glog.Warningf("%s is DEPRECATED and will be removed in a future version. Use %s instead.", alias, command)
}
