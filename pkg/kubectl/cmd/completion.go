/*
Copyright 2016 The Kubernetes Authors.

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
	"bytes"
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

const defaultBoilerPlate = `
# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
`

var (
	completion_long = templates.LongDesc(i18n.T(`
		Output shell completion code for the specified shell (bash, zsh or fish).
		The shell code must be evaluated to provide interactive
		completion of kubectl commands.  This can be done by sourcing it from
		the .bash_profile.

		Detailed instructions on how to do this are available here:
		https://kubernetes.io/docs/tasks/tools/install-kubectl/#enabling-shell-autocompletion

		Note for zsh users: [1] zsh completions are only supported in versions of zsh >= 5.2`))

	completion_example = templates.Examples(i18n.T(`
		# Installing bash completion on macOS using homebrew
		## If running Bash 3.2 included with macOS
		    brew install bash-completion
		## or, if running Bash 4.1+
		    brew install bash-completion@2
		## If kubectl is installed via homebrew, this should start working immediately.
		## If you've installed via other means, you may need add the completion to your completion directory
		    kubectl completion bash > $(brew --prefix)/etc/bash_completion.d/kubectl


		# Installing bash completion on Linux
		## Load the kubectl completion code for bash into the current shell
		    source <(kubectl completion bash)
		## Write bash completion code to a file and source if from .bash_profile
		    kubectl completion bash > ~/.kube/completion.bash.inc
		    printf "
		      # Kubectl shell completion
		      source '$HOME/.kube/completion.bash.inc'
		      " >> $HOME/.bash_profile
		    source $HOME/.bash_profile

		# Load the kubectl completion code for zsh[1] into the current shell
		    source <(kubectl completion zsh)
		# Set the kubectl completion code for zsh[1] to autoload on startup
		    kubectl completion zsh > "${fpath[1]}/_kubectl"

		# Load the kubectl completion code for fish into the current shell
		    kubectl completion fish | source
		# Set the kubectl completion code for fish to autoload on startup
		    kubectl completion fish > "$HOME/.config/fish/completions/kubectl.fish"`))
)

var (
	completion_shells = map[string]func(out io.Writer, boilerPlate string, cmd *cobra.Command) error{
		"bash": runCompletionBash,
		"zsh":  runCompletionZsh,
		"fish":  runCompletionFish,
	}
)

func NewCmdCompletion(out io.Writer, boilerPlate string) *cobra.Command {
	shells := []string{}
	for s := range completion_shells {
		shells = append(shells, s)
	}

	cmd := &cobra.Command{
		Use: "completion SHELL",
		DisableFlagsInUseLine: true,
		Short:   i18n.T("Output shell completion code for the specified shell (bash, zsh or fish)"),
		Long:    completion_long,
		Example: completion_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunCompletion(out, boilerPlate, cmd, args)
			cmdutil.CheckErr(err)
		},
		ValidArgs: shells,
	}

	return cmd
}

func RunCompletion(out io.Writer, boilerPlate string, cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return cmdutil.UsageErrorf(cmd, "Shell not specified.")
	}
	if len(args) > 1 {
		return cmdutil.UsageErrorf(cmd, "Too many arguments. Expected only the shell type.")
	}
	run, found := completion_shells[args[0]]
	if !found {
		return cmdutil.UsageErrorf(cmd, "Unsupported shell type %q.", args[0])
	}

	return run(out, boilerPlate, cmd.Parent())
}

func runCompletionBash(out io.Writer, boilerPlate string, kubectl *cobra.Command) error {
	if len(boilerPlate) == 0 {
		boilerPlate = defaultBoilerPlate
	}
	if _, err := out.Write([]byte(boilerPlate)); err != nil {
		return err
	}

	return kubectl.GenBashCompletion(out)
}

func runCompletionZsh(out io.Writer, boilerPlate string, kubectl *cobra.Command) error {
	zsh_head := "#compdef kubectl\n"

	out.Write([]byte(zsh_head))

	if len(boilerPlate) == 0 {
		boilerPlate = defaultBoilerPlate
	}
	if _, err := out.Write([]byte(boilerPlate)); err != nil {
		return err
	}

	zsh_initialization := `
__kubectl_bash_source() {
	alias shopt=':'
	alias _expand=_bash_expand
	alias _complete=_bash_comp
	emulate -L sh
	setopt kshglob noshglob braceexpand

	source "$@"
}

__kubectl_type() {
	# -t is not supported by zsh
	if [ "$1" == "-t" ]; then
		shift

		# fake Bash 4 to disable "complete -o nospace". Instead
		# "compopt +-o nospace" is used in the code to toggle trailing
		# spaces. We don't support that, but leave trailing spaces on
		# all the time
		if [ "$1" = "__kubectl_compopt" ]; then
			echo builtin
			return 0
		fi
	fi
	type "$@"
}

__kubectl_compgen() {
	local completions w
	completions=( $(compgen "$@") ) || return $?

	# filter by given word as prefix
	while [[ "$1" = -* && "$1" != -- ]]; do
		shift
		shift
	done
	if [[ "$1" == -- ]]; then
		shift
	fi
	for w in "${completions[@]}"; do
		if [[ "${w}" = "$1"* ]]; then
			echo "${w}"
		fi
	done
}

__kubectl_compopt() {
	true # don't do anything. Not supported by bashcompinit in zsh
}

__kubectl_ltrim_colon_completions()
{
	if [[ "$1" == *:* && "$COMP_WORDBREAKS" == *:* ]]; then
		# Remove colon-word prefix from COMPREPLY items
		local colon_word=${1%${1##*:}}
		local i=${#COMPREPLY[*]}
		while [[ $((--i)) -ge 0 ]]; do
			COMPREPLY[$i]=${COMPREPLY[$i]#"$colon_word"}
		done
	fi
}

__kubectl_get_comp_words_by_ref() {
	cur="${COMP_WORDS[COMP_CWORD]}"
	prev="${COMP_WORDS[${COMP_CWORD}-1]}"
	words=("${COMP_WORDS[@]}")
	cword=("${COMP_CWORD[@]}")
}

__kubectl_filedir() {
	local RET OLD_IFS w qw

	__debug "_filedir $@ cur=$cur"
	if [[ "$1" = \~* ]]; then
		# somehow does not work. Maybe, zsh does not call this at all
		eval echo "$1"
		return 0
	fi

	OLD_IFS="$IFS"
	IFS=$'\n'
	if [ "$1" = "-d" ]; then
		shift
		RET=( $(compgen -d) )
	else
		RET=( $(compgen -f) )
	fi
	IFS="$OLD_IFS"

	IFS="," __debug "RET=${RET[@]} len=${#RET[@]}"

	for w in ${RET[@]}; do
		if [[ ! "${w}" = "${cur}"* ]]; then
			continue
		fi
		if eval "[[ \"\${w}\" = *.$1 || -d \"\${w}\" ]]"; then
			qw="$(__kubectl_quote "${w}")"
			if [ -d "${w}" ]; then
				COMPREPLY+=("${qw}/")
			else
				COMPREPLY+=("${qw}")
			fi
		fi
	done
}

__kubectl_quote() {
    if [[ $1 == \'* || $1 == \"* ]]; then
        # Leave out first character
        printf %q "${1:1}"
    else
    	printf %q "$1"
    fi
}

autoload -U +X bashcompinit && bashcompinit

# use word boundary patterns for BSD or GNU sed
LWORD='[[:<:]]'
RWORD='[[:>:]]'
if sed --help 2>&1 | grep -q GNU; then
	LWORD='\<'
	RWORD='\>'
fi

__kubectl_convert_bash_to_zsh() {
	sed \
	-e 's/declare -F/whence -w/' \
	-e 's/_get_comp_words_by_ref "\$@"/_get_comp_words_by_ref "\$*"/' \
	-e 's/local \([a-zA-Z0-9_]*\)=/local \1; \1=/' \
	-e 's/flags+=("\(--.*\)=")/flags+=("\1"); two_word_flags+=("\1")/' \
	-e 's/must_have_one_flag+=("\(--.*\)=")/must_have_one_flag+=("\1")/' \
	-e "s/${LWORD}_filedir${RWORD}/__kubectl_filedir/g" \
	-e "s/${LWORD}_get_comp_words_by_ref${RWORD}/__kubectl_get_comp_words_by_ref/g" \
	-e "s/${LWORD}__ltrim_colon_completions${RWORD}/__kubectl_ltrim_colon_completions/g" \
	-e "s/${LWORD}compgen${RWORD}/__kubectl_compgen/g" \
	-e "s/${LWORD}compopt${RWORD}/__kubectl_compopt/g" \
	-e "s/${LWORD}declare${RWORD}/builtin declare/g" \
	-e "s/\\\$(type${RWORD}/\$(__kubectl_type/g" \
	<<'BASH_COMPLETION_EOF'
`
	out.Write([]byte(zsh_initialization))

	buf := new(bytes.Buffer)
	kubectl.GenBashCompletion(buf)
	out.Write(buf.Bytes())

	zsh_tail := `
BASH_COMPLETION_EOF
}

__kubectl_bash_source <(__kubectl_convert_bash_to_zsh)
_complete kubectl 2>/dev/null
`
	out.Write([]byte(zsh_tail))
	return nil
}

func runCompletionFish(out io.Writer, boilerPlate string, kubectl *cobra.Command) error {
	if len(boilerPlate) == 0 {
		boilerPlate = defaultBoilerPlate
	}
	if _, err := out.Write([]byte(boilerPlate)); err != nil {
		return err
	}

	fish_body := `
function __kubectl_no_command
    set -l cmd (commandline -poc)
    if not set -q cmd[2]
        return 0
    end
    return 1
end

function __kubectl_using_command
    set cmd (commandline -poc)

    if test (count $cmd) -gt (count $argv)
        set -e cmd[1]
        string match -q -- "$argv*" "$cmd"
        return $status
    end

    return 1
end

function __kubectl_using_option
    set cmd (commandline -poc)
    set query "("(string join -- "|" $argv)")"

    if test (count $cmd) -gt 1
        if string match -qr -- $query $cmd[-1]
            return 0
        end
    end
    return 1
end

function __kubectl_using_option_value -a option -a value
    set cmd (commandline -poc)

    if test (count $cmd) -gt 1
        string match -qr -- $option"[= ]"$value "$cmd"
        return $status
    end

    return 1
end

function __kubectl_using_go_template_format
    __kubectl_using_option_value output go-template
    or __kubectl_using_option_value output go-template-file
    or __kubectl_using_option_value o go-template
    or __kubectl_using_option_value o go-template-file
end

function __kubectl_using_resource_type
    for rt in (__kubectl_resource_types)
        if __fish_seen_subcommand_from $rt
            return 0
        end
    end

    return 1
end

function __kubectl_no_pod
    set cmd (commandline -pc)
    for i in (__kubectl_pods --no-prefix pods)
        if string match -q "*$i*" -- $cmd
            return 1
        end
    end
    return 0
end

function __kubectl_resource_types
    echo clusters
    echo componentstatuses
    echo configmaps
    echo daemonsets
    echo deployments
    echo endpoints
    echo events
    echo horizontalpodautoscalers
    echo ingresses
    echo jobs
    echo limitranges
    echo namespaces
    echo networkpolicies
    echo nodes
    echo persistentvolumeclaims
    echo persistentvolumes
    echo pods
    echo podsecuritypolicies
    echo podtemplates
    echo replicasets
    echo replicationcontrollers
    echo resourcequotas
    echo secrets
    echo serviceaccounts
    echo services
    echo statefulsets
    echo storageclasses
    echo thirdpartyresources
end

function __kubectl_resource_type_description -a type
    switch $type
        case clusters cluster
            echo "Cluster"
        case componentstatuses componentstatus cs
            echo "Status"
        case configmaps configmap cm
            echo "Config Map"
        case daemonsets daemonset ds
            echo "Daemon Set"
        case deployments deployment deploy
            echo "Deployment"
        case endpoints endpoint ep
            echo "Endpoint"
        case events event ev
            echo "Event"
        case horizontalpodautoscalers horizontalpodautoscaler hpa
            echo "Autoscaler"
        case ingresses ingress ing
            echo "Ingress"
        case jobs job
            echo "Job"
        case limitranges limitrange limits
            echo "Range"
        case namespaces namespace ns
            echo "Namespace"
        case networkpolicies networkpolicy
            echo "Policy"
        case nodes node no
            echo "Node"
        case persistentvolumeclaims persistentvolumeclaim pvc
            echo "Volume Claim"
        case persistentvolumes persistentvolume pv
            echo "Volume"
        case pods pod po
            echo "Pod"
        case podsecuritypolicies podsecuritypolicy psp
            echo "Policy"
        case podtemplates podtemplate
            echo "Template"
        case replicasets replicaset rs
            echo "Replica Set"
        case replicationcontrollers replicationcontroller rc
            echo "RC"
        case resourcequotas resourcequota quota
            echo "Quota"
        case secrets secret
            echo "Secret"
        case serviceaccounts serviceaccount sa
            echo "Account"
        case services service svc
            echo "Service"
        case statefulsets statefulset
            echo "Stateful Set"
        case storageclasses storageclass
            echo "Storage Class"
        case thirdpartyresources thirdpartyresource
            echo "Resource"
    end
end

function __kubectl_resources
    set -l prefix 1

    for i in $argv
        switch $i
            case '--no-prefix'
                set -e prefix
                set idx (contains -i -- --no-prefix $argv)
                set -e argv[$idx]
        end
    end

    set cmd (commandline -pc)
    set namespace (string replace -r '^kubectl .*(-n |--namespace[= ]?)([^ ]*) .*$' '$2' -- $cmd)

    for resource in $argv
        if set -lq prefix
            kubectl get $resource -n "$namespace" -o name ^/dev/null
        else
            kubectl get $resource -n "$namespace" -o name ^/dev/null | string replace -r '.*/' ''
        end
    end
end

function __kubectl_containers
    set namespace (string replace -r '^kubectl .*(-n |--namespace[= ]?)([^ ]*) .*$' '$2' -- $argv)
    set pod

    for i in (__kubectl_resources --no-prefix pods)
        if string match -q "*$i*" -- $argv
            set pod $i
            break
        end
    end

    if test -z "$pod"
        return
    end

    kubectl get -n "$namespace" pods "$pod" -o 'jsonpath={.spec.containers[*].name}' | string split ' '
end

function __kubectl_pods_completion
    set cmd (commandline -pc)
    set namespace (string replace -r '^kubectl .*(-n |--namespace[= ]?)([^ ]*) .*$' '$2' -- $cmd)

    kubectl get pods -n "$namespace" ^/dev/null | tail -n +2 | awk '{print $1"\tPod "$2" "$3}'
end

function __kubectl_generate_get_resource_subcommand_completions
    for subcmd in (__kubectl_resource_types)
        set arguments "(__kubectl_resources $subcmd | string replace -r '^.*/' '')"
        set description (__kubectl_resource_type_description $subcmd)
        complete -c kubectl -f -n "__kubectl_using_command get; and __fish_seen_subcommand_from $subcmd" -a $arguments -d $description
    end
end

function __kubectl_get_prefixed_resource_completions
    set cmd (commandline -po)

    if string match -qr '[a-zA-Z]+/.*$' -- $cmd[-1]
        set type (string replace -r '([a-zA-Z]+)/.*$' '$1' -- $cmd[-1])
        set description (__kubectl_resource_type_description $type)
        printf "%s\t$description\n" (__kubectl_resources $type)
    end
end

function __kubectl_output_formats
    echo json
    echo yaml
    echo wide
    echo name
    echo custom-columns=
    echo custom-columns-file=
    echo go-template=
    echo go-template-file=
    echo jsonpath=
    echo jsonpath-file=
end

function __kubectl_subcommands -a cmd
    switch $cmd
        case create
            echo configmap\t"Create a configmap from a local file, directory or literal value"
            echo deployment\t"Create a deployment with the specified name"
            echo namespace\t"Create a namespace with the specified name"
            echo quota\t"Create a quota with the specified name"
            echo secret\t"Create a secret using specified subcommand"
            echo service\t"Create a service using specified subcommand"
            echo serviceaccount\t"Create a service account with the specified name"
        case set
            echo image\t"Update image of a pod template"
            echo resources\t"Update resource requests/limits on objects with pod templates"
    end
end

# Global command-line options:

complete -c kubectl -l "alsologtostderr" -d "Log to standard error as well as files"
complete -c kubectl -l "as" -d "Username to impersonate for the operation"
complete -c kubectl -l "certificate-authority" -d "Path to a cert. file for the certificate authority"
complete -c kubectl -l "client-certificate" -d "Path to a client certificate file for TLS"
complete -c kubectl -l "client-key" -d "Path to a client key file for TLS"
complete -c kubectl -l "cluster" -d "The name of the kubeconfig cluster to use"
complete -c kubectl -l "context" -d "The name of the kubeconfig context to use"
complete -c kubectl -l "insecure-skip-tls-verify" -d "Certificate will not be checked for validity"
complete -c kubectl -l "kubeconfig" -d "Path to the kubeconfig file to use for CLI requests"
complete -c kubectl -l "log-backtrace-at" -d "Emit a stack trace when logging hits line file:N"
complete -c kubectl -l "log-dir" -d "Write log files in this directory"
complete -c kubectl -l "log-flush-frequency" -d "Maximum number of seconds between log flushes"
complete -c kubectl -l "logtostderr" -d "Log to standard error instead of files [default true]"
complete -c kubectl -l "match-server-version" -d "Require server version to match client version"
complete -c kubectl -s n -l "namespace" -d "The namespace scope for this CLI request"
complete -c kubectl -l "password" -d "Password for basic authentication to the API server"
complete -c kubectl -l "request-timeout" -d "Timeout for a single server request"
complete -c kubectl -s s -l "server" -d "The address and port of the Kubernetes API server"
complete -c kubectl -l "stderrthreshold" -d "Logs at or above this threshold go to stderr"
complete -c kubectl -l "token" -d "Bearer token for authentication to the API server"
complete -c kubectl -l "user" -d "The name of the kubeconfig user to use"
complete -c kubectl -l "username" -d "Username for basic authentication to the API server"
complete -c kubectl -s v -l "v" -d "Log level for V logs"
complete -c kubectl -l "vmodule" -d "List of settings for file-filtered logging"

complete -c kubectl -f -n "__kubectl_using_option -n --namespace" -a "(__kubectl_resources --no-prefix namespaces)" -d "Namespace"

# Basic Commands (Beginner):

## create
complete -c kubectl -f -n "__kubectl_no_command" -a create -d "Create a resource by filename or stdin"

complete -c kubectl -f -n "__kubectl_using_command create" -a "(__kubectl_subcommands create)"

complete -c kubectl -f -n "__kubectl_using_command create" -l "dry-run" -d "Only print the object that would be sent"
complete -c kubectl -f -n "__kubectl_using_command create" -l "edit" -d "Edit the API resource before creating"
complete -c kubectl -f -n "__kubectl_using_command create; and __kubectl_using_option_value edit true" -l "windows-line-endings" -d "Use Windows line-endings"
complete -c kubectl -n "__kubectl_using_command create" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command create; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -f -n "__kubectl_using_command create" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command create" -l "no-headers" -d "Don't print headers"
complete -c kubectl -f -n "__kubectl_using_command create" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command create; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command create; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -f -n "__kubectl_using_command create" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -f -n "__kubectl_using_command create" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -f -n "__kubectl_using_command create" -l "save-config" -d "The config of current object will be saved in its annotation"
complete -c kubectl -f -n "__kubectl_using_command create" -l "schema-cache-dir" -d "Load/store cached API schemas in this directory"
complete -c kubectl -f -n "__kubectl_using_command create" -s a -l "show-all" -d "When printing, show all resources"
complete -c kubectl -f -n "__kubectl_using_command create" -l "show-labels" -d "When printing, show all labels as the last column"
complete -c kubectl -f -n "__kubectl_using_command create" -l "sort-by" -d "Sort list types using this field specification"
complete -c kubectl -f -n "__kubectl_using_command create" -l "validate" -d "Use a schema to validate the input before sending it [default true]"

### create configmap
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from configmap" -l "from-file" -d "File or directory to find config files, with optional key prefix"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from configmap" -l "from-literal" -d "Specify a key and literal value to insert in configmap"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from configmap" -l "generator" -d "The name of the API generator to use"

### create deployment
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from deployment" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from deployment" -l "image" -d "Image name to run"

### create namespace
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from namespace" -l "generator" -d "The name of the API generator to use"

### create quota
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from quota" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from quota" -l "hard" -d "A comma-delimited set of resource=quantity pairs that define a hard limit"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from quota" -l "scopes" -d "A comma-delimited set of quota scopes that must all match each object tracked by the quota"

### create secret
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret" -a "docker-registry" -d "Create a secret for use with a Docker registry"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret" -a "generic" -d "Create a secret from a local file, directory or literal value"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret" -a "tls" -d "Create a TLS secret"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from docker-registry" -l "docker-email" -d "Email for Docker registry"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from docker-registry" -l "docker-password" -d "Password for Docker registry authentication"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from docker-registry" -l "docker-server" -d "Server location for Docker registry"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from docker-registry" -l "docker-username" -d "Username for Docker registry authentication"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from generic" -l "from-file" -d "File or directory to find config files, with optional key prefix"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from generic" -l "from-literal" -d "Specify a key and literal value to insert in configmap"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from generic" -l "type" -d "The type of secret to create"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from tls" -l "cert" -d "Path to PEM encoded public key certificate"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from tls" -l "key" -d "Path to private key associated with given certificate"

### create service
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service" -l "tcp" -d "Port pairs can be specified as '<port>:<targetPort>'"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service" -a "clusterip loadbalancer nodeport" -d "Service"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service; and __fish_seen_subcommand_from clusterip" -l "clusterip" -d "Assign your own ClusterIP or set to 'None' for a 'headless' service"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service; and __fish_seen_subcommand_from nodeport" -l "node-port" -d "Port used to expose the service on each node in a cluster"

### create serviceaccount
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from serviceaccount" -l "generator" -d "The name of the API generator to use"

## expose
complete -c kubectl -f -n "__kubectl_no_command" -a expose -d "Expose a resource as a new Kubernetes Service"

complete -c kubectl -f -n "__kubectl_using_command expose" -a "pod service replicationcontroller deployment replicaset" -d "Resource Type"

complete -c kubectl -f -n "__kubectl_using_command expose" -l "cluster-ip" -d "ClusterIP to be assigned to the service"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "container-port" -l "target-port" -d "Name or number for the port on the container that the service should direct traffic to"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "create-external-load-balancer" -d "Create an external load balancer for this service"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "dry-run" -d "Only print the object that would be sent"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "external-ip" -d "Additional external IP address to accept for the service"
complete -c kubectl -n "__kubectl_using_command expose" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command expose" -s l -l "labels" -d "Labels to apply to the service created by this call"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "load-balancer-ip" -d "IP to assign to the Load Balancer"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "name" -d "The name for the newly created object"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "no-headers" -d "Don't print headers"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "overrides" -d "An inline JSON override for the generated object"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "port" -d "The port that the service should serve on"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "protocol" -d "The network protocol for the service to be created"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option --protocol" -a "TCP UDP" -d "Protocol"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "save-config" -d "The config of current object will be saved in its annotation"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "selector" -d "A label selector to use for this service"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "session-affinity" -d "Set the session affinity for the service to this"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option --session-affinity" -a "None ClientIP" -d "Affinity"
complete -c kubectl -f -n "__kubectl_using_command expose" -s a -l "show-all" -d "Show all resources"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "show-labels" -d "Show all labels as the last column"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "sort-by" -d "Sort list types using this field specification"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "type" -d "Type for this service"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option --type" -a "ClusterIP NodePort LoadBalancer" -d "Type"

complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from pod" -a "(__kubectl_pods_completion)"
complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from service" -a "(__kubectl_resources --no-prefix services)" -d "Service"
complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from replicationcontroller" -a "(__kubectl_resources --no-prefix replicationcontrollers)" -d "RC"
complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from deployment" -a "(__kubectl_resources --no-prefix deployments)" -d "Deployment"
complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from replicaset" -a "(__kubectl_resources --no-prefix replicasets)" -d "Replica Set"

## run
complete -c kubectl -f -n "__kubectl_no_command" -a run -d "Run a particular image on the cluster"

complete -c kubectl -f -n "__kubectl_using_command run" -l "attach" -d "Wait for the Pod to start running"
complete -c kubectl -f -n "__kubectl_using_command run" -l command -d "Use extra arguments as the command field in the container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "dry-run" -d "Only print the object that would be sent"
complete -c kubectl -f -n "__kubectl_using_command run" -l "env" -d "Environment variables to set in the container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "expose" -d "A public, external service is created for the container(s) which are run"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_option --expose" -l "service-generator" -d "The name of the generator to use for creating a service"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_option --expose" -l "service-overrides" -d "An inline JSON override for the generated service object"
complete -c kubectl -f -n "__kubectl_using_command run" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command run" -l "hostport" -d "The host port mapping for the container port"
complete -c kubectl -f -n "__kubectl_using_command run" -l "image" -d "The image for the container to run"
complete -c kubectl -f -n "__kubectl_using_command run" -l "image-pull-policy" -d "The image pull policy for the container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command run" -s l -l "labels" -d "Labels to apply to the pod(s)"
complete -c kubectl -f -n "__kubectl_using_command run" -l "leave-stdin-open" -d "Leave stdin open after the first attach completes"
complete -c kubectl -f -n "__kubectl_using_command run" -l "limits" -d "The resource requirement limits for this container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "no-headers" -d "Don't print headers"
complete -c kubectl -f -n "__kubectl_using_command run" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -f -n "__kubectl_using_command run" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -f -n "__kubectl_using_command run" -l "overrides" -d "An inline JSON override for the generated object"
complete -c kubectl -f -n "__kubectl_using_command run" -l "port" -d "The port that this container exposes"
complete -c kubectl -f -n "__kubectl_using_command run" -l "quiet" -d "Suppress prompt messages"
complete -c kubectl -f -n "__kubectl_using_command run" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -f -n "__kubectl_using_command run" -s r -l "replicas" -d "Number of replicas to create for this container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "requests" -d "The resource requirement requests for this container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "restart" -d "The restart policy for this Pod"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_option --restart" -a "Always OnFailure Never" -d "Policy"
complete -c kubectl -f -n "__kubectl_using_command run" -l "rm" -d "Delete resources created in this command for attached containers"
complete -c kubectl -f -n "__kubectl_using_command run" -l "save-config" -d "The config of current object will be saved in its annotation"
complete -c kubectl -f -n "__kubectl_using_command run" -l "schedule" -d "A schedule in the Cron format the job should be run with"
complete -c kubectl -f -n "__kubectl_using_command run" -s a -l "show-all" -d "Show all resources"
complete -c kubectl -f -n "__kubectl_using_command run" -l "show-labels" -d "Show all labels as the last column"
complete -c kubectl -f -n "__kubectl_using_command run" -l "sort-by" -d "Sort list types using this field specification"
complete -c kubectl -f -n "__kubectl_using_command run" -s i -l "stdin" -d "Keep stdin open on the container(s) in the pod"
complete -c kubectl -f -n "__kubectl_using_command run" -s t -l "tty" -d "Allocated a TTY for each container in the pod"

## set
complete -c kubectl -f -n "__kubectl_no_command" -a set -d "Set specific features on objects"
complete -c kubectl -f -n "__kubectl_using_command set" -a "(__kubectl_subcommands set)"

### set image
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -a "pod replicationcontroller deployment daemonset job replicaset" -d "Resource Type"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -a "(__kubectl_resources pod replicationcontroller deployment daemonset job replicaset)" -d "Resource"

complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "all" -d "Select all resources in the namespace of the specified resource types"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from image; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "local" -d "Set image will NOT contact api-server but run locally"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "no-headers" -d "Don't print headers"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from image; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from image; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -s l -l "selector" -d "Selector (label query) to filter on"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -s a -l "show-all" -d "Show all resources"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "show-labels" -d "Show all labels as the last column"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "sort-by" -d "Sort list types using this field specification"

### set resources
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -a "replicationcontroller deployment daemonset job replicaset" -d "Resource Type"

complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "all" -d "Select all resources in the namespace of the specified resource types"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -s c -l "containers" -d "The names of containers in the selected pod templates to change"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "dry-run" -d "Only print the object that would be sent"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "limits" -d "The resource requirement requests for this container"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "local" -d "Set resources will NOT contact api-server but run locally"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "no-headers" -d "Don't print headers"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "requests" -d "The resource requirement requests for this container"

# Basic Commands (Intermediate):

## get
complete -c kubectl -f -n "__kubectl_no_command" -a get -d "Display one or many resources"

complete -c kubectl -f -n "__kubectl_using_command get" -l "all-namespaces" -d "List the requested object(s) across all namespaces"
complete -c kubectl -f -n "__kubectl_using_command get" -l "export" -d "Strip cluster-specific info"
complete -c kubectl -f -n "__kubectl_using_command get" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command get; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -f -n "__kubectl_using_command get" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command get" -s L -l "label-columns" -d "List of labels to be presented as columns"
complete -c kubectl -f -n "__kubectl_using_command get" -l "no-headers" -d "Don't print headers"
complete -c kubectl -n "__kubectl_using_command get" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command get; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command get; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -n "__kubectl_using_command get" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -n "__kubectl_using_command get" -l "raw" -d "Raw URI to request from the server"
complete -c kubectl -n "__kubectl_using_command get" -s l -l "selector" -d "Selector (label query) to filter on"
complete -c kubectl -n "__kubectl_using_command get" -s a -l "show-all" -d "Show all resources"
complete -c kubectl -n "__kubectl_using_command get" -l "show-kind" -d "List the resource type for the requested object(s)"
complete -c kubectl -n "__kubectl_using_command get" -l "show-labels" -d "Show all labels as the last column"
complete -c kubectl -n "__kubectl_using_command get" -l "sort-by" -d "Sort list types using this field specification"
complete -c kubectl -n "__kubectl_using_command get" -s w -l "watch" -d "Watch for changes after listing/getting"
complete -c kubectl -n "__kubectl_using_command get" -l "watch-only" -d "Watch for changes to the requested object(s)"

complete -c kubectl -f -n "__kubectl_using_command get; and not __kubectl_using_resource_type" -a "(__kubectl_resource_types)" -d "Resource Type"

# Generate completions for all resource sub-commands
__kubectl_generate_get_resource_subcommand_completions

complete -c kubectl -f -n "__kubectl_using_command get; and not __kubectl_using_resource_type" -a "(__kubectl_get_prefixed_resource_completions)"

## explain
complete -c kubectl -f -n "__kubectl_no_command" -a explain -d "Documentation of resources"

complete -c kubectl -f -n "__kubectl_using_command explain" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command explain" -l "recursive" -d "Print the fields of fields"

complete -c kubectl -f -n "__kubectl_using_command explain; and not __kubectl_using_resource_type" -a "(__kubectl_resource_types)" -d "Resource Type"

## edit
complete -c kubectl -f -n "__kubectl_no_command" -a edit -d "Edit a resource on the server"

complete -c kubectl -n "__kubectl_using_command edit" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command edit; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command edit" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command edit; __kubectl_using_option -o --output" -a "yaml json" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "save-config" -d "The config of current object will be saved in its annotation"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "schema-cache-dir" -d "Load/store cached API schemas in this directory"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "validate" -d "Use a schema to validate the input before sending"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "windows-line-endings" -d "Use Windows line-endings"

## delete
complete -c kubectl -f -n "__kubectl_no_command" -a delete -d "Delete resources"

# Deploy Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a rollout -d "Manage a deployment rollout"
complete -c kubectl -f -n "__kubectl_no_command" -a rolling-update -d "Perform a rolling update of the given ReplicationController"
complete -c kubectl -f -n "__kubectl_no_command" -a scale -d "Set a new size for a Deployment, ReplicaSet, Replication Controller, or Job"
complete -c kubectl -f -n "__kubectl_no_command" -a autoscale -d "Auto-scale a Deployment, ReplicaSet, or ReplicationController"

# Cluster Management Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a certificate -d "Modify certificate resources."
complete -c kubectl -f -n "__kubectl_no_command" -a cluster-info -d "Display cluster info"
complete -c kubectl -f -n "__kubectl_no_command" -a top -d "Display Resource (CPU/Memory/Storage) usage"
complete -c kubectl -f -n "__kubectl_no_command" -a cordon -d "Mark node as unschedulable"
complete -c kubectl -f -n "__kubectl_no_command" -a uncordon -d "Mark node as schedulable"
complete -c kubectl -f -n "__kubectl_no_command" -a drain -d "Drain node in preparation for maintenance"
complete -c kubectl -f -n "__kubectl_no_command" -a taint -d "Update the taints on one or more nodes"

# Troubleshooting and Debugging Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a describe -d "Show details of a specific resource or group of resources"
complete -c kubectl -f -n "__kubectl_no_command" -a logs -d "Print the logs for a container in a pod"
complete -c kubectl -f -n "__kubectl_no_command" -a attach -d "Attach to a running container"
complete -c kubectl -f -n "__kubectl_no_command" -a exec -d "Execute a command in a container"
complete -c kubectl -f -n "__kubectl_no_command" -a port-forward -d "Forward one or more local ports to a pod"
complete -c kubectl -f -n "__kubectl_no_command" -a proxy -d "Run a proxy to the Kubernetes API server"
complete -c kubectl -f -n "__kubectl_no_command" -a cp -d "Copy files and directories to and from containers."

# Advanced Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a apply -d "Apply a configuration to a resource by filename or stdin"
complete -c kubectl -f -n "__kubectl_no_command" -a patch -d "Update field(s) of a resource using strategic merge patch"
complete -c kubectl -f -n "__kubectl_no_command" -a replace -d "Replace a resource by filename or stdin"
complete -c kubectl -f -n "__kubectl_no_command" -a convert -d "Convert config files between different API versions"

# Settings Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a label -d "Update the labels on a resource"
complete -c kubectl -f -n "__kubectl_no_command" -a annotate -d "Update the annotations on a resource"
complete -c kubectl -f -n "__kubectl_no_command" -a completion -d "Output shell completion code for the given SHELL"

# Other Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a api-versions -d "Print the supported API versions on the server"
complete -c kubectl -f -n "__kubectl_no_command" -a config -d "Modify kubeconfig files"
complete -c kubectl -f -n "__kubectl_no_command" -a help -d "Help about any command"
complete -c kubectl -f -n "__kubectl_no_command" -a version -d "Print the client and server version information"
complete -c kubectl -f -n "__kubectl_no_command" -a options -d "Print the shared options"

# Attach
complete -c kubectl -n "__kubectl_using_command attach" -a "(__kubectl_pods_completion)"

# Exec
complete -c kubectl -n "__kubectl_using_command exec" -a "(__kubectl_pods_completion)"

# Describe
complete -c kubectl -n "__kubectl_using_command describe" -a "(__kubectl_resource_types)" -d "Resource Type"
complete -c kubectl -n "__kubectl_using_command describe" -a "(__kubectl_pods_completion)"

# Logs
complete -f -c kubectl -n "__kubectl_using_command logs; and __kubectl_no_pod" -a "(__kubectl_pods_completion)"
# kubernetes - is an open source system for managing containerized
# applications across multiple hosts, providing basic mechanisms for
# deployment, maintenance, and scaling of applications.
# See: https://kubernetes.io

function __kubectl_no_command
    set -l cmd (commandline -poc)
    if not set -q cmd[2]
        return 0
    end
    return 1
end

function __kubectl_using_command
    set cmd (commandline -poc)

    if test (count $cmd) -gt (count $argv)
        set -e cmd[1]
        string match -q -- "$argv*" "$cmd"
        return $status
    end

    return 1
end

function __kubectl_using_option
    set cmd (commandline -poc)
    set query "("(string join -- "|" $argv)")"

    if test (count $cmd) -gt 1
        if string match -qr -- $query $cmd[-1]
            return 0
        end
    end
    return 1
end

function __kubectl_using_option_value -a option -a value
    set cmd (commandline -poc)

    if test (count $cmd) -gt 1
        string match -qr -- $option"[= ]"$value "$cmd"
        return $status
    end

    return 1
end

function __kubectl_using_go_template_format
    __kubectl_using_option_value output go-template
    or __kubectl_using_option_value output go-template-file
    or __kubectl_using_option_value o go-template
    or __kubectl_using_option_value o go-template-file
end

function __kubectl_using_resource_type
    for rt in (__kubectl_resource_types)
        if __fish_seen_subcommand_from $rt
            return 0
        end
    end

    return 1
end

function __kubectl_no_pod
    set cmd (commandline -pc)
    for i in (__kubectl_pods --no-prefix pods)
        if string match -q "*$i*" -- $cmd
            return 1
        end
    end
    return 0
end

function __kubectl_resource_types
    echo clusters
    echo componentstatuses
    echo configmaps
    echo daemonsets
    echo deployments
    echo endpoints
    echo events
    echo horizontalpodautoscalers
    echo ingresses
    echo jobs
    echo limitranges
    echo namespaces
    echo networkpolicies
    echo nodes
    echo persistentvolumeclaims
    echo persistentvolumes
    echo pods
    echo podsecuritypolicies
    echo podtemplates
    echo replicasets
    echo replicationcontrollers
    echo resourcequotas
    echo secrets
    echo serviceaccounts
    echo services
    echo statefulsets
    echo storageclasses
    echo thirdpartyresources
end

function __kubectl_resource_type_description -a type
    switch $type
        case clusters cluster
            echo "Cluster"
        case componentstatuses componentstatus cs
            echo "Status"
        case configmaps configmap cm
            echo "Config Map"
        case daemonsets daemonset ds
            echo "Daemon Set"
        case deployments deployment deploy
            echo "Deployment"
        case endpoints endpoint ep
            echo "Endpoint"
        case events event ev
            echo "Event"
        case horizontalpodautoscalers horizontalpodautoscaler hpa
            echo "Autoscaler"
        case ingresses ingress ing
            echo "Ingress"
        case jobs job
            echo "Job"
        case limitranges limitrange limits
            echo "Range"
        case namespaces namespace ns
            echo "Namespace"
        case networkpolicies networkpolicy
            echo "Policy"
        case nodes node no
            echo "Node"
        case persistentvolumeclaims persistentvolumeclaim pvc
            echo "Volume Claim"
        case persistentvolumes persistentvolume pv
            echo "Volume"
        case pods pod po
            echo "Pod"
        case podsecuritypolicies podsecuritypolicy psp
            echo "Policy"
        case podtemplates podtemplate
            echo "Template"
        case replicasets replicaset rs
            echo "Replica Set"
        case replicationcontrollers replicationcontroller rc
            echo "RC"
        case resourcequotas resourcequota quota
            echo "Quota"
        case secrets secret
            echo "Secret"
        case serviceaccounts serviceaccount sa
            echo "Account"
        case services service svc
            echo "Service"
        case statefulsets statefulset
            echo "Stateful Set"
        case storageclasses storageclass
            echo "Storage Class"
        case thirdpartyresources thirdpartyresource
            echo "Resource"
    end
end

function __kubectl_resources
    set -l prefix 1

    for i in $argv
        switch $i
            case '--no-prefix'
                set -e prefix
                set idx (contains -i -- --no-prefix $argv)
                set -e argv[$idx]
        end
    end

    set cmd (commandline -pc)
    set namespace (string replace -r '^kubectl .*(-n |--namespace[= ]?)([^ ]*) .*$' '$2' -- $cmd)

    for resource in $argv
        if set -lq prefix
            kubectl get $resource -n "$namespace" -o name ^/dev/null
        else
            kubectl get $resource -n "$namespace" -o name ^/dev/null | string replace -r '.*/' ''
        end
    end
end

function __kubectl_containers
    set namespace (string replace -r '^kubectl .*(-n |--namespace[= ]?)([^ ]*) .*$' '$2' -- $argv)
    set pod

    for i in (__kubectl_resources --no-prefix pods)
        if string match -q "*$i*" -- $argv
            set pod $i
            break
        end
    end

    if test -z "$pod"
        return
    end

    kubectl get -n "$namespace" pods "$pod" -o 'jsonpath={.spec.containers[*].name}' | string split ' '
end

function __kubectl_pods_completion
    set cmd (commandline -pc)
    set namespace (string replace -r '^kubectl .*(-n |--namespace[= ]?)([^ ]*) .*$' '$2' -- $cmd)

    kubectl get pods -n "$namespace" ^/dev/null | tail -n +2 | awk '{print $1"\tPod "$2" "$3}'
end

function __kubectl_generate_get_resource_subcommand_completions
    for subcmd in (__kubectl_resource_types)
        set arguments "(__kubectl_resources $subcmd | string replace -r '^.*/' '')"
        set description (__kubectl_resource_type_description $subcmd)
        complete -c kubectl -f -n "__kubectl_using_command get; and __fish_seen_subcommand_from $subcmd" -a $arguments -d $description
    end
end

function __kubectl_get_prefixed_resource_completions
    set cmd (commandline -po)

    if string match -qr '[a-zA-Z]+/.*$' -- $cmd[-1]
        set type (string replace -r '([a-zA-Z]+)/.*$' '$1' -- $cmd[-1])
        set description (__kubectl_resource_type_description $type)
        printf "%s\t$description\n" (__kubectl_resources $type)
    end
end

function __kubectl_output_formats
    echo json
    echo yaml
    echo wide
    echo name
    echo custom-columns=
    echo custom-columns-file=
    echo go-template=
    echo go-template-file=
    echo jsonpath=
    echo jsonpath-file=
end

function __kubectl_subcommands -a cmd
    switch $cmd
        case create
            echo configmap\t"Create a configmap from a local file, directory or literal value"
            echo deployment\t"Create a deployment with the specified name"
            echo namespace\t"Create a namespace with the specified name"
            echo quota\t"Create a quota with the specified name"
            echo secret\t"Create a secret using specified subcommand"
            echo service\t"Create a service using specified subcommand"
            echo serviceaccount\t"Create a service account with the specified name"
        case set
            echo image\t"Update image of a pod template"
            echo resources\t"Update resource requests/limits on objects with pod templates"
    end
end

# Global command-line options:

complete -c kubectl -l "alsologtostderr" -d "Log to standard error as well as files"
complete -c kubectl -l "as" -d "Username to impersonate for the operation"
complete -c kubectl -l "certificate-authority" -d "Path to a cert. file for the certificate authority"
complete -c kubectl -l "client-certificate" -d "Path to a client certificate file for TLS"
complete -c kubectl -l "client-key" -d "Path to a client key file for TLS"
complete -c kubectl -l "cluster" -d "The name of the kubeconfig cluster to use"
complete -c kubectl -l "context" -d "The name of the kubeconfig context to use"
complete -c kubectl -l "insecure-skip-tls-verify" -d "Certificate will not be checked for validity"
complete -c kubectl -l "kubeconfig" -d "Path to the kubeconfig file to use for CLI requests"
complete -c kubectl -l "log-backtrace-at" -d "Emit a stack trace when logging hits line file:N"
complete -c kubectl -l "log-dir" -d "Write log files in this directory"
complete -c kubectl -l "log-flush-frequency" -d "Maximum number of seconds between log flushes"
complete -c kubectl -l "logtostderr" -d "Log to standard error instead of files [default true]"
complete -c kubectl -l "match-server-version" -d "Require server version to match client version"
complete -c kubectl -s n -l "namespace" -d "The namespace scope for this CLI request"
complete -c kubectl -l "password" -d "Password for basic authentication to the API server"
complete -c kubectl -l "request-timeout" -d "Timeout for a single server request"
complete -c kubectl -s s -l "server" -d "The address and port of the Kubernetes API server"
complete -c kubectl -l "stderrthreshold" -d "Logs at or above this threshold go to stderr"
complete -c kubectl -l "token" -d "Bearer token for authentication to the API server"
complete -c kubectl -l "user" -d "The name of the kubeconfig user to use"
complete -c kubectl -l "username" -d "Username for basic authentication to the API server"
complete -c kubectl -s v -l "v" -d "Log level for V logs"
complete -c kubectl -l "vmodule" -d "List of settings for file-filtered logging"

complete -c kubectl -f -n "__kubectl_using_option -n --namespace" -a "(__kubectl_resources --no-prefix namespaces)" -d "Namespace"

# Basic Commands (Beginner):

## create
complete -c kubectl -f -n "__kubectl_no_command" -a create -d "Create a resource by filename or stdin"

complete -c kubectl -f -n "__kubectl_using_command create" -a "(__kubectl_subcommands create)"

complete -c kubectl -f -n "__kubectl_using_command create" -l "dry-run" -d "Only print the object that would be sent"
complete -c kubectl -f -n "__kubectl_using_command create" -l "edit" -d "Edit the API resource before creating"
complete -c kubectl -f -n "__kubectl_using_command create; and __kubectl_using_option_value edit true" -l "windows-line-endings" -d "Use Windows line-endings"
complete -c kubectl -n "__kubectl_using_command create" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command create; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -f -n "__kubectl_using_command create" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command create" -l "no-headers" -d "Don't print headers"
complete -c kubectl -f -n "__kubectl_using_command create" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command create; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command create; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -f -n "__kubectl_using_command create" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -f -n "__kubectl_using_command create" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -f -n "__kubectl_using_command create" -l "save-config" -d "The config of current object will be saved in its annotation"
complete -c kubectl -f -n "__kubectl_using_command create" -l "schema-cache-dir" -d "Load/store cached API schemas in this directory"
complete -c kubectl -f -n "__kubectl_using_command create" -s a -l "show-all" -d "When printing, show all resources"
complete -c kubectl -f -n "__kubectl_using_command create" -l "show-labels" -d "When printing, show all labels as the last column"
complete -c kubectl -f -n "__kubectl_using_command create" -l "sort-by" -d "Sort list types using this field specification"
complete -c kubectl -f -n "__kubectl_using_command create" -l "validate" -d "Use a schema to validate the input before sending it [default true]"

### create configmap
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from configmap" -l "from-file" -d "File or directory to find config files, with optional key prefix"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from configmap" -l "from-literal" -d "Specify a key and literal value to insert in configmap"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from configmap" -l "generator" -d "The name of the API generator to use"

### create deployment
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from deployment" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from deployment" -l "image" -d "Image name to run"

### create namespace
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from namespace" -l "generator" -d "The name of the API generator to use"

### create quota
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from quota" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from quota" -l "hard" -d "A comma-delimited set of resource=quantity pairs that define a hard limit"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from quota" -l "scopes" -d "A comma-delimited set of quota scopes that must all match each object tracked by the quota"

### create secret
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret" -a "docker-registry" -d "Create a secret for use with a Docker registry"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret" -a "generic" -d "Create a secret from a local file, directory or literal value"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret" -a "tls" -d "Create a TLS secret"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from docker-registry" -l "docker-email" -d "Email for Docker registry"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from docker-registry" -l "docker-password" -d "Password for Docker registry authentication"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from docker-registry" -l "docker-server" -d "Server location for Docker registry"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from docker-registry" -l "docker-username" -d "Username for Docker registry authentication"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from generic" -l "from-file" -d "File or directory to find config files, with optional key prefix"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from generic" -l "from-literal" -d "Specify a key and literal value to insert in configmap"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from generic" -l "type" -d "The type of secret to create"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from tls" -l "cert" -d "Path to PEM encoded public key certificate"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from secret; and __fish_seen_subcommand_from tls" -l "key" -d "Path to private key associated with given certificate"

### create service
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service" -l "tcp" -d "Port pairs can be specified as '<port>:<targetPort>'"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service" -a "clusterip loadbalancer nodeport" -d "Service"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service; and __fish_seen_subcommand_from clusterip" -l "clusterip" -d "Assign your own ClusterIP or set to 'None' for a 'headless' service"
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from service; and __fish_seen_subcommand_from nodeport" -l "node-port" -d "Port used to expose the service on each node in a cluster"

### create serviceaccount
complete -c kubectl -f -n "__kubectl_using_command create; and __fish_seen_subcommand_from serviceaccount" -l "generator" -d "The name of the API generator to use"

## expose
complete -c kubectl -f -n "__kubectl_no_command" -a expose -d "Expose a resource as a new Kubernetes Service"

complete -c kubectl -f -n "__kubectl_using_command expose" -a "pod service replicationcontroller deployment replicaset" -d "Resource Type"

complete -c kubectl -f -n "__kubectl_using_command expose" -l "cluster-ip" -d "ClusterIP to be assigned to the service"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "container-port" -l "target-port" -d "Name or number for the port on the container that the service should direct traffic to"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "create-external-load-balancer" -d "Create an external load balancer for this service"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "dry-run" -d "Only print the object that would be sent"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "external-ip" -d "Additional external IP address to accept for the service"
complete -c kubectl -n "__kubectl_using_command expose" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command expose" -s l -l "labels" -d "Labels to apply to the service created by this call"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "load-balancer-ip" -d "IP to assign to the Load Balancer"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "name" -d "The name for the newly created object"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "no-headers" -d "Don't print headers"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "overrides" -d "An inline JSON override for the generated object"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "port" -d "The port that the service should serve on"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "protocol" -d "The network protocol for the service to be created"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option --protocol" -a "TCP UDP" -d "Protocol"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "save-config" -d "The config of current object will be saved in its annotation"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "selector" -d "A label selector to use for this service"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "session-affinity" -d "Set the session affinity for the service to this"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option --session-affinity" -a "None ClientIP" -d "Affinity"
complete -c kubectl -f -n "__kubectl_using_command expose" -s a -l "show-all" -d "Show all resources"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "show-labels" -d "Show all labels as the last column"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "sort-by" -d "Sort list types using this field specification"
complete -c kubectl -f -n "__kubectl_using_command expose" -l "type" -d "Type for this service"
complete -c kubectl -f -n "__kubectl_using_command expose; and __kubectl_using_option --type" -a "ClusterIP NodePort LoadBalancer" -d "Type"

complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from pod" -a "(__kubectl_pods_completion)"
complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from service" -a "(__kubectl_resources --no-prefix services)" -d "Service"
complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from replicationcontroller" -a "(__kubectl_resources --no-prefix replicationcontrollers)" -d "RC"
complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from deployment" -a "(__kubectl_resources --no-prefix deployments)" -d "Deployment"
complete -c kubectl -f -n "__kubectl_using_command expose; and __fish_seen_subcommand_from replicaset" -a "(__kubectl_resources --no-prefix replicasets)" -d "Replica Set"

## run
complete -c kubectl -f -n "__kubectl_no_command" -a run -d "Run a particular image on the cluster"

complete -c kubectl -f -n "__kubectl_using_command run" -l "attach" -d "Wait for the Pod to start running"
complete -c kubectl -f -n "__kubectl_using_command run" -l command -d "Use extra arguments as the command field in the container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "dry-run" -d "Only print the object that would be sent"
complete -c kubectl -f -n "__kubectl_using_command run" -l "env" -d "Environment variables to set in the container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "expose" -d "A public, external service is created for the container(s) which are run"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_option --expose" -l "service-generator" -d "The name of the generator to use for creating a service"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_option --expose" -l "service-overrides" -d "An inline JSON override for the generated service object"
complete -c kubectl -f -n "__kubectl_using_command run" -l "generator" -d "The name of the API generator to use"
complete -c kubectl -f -n "__kubectl_using_command run" -l "hostport" -d "The host port mapping for the container port"
complete -c kubectl -f -n "__kubectl_using_command run" -l "image" -d "The image for the container to run"
complete -c kubectl -f -n "__kubectl_using_command run" -l "image-pull-policy" -d "The image pull policy for the container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command run" -s l -l "labels" -d "Labels to apply to the pod(s)"
complete -c kubectl -f -n "__kubectl_using_command run" -l "leave-stdin-open" -d "Leave stdin open after the first attach completes"
complete -c kubectl -f -n "__kubectl_using_command run" -l "limits" -d "The resource requirement limits for this container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "no-headers" -d "Don't print headers"
complete -c kubectl -f -n "__kubectl_using_command run" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -f -n "__kubectl_using_command run" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -f -n "__kubectl_using_command run" -l "overrides" -d "An inline JSON override for the generated object"
complete -c kubectl -f -n "__kubectl_using_command run" -l "port" -d "The port that this container exposes"
complete -c kubectl -f -n "__kubectl_using_command run" -l "quiet" -d "Suppress prompt messages"
complete -c kubectl -f -n "__kubectl_using_command run" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -f -n "__kubectl_using_command run" -s r -l "replicas" -d "Number of replicas to create for this container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "requests" -d "The resource requirement requests for this container"
complete -c kubectl -f -n "__kubectl_using_command run" -l "restart" -d "The restart policy for this Pod"
complete -c kubectl -f -n "__kubectl_using_command run; and __kubectl_using_option --restart" -a "Always OnFailure Never" -d "Policy"
complete -c kubectl -f -n "__kubectl_using_command run" -l "rm" -d "Delete resources created in this command for attached containers"
complete -c kubectl -f -n "__kubectl_using_command run" -l "save-config" -d "The config of current object will be saved in its annotation"
complete -c kubectl -f -n "__kubectl_using_command run" -l "schedule" -d "A schedule in the Cron format the job should be run with"
complete -c kubectl -f -n "__kubectl_using_command run" -s a -l "show-all" -d "Show all resources"
complete -c kubectl -f -n "__kubectl_using_command run" -l "show-labels" -d "Show all labels as the last column"
complete -c kubectl -f -n "__kubectl_using_command run" -l "sort-by" -d "Sort list types using this field specification"
complete -c kubectl -f -n "__kubectl_using_command run" -s i -l "stdin" -d "Keep stdin open on the container(s) in the pod"
complete -c kubectl -f -n "__kubectl_using_command run" -s t -l "tty" -d "Allocated a TTY for each container in the pod"

## set
complete -c kubectl -f -n "__kubectl_no_command" -a set -d "Set specific features on objects"
complete -c kubectl -f -n "__kubectl_using_command set" -a "(__kubectl_subcommands set)"

### set image
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -a "pod replicationcontroller deployment daemonset job replicaset" -d "Resource Type"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -a "(__kubectl_resources pod replicationcontroller deployment daemonset job replicaset)" -d "Resource"

complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "all" -d "Select all resources in the namespace of the specified resource types"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from image; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "local" -d "Set image will NOT contact api-server but run locally"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "no-headers" -d "Don't print headers"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from image; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from image; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -s l -l "selector" -d "Selector (label query) to filter on"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -s a -l "show-all" -d "Show all resources"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "show-labels" -d "Show all labels as the last column"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from image" -l "sort-by" -d "Sort list types using this field specification"

### set resources
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -a "replicationcontroller deployment daemonset job replicaset" -d "Resource Type"

complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "all" -d "Select all resources in the namespace of the specified resource types"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -s c -l "containers" -d "The names of containers in the selected pod templates to change"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "dry-run" -d "Only print the object that would be sent"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "limits" -d "The resource requirement requests for this container"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "local" -d "Set resources will NOT contact api-server but run locally"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "no-headers" -d "Don't print headers"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -n "__kubectl_using_command set; and __fish_seen_subcommand_from resources" -l "requests" -d "The resource requirement requests for this container"

# Basic Commands (Intermediate):

## get
complete -c kubectl -f -n "__kubectl_no_command" -a get -d "Display one or many resources"

complete -c kubectl -f -n "__kubectl_using_command get" -l "all-namespaces" -d "List the requested object(s) across all namespaces"
complete -c kubectl -f -n "__kubectl_using_command get" -l "export" -d "Strip cluster-specific info"
complete -c kubectl -f -n "__kubectl_using_command get" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command get; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -f -n "__kubectl_using_command get" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command get" -s L -l "label-columns" -d "List of labels to be presented as columns"
complete -c kubectl -f -n "__kubectl_using_command get" -l "no-headers" -d "Don't print headers"
complete -c kubectl -n "__kubectl_using_command get" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command get; and __kubectl_using_option -o --output" -a "(__kubectl_output_formats)" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command get; and __kubectl_using_go_template_format" -l "template" -d "Template string or path to template file to use"
complete -c kubectl -n "__kubectl_using_command get" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -n "__kubectl_using_command get" -l "raw" -d "Raw URI to request from the server"
complete -c kubectl -n "__kubectl_using_command get" -s l -l "selector" -d "Selector (label query) to filter on"
complete -c kubectl -n "__kubectl_using_command get" -s a -l "show-all" -d "Show all resources"
complete -c kubectl -n "__kubectl_using_command get" -l "show-kind" -d "List the resource type for the requested object(s)"
complete -c kubectl -n "__kubectl_using_command get" -l "show-labels" -d "Show all labels as the last column"
complete -c kubectl -n "__kubectl_using_command get" -l "sort-by" -d "Sort list types using this field specification"
complete -c kubectl -n "__kubectl_using_command get" -s w -l "watch" -d "Watch for changes after listing/getting"
complete -c kubectl -n "__kubectl_using_command get" -l "watch-only" -d "Watch for changes to the requested object(s)"

complete -c kubectl -f -n "__kubectl_using_command get; and not __kubectl_using_resource_type" -a "(__kubectl_resource_types)" -d "Resource Type"

# Generate completions for all resource sub-commands
__kubectl_generate_get_resource_subcommand_completions

complete -c kubectl -f -n "__kubectl_using_command get; and not __kubectl_using_resource_type" -a "(__kubectl_get_prefixed_resource_completions)"

## explain
complete -c kubectl -f -n "__kubectl_no_command" -a explain -d "Documentation of resources"

complete -c kubectl -f -n "__kubectl_using_command explain" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command explain" -l "recursive" -d "Print the fields of fields"

complete -c kubectl -f -n "__kubectl_using_command explain; and not __kubectl_using_resource_type" -a "(__kubectl_resource_types)" -d "Resource Type"

## edit
complete -c kubectl -f -n "__kubectl_no_command" -a edit -d "Edit a resource on the server"

complete -c kubectl -n "__kubectl_using_command edit" -s f -l "filename" -d "Filename, directory, or URL to files"
complete -c kubectl -f -n "__kubectl_using_command edit; and __kubectl_using_option -f --filename" -s R -l "recursive" -d "Process the directory used in -f, --filename recursively"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "include-extended-apis" -d "Include definitions of new APIs [default true]"
complete -c kubectl -f -n "__kubectl_using_command edit" -s o -l "output" -d "Output format"
complete -c kubectl -f -n "__kubectl_using_command edit; __kubectl_using_option -o --output" -a "yaml json" -d "Format"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "output-version" -d "Format object with the given group version"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "record" -d "Record current kubectl command in the resource annotation"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "save-config" -d "The config of current object will be saved in its annotation"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "schema-cache-dir" -d "Load/store cached API schemas in this directory"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "validate" -d "Use a schema to validate the input before sending"
complete -c kubectl -f -n "__kubectl_using_command edit" -l "windows-line-endings" -d "Use Windows line-endings"

## delete
complete -c kubectl -f -n "__kubectl_no_command" -a delete -d "Delete resources"

# Deploy Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a rollout -d "Manage a deployment rollout"
complete -c kubectl -f -n "__kubectl_no_command" -a rolling-update -d "Perform a rolling update of the given ReplicationController"
complete -c kubectl -f -n "__kubectl_no_command" -a scale -d "Set a new size for a Deployment, ReplicaSet, Replication Controller, or Job"
complete -c kubectl -f -n "__kubectl_no_command" -a autoscale -d "Auto-scale a Deployment, ReplicaSet, or ReplicationController"

# Cluster Management Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a certificate -d "Modify certificate resources."
complete -c kubectl -f -n "__kubectl_no_command" -a cluster-info -d "Display cluster info"
complete -c kubectl -f -n "__kubectl_no_command" -a top -d "Display Resource (CPU/Memory/Storage) usage"
complete -c kubectl -f -n "__kubectl_no_command" -a cordon -d "Mark node as unschedulable"
complete -c kubectl -f -n "__kubectl_no_command" -a uncordon -d "Mark node as schedulable"
complete -c kubectl -f -n "__kubectl_no_command" -a drain -d "Drain node in preparation for maintenance"
complete -c kubectl -f -n "__kubectl_no_command" -a taint -d "Update the taints on one or more nodes"

# Troubleshooting and Debugging Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a describe -d "Show details of a specific resource or group of resources"
complete -c kubectl -f -n "__kubectl_no_command" -a logs -d "Print the logs for a container in a pod"
complete -c kubectl -f -n "__kubectl_no_command" -a attach -d "Attach to a running container"
complete -c kubectl -f -n "__kubectl_no_command" -a exec -d "Execute a command in a container"
complete -c kubectl -f -n "__kubectl_no_command" -a port-forward -d "Forward one or more local ports to a pod"
complete -c kubectl -f -n "__kubectl_no_command" -a proxy -d "Run a proxy to the Kubernetes API server"
complete -c kubectl -f -n "__kubectl_no_command" -a cp -d "Copy files and directories to and from containers."

# Advanced Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a apply -d "Apply a configuration to a resource by filename or stdin"
complete -c kubectl -f -n "__kubectl_no_command" -a patch -d "Update field(s) of a resource using strategic merge patch"
complete -c kubectl -f -n "__kubectl_no_command" -a replace -d "Replace a resource by filename or stdin"
complete -c kubectl -f -n "__kubectl_no_command" -a convert -d "Convert config files between different API versions"

# Settings Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a label -d "Update the labels on a resource"
complete -c kubectl -f -n "__kubectl_no_command" -a annotate -d "Update the annotations on a resource"
complete -c kubectl -f -n "__kubectl_no_command" -a completion -d "Output shell completion code for the given SHELL"

# Other Commands:
complete -c kubectl -f -n "__kubectl_no_command" -a api-versions -d "Print the supported API versions on the server"
complete -c kubectl -f -n "__kubectl_no_command" -a config -d "Modify kubeconfig files"
complete -c kubectl -f -n "__kubectl_no_command" -a help -d "Help about any command"
complete -c kubectl -f -n "__kubectl_no_command" -a version -d "Print the client and server version information"
complete -c kubectl -f -n "__kubectl_no_command" -a options -d "Print the shared options"

# Attach
complete -c kubectl -n "__kubectl_using_command attach" -a "(__kubectl_pods_completion)"

# Exec
complete -c kubectl -n "__kubectl_using_command exec" -a "(__kubectl_pods_completion)"

# Describe
complete -c kubectl -n "__kubectl_using_command describe" -a "(__kubectl_resource_types)" -d "Resource Type"
complete -c kubectl -n "__kubectl_using_command describe" -a "(__kubectl_pods_completion)"

# Logs
complete -f -c kubectl -n "__kubectl_using_command logs; and __kubectl_no_pod" -a "(__kubectl_pods_completion)"
complete -f -c kubectl -n "__kubectl_using_command logs" -a "(__kubectl_containers (commandline -c))" -d "Container"`

	out.Write([]byte(fish_body))
	return nil
}
