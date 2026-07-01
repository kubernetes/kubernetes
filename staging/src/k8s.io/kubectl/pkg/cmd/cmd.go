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
	"net/http"
	"os"
	"strings"
	"sync/atomic"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/klog/v2"
	"k8s.io/kubectl/pkg/cmd/annotate"
	"k8s.io/kubectl/pkg/cmd/apiresources"
	"k8s.io/kubectl/pkg/cmd/apply"
	"k8s.io/kubectl/pkg/cmd/attach"
	"k8s.io/kubectl/pkg/cmd/auth"
	"k8s.io/kubectl/pkg/cmd/autoscale"
	"k8s.io/kubectl/pkg/cmd/certificates"
	"k8s.io/kubectl/pkg/cmd/clusterinfo"
	"k8s.io/kubectl/pkg/cmd/completion"
	cmdconfig "k8s.io/kubectl/pkg/cmd/config"
	"k8s.io/kubectl/pkg/cmd/cp"
	"k8s.io/kubectl/pkg/cmd/create"
	"k8s.io/kubectl/pkg/cmd/debug"
	"k8s.io/kubectl/pkg/cmd/delete"
	"k8s.io/kubectl/pkg/cmd/describe"
	"k8s.io/kubectl/pkg/cmd/diff"
	"k8s.io/kubectl/pkg/cmd/drain"
	"k8s.io/kubectl/pkg/cmd/edit"
	"k8s.io/kubectl/pkg/cmd/events"
	cmdexec "k8s.io/kubectl/pkg/cmd/exec"
	"k8s.io/kubectl/pkg/cmd/explain"
	"k8s.io/kubectl/pkg/cmd/expose"
	"k8s.io/kubectl/pkg/cmd/get"
	kuberccmd "k8s.io/kubectl/pkg/cmd/kuberc"
	"k8s.io/kubectl/pkg/cmd/kustomize"
	"k8s.io/kubectl/pkg/cmd/label"
	"k8s.io/kubectl/pkg/cmd/logs"
	"k8s.io/kubectl/pkg/cmd/options"
	"k8s.io/kubectl/pkg/cmd/patch"
	"k8s.io/kubectl/pkg/cmd/plugin"
	"k8s.io/kubectl/pkg/cmd/portforward"
	"k8s.io/kubectl/pkg/cmd/proxy"
	"k8s.io/kubectl/pkg/cmd/replace"
	"k8s.io/kubectl/pkg/cmd/rollout"
	"k8s.io/kubectl/pkg/cmd/run"
	"k8s.io/kubectl/pkg/cmd/scale"
	"k8s.io/kubectl/pkg/cmd/set"
	"k8s.io/kubectl/pkg/cmd/taint"
	"k8s.io/kubectl/pkg/cmd/top"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/cmd/version"
	"k8s.io/kubectl/pkg/cmd/wait"
	"k8s.io/kubectl/pkg/kuberc"
	utilcomp "k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type KubectlOptions struct {
	PluginHandler PluginHandler
	Arguments     []string
	ConfigFlags   *genericclioptions.ConfigFlags

	genericiooptions.IOStreams
}

// defaultConfigFlags 构造 kubectl 根命令默认使用的 kubeconfig/client 配置 flags。
//
// 功能分析：
//  1. NewConfigFlags(true) 会启用一组标准 kubeconfig flags，例如 --kubeconfig、
//     --context、--namespace、--cluster 和 --user。
//  2. WithDeprecatedPasswordFlag 保留历史 password flag 的兼容入口，避免旧脚本在升级
//     kubectl 后直接失效。
//  3. Discovery QPS/Burst 在这里被调高，因为 kubectl 命令构建和资源补全经常需要访问
//     discovery；过低的默认值会让大型集群中的 get/apply/explain 等命令体验变差。
//
// 注意点：
//  1. 这里返回的是根命令共享配置，改动会影响几乎所有 kubectl 子命令。
//  2. 如果某个子命令需要特殊 REST config 行为，应优先在该子命令或 Factory 层定制，
//     不要轻易改变全局默认值。
func defaultConfigFlags() *genericclioptions.ConfigFlags {
	return genericclioptions.NewConfigFlags(true).WithDeprecatedPasswordFlag().WithDiscoveryBurst(300).WithDiscoveryQPS(50.0)
}

// NewDefaultKubectlCommand creates the `kubectl` command with default arguments.
//
// 功能分析：
//  1. 为真实 kubectl 进程绑定标准输入、标准输出和标准错误输出。
//  2. 创建默认插件处理器，用于把未知命令转发给 PATH 中的 kubectl-* 插件。
//  3. 使用 os.Args 作为命令行来源，并把 warning printer 挂到默认 kubeconfig flags 上。
//
// 注意点：
//  1. 这是生产二进制使用的便捷入口；测试或嵌入式调用通常应使用
//     NewDefaultKubectlCommandWithArgs 注入自定义参数和 IOStreams。
//  2. 这里不执行命令，只创建命令树；实际执行由调用者交给 component-base/cli。
func NewDefaultKubectlCommand() *cobra.Command {
	ioStreams := genericiooptions.IOStreams{In: os.Stdin, Out: os.Stdout, ErrOut: os.Stderr}
	return NewDefaultKubectlCommandWithArgs(KubectlOptions{
		PluginHandler: NewDefaultPluginHandler(plugin.ValidPluginFilenamePrefixes),
		Arguments:     os.Args,
		ConfigFlags:   defaultConfigFlags().WithWarningPrinter(ioStreams),
		IOStreams:     ioStreams,
	})
}

// NewDefaultKubectlCommandWithArgs creates the `kubectl` command with arguments.
//
// 功能分析：
//  1. 先通过 NewKubectlCommand 创建内置 kubectl 命令树。
//  2. 如果用户输入的命令不是内置命令，则按 kubectl 插件规则查找 PATH 中的插件可执行文件。
//  3. 对少数允许插件子命令扩展的内置命令，例如 create，会在内置子命令不存在时继续尝试
//     查找更长的插件名。
//
// 注意点：
//  1. 插件分发发生在 Cobra Execute 前，因此 Cobra 自动注入的 help/completion 命令需要
//     在这里特殊排除，避免被误认为插件。
//  2. 找到插件后 HandlePluginCommand 可能直接替换当前进程或退出进程；调用者不能假设
//     该函数总是普通返回。
//  3. minArgs 的计算决定是否允许回退到更短插件名，错误修改可能导致
//     kubectl create foo 意外执行 kubectl-create。
func NewDefaultKubectlCommandWithArgs(o KubectlOptions) *cobra.Command {
	cmd := NewKubectlCommand(o)

	if o.PluginHandler == nil || len(o.Arguments) <= 1 {
		return cmd
	}

	cmdPathPieces := o.Arguments[1:]
	// only look for suitable extension executables if
	// the specified command does not already exist
	foundCmd, foundArgs, err := cmd.Find(cmdPathPieces)
	if err != nil {
		// Also check the commands that will be added by Cobra.
		// These commands are only added once rootCmd.Execute() is called, so we
		// need to check them explicitly here.
		var cmdName string // first "non-flag" arguments
		for _, arg := range cmdPathPieces {
			if !strings.HasPrefix(arg, "-") {
				cmdName = arg
				break
			}
		}

		switch cmdName {
		case "help", cobra.ShellCompRequestCmd, cobra.ShellCompNoDescRequestCmd:
			// Don't search for a plugin
		default:
			if err := HandlePluginCommand(o.PluginHandler, cmdPathPieces, 1); err != nil {
				fmt.Fprintf(o.IOStreams.ErrOut, "Error: %v\n", err)
				os.Exit(1)
			}
		}
	}
	// Command exists(e.g. kubectl create), but it is not certain that
	// subcommand also exists (e.g. kubectl create networkpolicy)
	// we also have to eliminate kubectl create -f
	if IsSubcommandPluginAllowed(foundCmd.Name()) && len(foundArgs) >= 1 && !strings.HasPrefix(foundArgs[0], "-") {
		subcommand := foundArgs[0]
		builtinSubcmdExist := false
		for _, subcmd := range foundCmd.Commands() {
			if subcmd.Name() == subcommand {
				builtinSubcmdExist = true
				break
			}
		}

		if !builtinSubcmdExist {
			if err := HandlePluginCommand(o.PluginHandler, cmdPathPieces, len(cmdPathPieces)-len(foundArgs)+1); err != nil {
				fmt.Fprintf(o.IOStreams.ErrOut, "Error: %v\n", err)
				os.Exit(1)
			}
		}
	}

	return cmd
}

// NewKubectlCommand creates the `kubectl` command and its nested children.
//
// 功能分析：
//  1. 创建 kubectl Cobra 根命令，并注册全局 persistent flags、warning 处理、profiling、
//     kubeconfig flags、kuberc 偏好和命令头注入 hooks。
//  2. 构造 cmdutil.Factory。绝大多数子命令通过这个 Factory 获取 RESTMapper、client、
//     discovery、builder、printer 和 validator 等依赖。
//  3. 按帮助文档中的命令组装载 create/get/apply/logs/exec/proxy 等内置子命令，并注册
//     completion、插件帮助分组、alpha/config/plugin/version/api-resources/options 等命令。
//  4. 在最终返回前应用 kuberc 偏好，允许用户偏好影响命令默认行为。
//
// 注意点：
//  1. 这是 kubectl 命令树的中心 wiring 函数，新增全局行为时要评估所有子命令影响。
//  2. PersistentPreRunE/PersistentPostRunE 已被 warning、profiling、completion、kuberc
//     和命令头逻辑复用；新增 hook 时必须串联 existing hook，不能直接覆盖。
//  3. proxy 命令对命令头 RoundTripper 不兼容，因此这里用 isProxyCmd 在执行期跳过注入。
//  4. 新增子命令时要同时考虑帮助分组、completion、kuberc 偏好和测试中的命令树断言。
func NewKubectlCommand(o KubectlOptions) *cobra.Command {
	warningHandler := rest.NewWarningWriter(o.IOStreams.ErrOut, rest.WarningWriterOptions{Deduplicate: true, Color: printers.AllowsColorOutput(o.IOStreams.ErrOut)})
	warningsAsErrors := false
	var finishProfiling func() error
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "kubectl",
		Short: i18n.T("kubectl controls the Kubernetes cluster manager"),
		Long: templates.LongDesc(`
      kubectl controls the Kubernetes cluster manager.

      Find more information at:
            https://kubernetes.io/docs/reference/kubectl/`),
		Run: runHelp,
		// Hook before and after Run initialize and write profiles to disk,
		// respectively.
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			rest.SetDefaultWarningHandler(warningHandler)

			if cmd.Name() == cobra.ShellCompRequestCmd {
				// This is the __complete or __completeNoDesc command which
				// indicates shell completion has been requested.
				plugin.SetupPluginCompletion(cmd, args)
			}

			var err error
			finishProfiling, err = initProfiling()
			return err
		},
		PersistentPostRunE: func(*cobra.Command, []string) error {
			if finishProfiling != nil {
				if err := finishProfiling(); err != nil {
					return err
				}
			}
			if warningsAsErrors {
				count := warningHandler.WarningCount()
				switch count {
				case 0:
					// no warnings
				case 1:
					return fmt.Errorf("%d warning received", count)
				default:
					return fmt.Errorf("%d warnings received", count)
				}
			}
			return nil
		},
	}
	// From this point and forward we get warnings on flags that contain "_" separators
	// when adding them with hyphen instead of the original name.
	cmds.SetGlobalNormalizationFunc(cliflag.WarnWordSepNormalizeFunc)

	flags := cmds.PersistentFlags()

	addProfilingFlags(flags)

	flags.BoolVar(&warningsAsErrors, "warnings-as-errors", warningsAsErrors, "Treat warnings received from the server as errors and exit with a non-zero exit code")

	pref := kuberc.NewPreferences()
	if !cmdutil.KubeRC.IsDisabled() {
		pref.AddFlags(flags)
	}

	kubeConfigFlags := o.ConfigFlags
	if kubeConfigFlags == nil {
		kubeConfigFlags = defaultConfigFlags().WithWarningPrinter(o.IOStreams)
	}
	kubeConfigFlags.AddFlags(flags)
	matchVersionKubeConfigFlags := cmdutil.NewMatchVersionFlags(kubeConfigFlags)
	matchVersionKubeConfigFlags.AddFlags(flags)
	// Updates hooks to add kubectl command headers: SIG CLI KEP 859.
	var isProxyCmd atomic.Bool
	addCmdHeaderHooks(cmds, kubeConfigFlags, &isProxyCmd)

	f := cmdutil.NewFactory(matchVersionKubeConfigFlags)

	// Proxy command is incompatible with the headers set by
	// CommandHeaderRoundTripper, so the RoundTripper hooks set in
	// `addCmdHeaderHooks` needs to be aware that the subcommand is `proxy`
	proxyCmd := proxy.NewCmdProxy(f, o.IOStreams)
	proxyCmd.PreRun = func(cmd *cobra.Command, args []string) {
		isProxyCmd.Store(true)
	}

	// Avoid import cycle by setting ValidArgsFunction here instead of in NewCmdGet()
	getCmd := get.NewCmdGet("kubectl", f, o.IOStreams)
	getCmd.ValidArgsFunction = utilcomp.ResourceTypeAndNameCompletionFunc(f)
	debugCmd := debug.NewCmdDebug(f, o.IOStreams)
	debugCmd.ValidArgsFunction = utilcomp.ResourceTypeAndNameCompletionFunc(f)

	groups := templates.CommandGroups{
		{
			Message: "Basic Commands (Beginner):",
			Commands: []*cobra.Command{
				create.NewCmdCreate(f, o.IOStreams),
				expose.NewCmdExposeService(f, o.IOStreams),
				run.NewCmdRun(f, o.IOStreams),
				set.NewCmdSet(f, o.IOStreams),
			},
		},
		{
			Message: "Basic Commands (Intermediate):",
			Commands: []*cobra.Command{
				explain.NewCmdExplain("kubectl", f, o.IOStreams),
				getCmd,
				edit.NewCmdEdit(f, o.IOStreams),
				delete.NewCmdDelete(f, o.IOStreams),
			},
		},
		{
			Message: "Deploy Commands:",
			Commands: []*cobra.Command{
				rollout.NewCmdRollout("kubectl", f, o.IOStreams),
				scale.NewCmdScale(f, o.IOStreams),
				autoscale.NewCmdAutoscale(f, o.IOStreams),
			},
		},
		{
			Message: "Cluster Management Commands:",
			Commands: []*cobra.Command{
				certificates.NewCmdCertificate(f, o.IOStreams),
				clusterinfo.NewCmdClusterInfo(f, o.IOStreams),
				top.NewCmdTop(f, o.IOStreams),
				drain.NewCmdCordon(f, o.IOStreams),
				drain.NewCmdUncordon(f, o.IOStreams),
				drain.NewCmdDrain(f, o.IOStreams),
				taint.NewCmdTaint(f, o.IOStreams),
			},
		},
		{
			Message: "Troubleshooting and Debugging Commands:",
			Commands: []*cobra.Command{
				describe.NewCmdDescribe("kubectl", f, o.IOStreams),
				logs.NewCmdLogs(f, o.IOStreams),
				attach.NewCmdAttach(f, o.IOStreams),
				cmdexec.NewCmdExec(f, o.IOStreams),
				portforward.NewCmdPortForward(f, o.IOStreams),
				proxyCmd,
				cp.NewCmdCp(f, o.IOStreams),
				auth.NewCmdAuth(f, o.IOStreams),
				debugCmd,
				events.NewCmdEvents(f, o.IOStreams),
			},
		},
		{
			Message: "Advanced Commands:",
			Commands: []*cobra.Command{
				diff.NewCmdDiff(f, o.IOStreams),
				apply.NewCmdApply("kubectl", f, o.IOStreams),
				patch.NewCmdPatch(f, o.IOStreams),
				replace.NewCmdReplace(f, o.IOStreams),
				wait.NewCmdWait(f, o.IOStreams),
				kustomize.NewCmdKustomize(o.IOStreams),
			},
		},
		{
			Message: "Settings Commands:",
			Commands: []*cobra.Command{
				label.NewCmdLabel(f, o.IOStreams),
				annotate.NewCmdAnnotate("kubectl", f, o.IOStreams),
				completion.NewCmdCompletion(o.IOStreams.Out, ""),
			},
		},
	}
	groups.Add(cmds)

	filters := []string{"options"}

	// Hide the "alpha" subcommand if there are no alpha commands in this build.
	alpha := NewCmdAlpha(f, o.IOStreams)
	if !alpha.HasSubCommands() {
		filters = append(filters, alpha.Name())
	}

	// Add plugin command group to the list of command groups.
	// The commands are only injected for the scope of showing help and completion, they are not
	// invoked directly.
	pluginCommandGroup := plugin.GetPluginCommandGroup(cmds)
	groups = append(groups, pluginCommandGroup)

	templates.ActsAsRootCommand(cmds, filters, groups...)

	utilcomp.SetFactoryForCompletion(f)
	registerCompletionFuncForGlobalFlags(cmds, f)

	cmds.AddCommand(alpha)
	cmds.AddCommand(cmdconfig.NewCmdConfig(f, clientcmd.NewDefaultPathOptions(), o.IOStreams))
	cmds.AddCommand(plugin.NewCmdPlugin(o.IOStreams))
	cmds.AddCommand(version.NewCmdVersion(f, o.IOStreams))
	cmds.AddCommand(apiresources.NewCmdAPIVersions(f, o.IOStreams))
	cmds.AddCommand(apiresources.NewCmdAPIResources(f, o.IOStreams))
	cmds.AddCommand(options.NewCmdOptions(o.IOStreams.Out))
	if !cmdutil.KubeRC.IsDisabled() {
		cmds.AddCommand(kuberccmd.NewCmdKubeRC(o.IOStreams))
	}

	// Stop warning about normalization of flags. That makes it possible to
	// add the klog flags later.
	cmds.SetGlobalNormalizationFunc(cliflag.WordSepNormalizeFunc)

	if !cmdutil.KubeRC.IsDisabled() {
		existingPreRunE := cmds.PersistentPreRunE
		cmds.PersistentPreRunE = func(cmd *cobra.Command, args []string) error {
			if originalCommandArgs, ok := cmd.Annotations[kuberc.KubeRCOriginalCommandAnnotation]; ok {
				originalCommand := fmt.Sprintf("%s %s", cmd.Root().Name(), originalCommandArgs)
				klog.V(1).Info(fmt.Sprintf("original command: %q", originalCommand))
			}
			return existingPreRunE(cmd, args)
		}
		_, err := pref.Apply(cmds, kubeConfigFlags, o.Arguments, o.IOStreams.ErrOut)
		if err != nil {
			fmt.Fprintf(o.IOStreams.ErrOut, "error occurred while applying preferences %v\n", err)
			os.Exit(1)
		}
	}

	return cmds
}

// addCmdHeaderHooks performs updates on two hooks:
//  1. Modifies the passed "cmds" persistent pre-run function to parse command headers.
//     These headers will be subsequently added as X-headers to every
//     REST call.
//  2. Adds CommandHeaderRoundTripper as a wrapper around the standard
//     RoundTripper. CommandHeaderRoundTripper adds X-Headers then delegates
//     to standard RoundTripper.
//
// See SIG CLI KEP 859 for more information:
//
//	https://github.com/kubernetes/enhancements/tree/master/keps/sig-cli/859-kubectl-headers
func addCmdHeaderHooks(cmds *cobra.Command, kubeConfigFlags *genericclioptions.ConfigFlags, isProxyCmd *atomic.Bool) {
	crt := &genericclioptions.CommandHeaderRoundTripper{}
	existingPreRunE := cmds.PersistentPreRunE
	// Add command parsing to the existing persistent pre-run function.
	cmds.PersistentPreRunE = func(cmd *cobra.Command, args []string) error {
		crt.ParseCommandHeaders(cmd, args)
		return existingPreRunE(cmd, args)
	}
	wrapConfigFn := kubeConfigFlags.WrapConfigFn
	// Wraps CommandHeaderRoundTripper around standard RoundTripper.
	kubeConfigFlags.WithWrapConfigFn(func(c *rest.Config) *rest.Config {
		if wrapConfigFn != nil {
			c = wrapConfigFn(c)
		}
		c.Wrap(func(rt http.RoundTripper) http.RoundTripper {
			// Must be separate RoundTripper; not "crt" closure.
			// Fixes: https://github.com/kubernetes/kubectl/issues/1098
			return &genericclioptions.CommandHeaderRoundTripper{
				Delegate:    rt,
				Headers:     crt.Headers,
				SkipHeaders: isProxyCmd, // proxy command is incompatible with these headers
			}
		})
		return c
	})
}

// runHelp 是 kubectl 根命令没有匹配到具体子命令时的默认执行函数。
//
// 功能分析：
//  1. 直接输出当前命令的帮助信息，保持裸运行 kubectl 时展示使用说明。
//  2. 不返回错误，避免把“只输入 kubectl”当成命令执行失败。
//
// 注意点：这里忽略 Help 的返回值沿用了既有行为；如果未来需要严格处理输出错误，
// 需要同步评估命令行兼容性和测试期望。
func runHelp(cmd *cobra.Command, args []string) {
	cmd.Help()
}

// registerCompletionFuncForGlobalFlags 为 kubectl 全局 kubeconfig flags 注册 shell completion。
//
// 功能分析：
//  1. --namespace 通过实时资源补全列出 namespace。
//  2. --context、--cluster、--user 从 kubeconfig 中读取候选项，避免用户手工记忆配置名。
//
// 注意点：
//  1. 这些 completion 依赖 Factory 或本地 kubeconfig，执行时应避免产生破坏性副作用。
//  2. RegisterFlagCompletionFunc 的错误通过 CheckErr 处理；注册失败会直接终止命令构建。
func registerCompletionFuncForGlobalFlags(cmd *cobra.Command, f cmdutil.Factory) {
	cmdutil.CheckErr(cmd.RegisterFlagCompletionFunc(
		"namespace",
		func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
			return utilcomp.CompGetResource(f, "namespace", toComplete), cobra.ShellCompDirectiveNoFileComp
		}))
	cmdutil.CheckErr(cmd.RegisterFlagCompletionFunc(
		"context",
		func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
			return utilcomp.ListContextsInConfig(toComplete), cobra.ShellCompDirectiveNoFileComp
		}))
	cmdutil.CheckErr(cmd.RegisterFlagCompletionFunc(
		"cluster",
		func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
			return utilcomp.ListClustersInConfig(toComplete), cobra.ShellCompDirectiveNoFileComp
		}))
	cmdutil.CheckErr(cmd.RegisterFlagCompletionFunc(
		"user",
		func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
			return utilcomp.ListUsersInConfig(toComplete), cobra.ShellCompDirectiveNoFileComp
		}))
}

// GetLogVerbosity parses the provided command-line arguments to determine
// the verbosity level for logging. Returns string representing the verbosity
// level, or 0 if no verbosity flag is specified.
//
// 功能分析：
//  1. 支持 "-v 6"、"--v 6"、"-v=6" 和 "--v=6" 这几种 klog verbosity 写法。
//  2. 遇到 "--" 后立即停止解析，因为后续参数属于子命令或远程命令，不应再解释为
//     kubectl 自身的日志 flags。
//
// 注意点：
//  1. 这是命令正式 flag 解析前的轻量扫描，只用于尽早设置日志级别，不做完整校验。
//  2. 未找到 verbosity 时返回 "0"，与 klog 默认等级保持一致。
func GetLogVerbosity(args []string) string {
	for i, arg := range args {
		if arg == "--" {
			// flags after "--" does not represent any flag of
			// the command. We should short cut the iteration in here.
			break
		}

		if arg == "--v" || arg == "-v" {
			if i+1 < len(args) {
				return args[i+1]
			}
		} else if strings.Contains(arg, "--v=") || strings.Contains(arg, "-v=") {
			parg := strings.Split(arg, "=")
			if len(parg) > 1 && parg[1] != "" {
				return parg[1]
			}
		}
	}

	return "0"
}
