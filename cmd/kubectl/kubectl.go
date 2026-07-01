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

package main

import (
	"os"

	"k8s.io/component-base/cli"
	"k8s.io/component-base/logs"
	"k8s.io/kubectl/pkg/cmd"
	"k8s.io/kubectl/pkg/cmd/util"

	// Import to initialize client auth plugins.
	_ "k8s.io/client-go/plugin/pkg/client/auth"
)

// main 是 kubectl 二进制的最外层入口，只负责完成进程级初始化并把实际命令树交给
// staging/src/k8s.io/kubectl/pkg/cmd 构建和执行。
//
// 功能分析：
//  1. 先从原始 os.Args 中提前解析日志 verbosity。kubectl 正常的 flag 解析发生在
//     cli.RunNoErrOutput 内部，但命令树构建阶段就可能需要输出 klog，例如插件发现、
//     .kuberc 解析或命令初始化失败路径，所以这里必须在构建命令前设置日志级别。
//  2. 调用 cmd.NewDefaultKubectlCommand 创建完整 kubectl Cobra 根命令。具体子命令、
//     kubeconfig flags、插件处理、completion 和 warning 处理都在 staging 的 kubectl
//     命令包中完成。
//  3. 使用 cli.RunNoErrOutput 执行命令，并把错误交给 util.CheckErr 统一格式化。
//
// 注意点：
//  1. 这里不要直接添加业务逻辑。kubectl 的命令行为应放在
//     staging/src/k8s.io/kubectl/pkg/cmd 或 cli-runtime 中，保持 cmd/kubectl 只是薄入口。
//  2. client-go auth 插件通过 blank import 注册；删除该 import 会影响云厂商 kubeconfig
//     exec/auth-provider 相关认证能力。
//  3. util.CheckErr 会在错误时退出进程，因此 main 不需要再显式 os.Exit(1)。
func main() {
	// We need to manually parse the arguments looking for verbosity flag and
	// set appropriate level here, because in the normal flow the flag parsing,
	// including the logging verbosity, happens inside cli.RunNoErrOutput.
	// Doing it here ensures we can continue using klog during kubectl command
	// construction, which includes handling plugins and parsing .kuberc file,
	// for example.
	logs.GlogSetter(cmd.GetLogVerbosity(os.Args)) // nolint:errcheck
	command := cmd.NewDefaultKubectlCommand()
	if err := cli.RunNoErrOutput(command); err != nil {
		// Pretty-print the error and exit with an error.
		util.CheckErr(err)
	}
}
