/*
Copyright The Kubernetes Authors.

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
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
)

// PluginHandler is capable of parsing command line arguments
// and performing executable filename lookups to search
// for valid plugin files, and execute found plugins.
type PluginHandler interface {
	// exists at the given filename, or a boolean false.
	// Lookup will iterate over a list of given prefixes
	// in order to recognize valid plugin filenames.
	// The first filepath to match a prefix is returned.
	Lookup(filename string) (string, bool)
	// Execute receives an executable's filepath, a slice
	// of arguments, and a slice of environment variables
	// to relay to the executable.
	Execute(executablePath string, cmdArgs, environment []string) error
}

// DefaultPluginHandler implements PluginHandler
type DefaultPluginHandler struct {
	ValidPrefixes []string
}

// NewDefaultPluginHandler instantiates the DefaultPluginHandler with a list of
// given filename prefixes used to identify valid plugin filenames.
//
// 功能分析：
//  1. ValidPrefixes 通常包含 kubectl 插件允许的文件名前缀，例如 kubectl。
//  2. 后续 Lookup 会把前缀和用户输入的命令片段拼成可执行文件名，并在 PATH 中查找。
//
// 注意点：插件命名策略集中在调用方传入的 validPrefixes 中；这里不校验前缀内容，
// 因此新增前缀时应同时评估安全边界和 PATH 查找行为。
func NewDefaultPluginHandler(validPrefixes []string) *DefaultPluginHandler {
	return &DefaultPluginHandler{
		ValidPrefixes: validPrefixes,
	}
}

// Lookup implements PluginHandler.
//
// 功能分析：
//  1. 按 ValidPrefixes 顺序尝试查找 "<prefix>-<filename>" 可执行文件。
//  2. 使用 exec.LookPath 遵循系统 PATH 解析规则，找到第一个可执行插件后立即返回。
//
// 注意点：
//  1. shouldSkipOnLookPathErr 会过滤某些平台相关或可忽略的查找错误，避免一个前缀失败就
//     阻断后续前缀。
//  2. 查找顺序就是优先级；多个插件同名时，PATH 顺序和 ValidPrefixes 顺序共同决定结果。
func (h *DefaultPluginHandler) Lookup(filename string) (string, bool) {
	for _, prefix := range h.ValidPrefixes {
		path, err := exec.LookPath(fmt.Sprintf("%s-%s", prefix, filename))
		if shouldSkipOnLookPathErr(err) || len(path) == 0 {
			continue
		}
		return path, true
	}
	return "", false
}

// Execute implements PluginHandler.
//
// 功能分析：
//  1. 在类 Unix 平台使用 syscall.Exec 用插件进程替换当前 kubectl 进程，保留 argv 和环境。
//     这样插件的退出码、信号行为和标准流更接近原生命令。
//  2. Windows 不支持同样的 exec syscall，因此退化为 os/exec 启动子进程，并把 stdin、
//     stdout、stderr 和环境变量转交给插件。
//
// 注意点：
//  1. Unix 分支成功时不会返回；后续代码不会继续执行。
//  2. Windows 分支在插件成功后显式 os.Exit(0)，失败时返回错误交给上层格式化。
//  3. environment 由调用方传入，通常是 os.Environ；如果未来需要修改插件环境，应在调用
//     HandlePluginCommand 前完成。
func (h *DefaultPluginHandler) Execute(executablePath string, cmdArgs, environment []string) error {
	// Windows does not support exec syscall.
	if runtime.GOOS == "windows" {
		cmd := command(executablePath, cmdArgs...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Stdin = os.Stdin
		cmd.Env = environment
		err := cmd.Run()
		if err == nil {
			os.Exit(0)
		}
		return err
	}

	// invoke cmd binary relaying the environment and args given
	// append executablePath to cmdArgs, as execve will make first argument the "binary name".
	return syscall.Exec(executablePath, append([]string{executablePath}, cmdArgs...), environment)
}

// command 构造 Windows 插件执行时使用的 exec.Cmd，并尽量解析出真实可执行路径。
//
// 功能分析：
//  1. exec.Cmd 要求 Path 和 Args[0] 同时表达被执行程序，这里保持两者一致。
//  2. 当 name 不包含路径分隔符时，再调用 exec.LookPath 解析 PATH，保留 .exe、.bat 等
//     Windows 扩展名解析结果。
//
// 注意点：即使 LookPath 返回 ErrDot，只要返回了可用路径也会更新 cmd.Path，这是为了保留
// Go 在 Windows 上解析出的实际扩展名，避免插件文件存在但执行路径不完整。
func command(name string, arg ...string) *exec.Cmd {
	cmd := &exec.Cmd{
		Path: name,
		Args: append([]string{name}, arg...),
	}
	if filepath.Base(name) == name {
		lp, err := exec.LookPath(name)
		if lp != "" && !shouldSkipOnLookPathErr(err) {
			// Update cmd.Path even if err is non-nil.
			// If err is ErrDot (especially on Windows), lp may include a resolved
			// extension (like .exe or .bat) that should be preserved.
			cmd.Path = lp
		}
	}
	return cmd
}

// HandlePluginCommand receives a pluginHandler and command-line arguments and attempts to find
// a plugin executable on the PATH that satisfies the given arguments.
//
// 功能分析：
//  1. 从命令行参数开头连续收集非 flag 片段，并把片段中的 "-" 转成 "_"，兼容 kubectl
//     插件命名约定。
//  2. 从最长命令名开始查找插件。例如 "kubectl foo bar" 会先查找 kubectl-foo-bar，
//     找不到再尝试更短名称，直到 minArgs 限制停止。
//  3. 找到插件后，把没有参与插件文件名匹配的剩余参数传给插件执行。
//
// 注意点：
//  1. flag 不能放在插件名前面；一旦第一个参数就是 flag，会返回明确错误。
//  2. minArgs 用来防止过度回退，例如内置 "create" 的未知子命令不应轻易落到
//     kubectl-create 插件。
//  3. 找不到插件不是错误，返回 nil 让 kubectl 后续按普通未知命令路径处理。
//  4. 找到插件后 Execute 在 Unix 上通常不会返回，因为当前进程会被插件替换。
func HandlePluginCommand(pluginHandler PluginHandler, cmdArgs []string, minArgs int) error {
	var remainingArgs []string // all "non-flag" arguments
	for _, arg := range cmdArgs {
		if strings.HasPrefix(arg, "-") {
			break
		}
		remainingArgs = append(remainingArgs, strings.Replace(arg, "-", "_", -1))
	}

	if len(remainingArgs) == 0 {
		// the length of cmdArgs is at least 1
		return fmt.Errorf("flags cannot be placed before plugin name: %s", cmdArgs[0])
	}

	foundBinaryPath := ""

	// attempt to find binary, starting at longest possible name with given cmdArgs
	for len(remainingArgs) > 0 {
		path, found := pluginHandler.Lookup(strings.Join(remainingArgs, "-"))
		if !found {
			remainingArgs = remainingArgs[:len(remainingArgs)-1]
			if len(remainingArgs) < minArgs {
				// we shouldn't continue searching with shorter names.
				// this is especially for not searching kubectl-create plugin
				// when kubectl-create-foo plugin is not found.
				break
			}

			continue
		}

		foundBinaryPath = path
		break
	}

	if len(foundBinaryPath) == 0 {
		return nil
	}

	// invoke cmd binary relaying the current environment and args given
	if err := pluginHandler.Execute(foundBinaryPath, cmdArgs[len(remainingArgs):], os.Environ()); err != nil {
		return err
	}

	return nil
}

// IsSubcommandPluginAllowed returns the given command is allowed
// to use plugin as subcommand if the subcommand does not exist as builtin.
//
// 功能分析：目前只允许 create 这类内置命令扩展插件子命令，用来支持
// "kubectl create <custom>" 在内置子命令不存在时转发到插件。
//
// 注意点：扩大白名单会改变未知子命令的解析优先级，可能让原本应报错的命令被 PATH 中的
// 插件接管，新增前需要评估兼容性和安全影响。
func IsSubcommandPluginAllowed(foundCmd string) bool {
	allowedCmds := map[string]struct{}{"create": {}}
	_, ok := allowedCmds[foundCmd]
	return ok
}
