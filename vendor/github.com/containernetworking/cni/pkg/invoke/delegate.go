// Copyright 2016 CNI authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package invoke

import (
	"context"
	"os"
	"path/filepath"

	"github.com/containernetworking/cni/pkg/types"
)

func delegateCommon(delegatePlugin string, exec Exec) (string, Exec, error) {
	if exec == nil {
		exec = defaultExec
	}

	paths := filepath.SplitList(os.Getenv("CNI_PATH"))
	pluginPath, err := exec.FindInPath(delegatePlugin, paths)
	if err != nil {
		return "", nil, err
	}

	return pluginPath, exec, nil
}

// DelegateAdd calls the given delegate plugin with the CNI ADD action and
// JSON configuration
func DelegateAdd(ctx context.Context, delegatePlugin string, netconf []byte, exec Exec) (types.Result, error) {
	pluginPath, realExec, err := delegateCommon(delegatePlugin, exec)
	if err != nil {
		return nil, err
	}

	// DelegateAdd will override the original "CNI_COMMAND" env from process with ADD
	return ExecPluginWithResult(ctx, pluginPath, netconf, delegateArgs("ADD"), realExec)
}

// DelegateCheck calls the given delegate plugin with the CNI CHECK action and
// JSON configuration
func DelegateCheck(ctx context.Context, delegatePlugin string, netconf []byte, exec Exec) error {
	pluginPath, realExec, err := delegateCommon(delegatePlugin, exec)
	if err != nil {
		return err
	}

	// DelegateCheck will override the original CNI_COMMAND env from process with CHECK
	return ExecPluginWithoutResult(ctx, pluginPath, netconf, delegateArgs("CHECK"), realExec)
}

// DelegateDel calls the given delegate plugin with the CNI DEL action and
// JSON configuration
func DelegateDel(ctx context.Context, delegatePlugin string, netconf []byte, exec Exec) error {
	pluginPath, realExec, err := delegateCommon(delegatePlugin, exec)
	if err != nil {
		return err
	}

	// DelegateDel will override the original CNI_COMMAND env from process with DEL
	return ExecPluginWithoutResult(ctx, pluginPath, netconf, delegateArgs("DEL"), realExec)
}

// return CNIArgs used by delegation
func delegateArgs(action string) *DelegateArgs {
	return &DelegateArgs{
		Command: action,
	}
}
