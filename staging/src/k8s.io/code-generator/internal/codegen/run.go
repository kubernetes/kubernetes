/*
Copyright 2023 The Kubernetes Authors.

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

package codegen

import (
	"k8s.io/code-generator/internal/codegen/command"
	"k8s.io/code-generator/internal/codegen/command/client"
	"k8s.io/code-generator/internal/codegen/command/help"
	"k8s.io/code-generator/internal/codegen/command/helpers"
	"k8s.io/code-generator/internal/codegen/command/openapi"
	"k8s.io/code-generator/internal/codegen/execution"
)

// Run will run the K8s code-generator Vars.
func Run(opts ...execution.Option) {
	ex := execution.New(opts...)
	cmds := []Command{
		helpers.Command{},
		openapi.Command{},
		client.Command{},
	}
	cmd := parse(ex, append([]Command{
		help.Command{
			Others: asUsages(cmds),
		},
	}, cmds...))
	cmd.Run(ex)
}

func asUsages(cmds []Command) []command.Usage {
	usages := make([]command.Usage, 0, len(cmds))
	for _, cmd := range cmds {
		usages = append(usages, cmd)
	}
	return usages
}
