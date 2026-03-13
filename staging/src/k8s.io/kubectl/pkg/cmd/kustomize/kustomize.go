/*
Copyright 2019 The Kubernetes Authors.

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

package kustomize

import (
	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	"sigs.k8s.io/kustomize/kustomize/v5/commands/build"
	"sigs.k8s.io/kustomize/kyaml/filesys"
)

// NewCmdKustomize returns an adapted kustomize build command.
func NewCmdKustomize(streams genericiooptions.IOStreams) *cobra.Command {
	h := build.MakeHelp("kubectl", "kustomize")
	return build.NewCmdBuild(
		filesys.MakeFsOnDisk(),
		&build.Help{
			Use:     h.Use,
			Short:   i18n.T(h.Short),
			Long:    templates.LongDesc(i18n.T(h.Long)),
			Example: templates.Examples(i18n.T(h.Example)),
		},
		streams.Out)
}
