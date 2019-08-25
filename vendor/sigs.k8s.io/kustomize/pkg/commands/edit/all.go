// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package edit

import (
	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/pkg/commands/edit/add"
	"sigs.k8s.io/kustomize/pkg/commands/edit/fix"
	"sigs.k8s.io/kustomize/pkg/commands/edit/remove"
	"sigs.k8s.io/kustomize/pkg/commands/edit/set"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/loader"
)

// NewCmdEdit returns an instance of 'edit' subcommand.
func NewCmdEdit(
	fSys fs.FileSystem, v ifc.Validator, kf ifc.KunstructuredFactory) *cobra.Command {
	c := &cobra.Command{
		Use:   "edit",
		Short: "Edits a kustomization file",
		Long:  "",
		Example: `
	# Adds a configmap to the kustomization file
	kustomize edit add configmap NAME --from-literal=k=v

	# Sets the nameprefix field
	kustomize edit set nameprefix <prefix-value>

	# Sets the namesuffix field
	kustomize edit set namesuffix <suffix-value>
`,
		Args: cobra.MinimumNArgs(1),
	}

	c.AddCommand(
		add.NewCmdAdd(fSys, loader.NewFileLoaderAtCwd(v, fSys), kf),
		set.NewCmdSet(fSys, v),
		fix.NewCmdFix(fSys),
		remove.NewCmdRemove(fSys),
	)
	return c
}
