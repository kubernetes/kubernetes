// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package add

import (
	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
)

// NewCmdAdd returns an instance of 'add' subcommand.
func NewCmdAdd(
	fSys fs.FileSystem,
	ldr ifc.Loader,
	kf ifc.KunstructuredFactory) *cobra.Command {
	c := &cobra.Command{
		Use:   "add",
		Short: "Adds an item to the kustomization file.",
		Long:  "",
		Example: `
	# Adds a secret to the kustomization file
	kustomize edit add secret NAME --from-literal=k=v

	# Adds a configmap to the kustomization file
	kustomize edit add configmap NAME --from-literal=k=v

	# Adds a resource to the kustomization
	kustomize edit add resource <filepath>

	# Adds a patch to the kustomization
	kustomize edit add patch <filepath>

	# Adds one or more base directories to the kustomization
	kustomize edit add base <filepath>
	kustomize edit add base <filepath1>,<filepath2>,<filepath3>

	# Adds one or more commonLabels to the kustomization
	kustomize edit add label {labelKey1:labelValue1},{labelKey2:labelValue2}

	# Adds one or more commonAnnotations to the kustomization
	kustomize edit add annotation {annotationKey1:annotationValue1},{annotationKey2:annotationValue2}
`,
		Args: cobra.MinimumNArgs(1),
	}
	c.AddCommand(
		newCmdAddResource(fSys),
		newCmdAddPatch(fSys),
		newCmdAddSecret(fSys, ldr, kf),
		newCmdAddConfigMap(fSys, ldr, kf),
		newCmdAddBase(fSys),
		newCmdAddLabel(fSys, ldr.Validator().MakeLabelValidator()),
		newCmdAddAnnotation(fSys, ldr.Validator().MakeAnnotationValidator()),
	)
	return c
}
