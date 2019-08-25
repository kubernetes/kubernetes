// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package add

import (
	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/pkg/commands/kustfile"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/types"
)

// newCmdAddConfigMap returns a new command.
func newCmdAddConfigMap(
	fSys fs.FileSystem,
	ldr ifc.Loader,
	kf ifc.KunstructuredFactory) *cobra.Command {
	var flags flagsAndArgs
	cmd := &cobra.Command{
		Use:   "configmap NAME [--from-file=[key=]source] [--from-literal=key1=value1]",
		Short: "Adds a configmap to the kustomization file.",
		Long:  "",
		Example: `
	# Adds a configmap to the kustomization file (with a specified key)
	kustomize edit add configmap my-configmap --from-file=my-key=file/path --from-literal=my-literal=12345

	# Adds a configmap to the kustomization file (key is the filename)
	kustomize edit add configmap my-configmap --from-file=file/path

	# Adds a configmap from env-file
	kustomize edit add configmap my-configmap --from-env-file=env/path.env
`,
		RunE: func(_ *cobra.Command, args []string) error {
			err := flags.ExpandFileSource(fSys)
			if err != nil {
				return err
			}

			err = flags.Validate(args)
			if err != nil {
				return err
			}

			// Load the kustomization file.
			mf, err := kustfile.NewKustomizationFile(fSys)
			if err != nil {
				return err
			}

			kustomization, err := mf.Read()
			if err != nil {
				return err
			}

			// Add the flagsAndArgs map to the kustomization file.
			err = addConfigMap(ldr, kustomization, flags, kf)
			if err != nil {
				return err
			}

			// Write out the kustomization file with added configmap.
			return mf.Write(kustomization)
		},
	}

	cmd.Flags().StringSliceVar(
		&flags.FileSources,
		"from-file",
		[]string{},
		"Key file can be specified using its file path, in which case file basename will be used as configmap "+
			"key, or optionally with a key and file path, in which case the given key will be used.  Specifying a "+
			"directory will iterate each named file in the directory whose basename is a valid configmap key.")
	cmd.Flags().StringArrayVar(
		&flags.LiteralSources,
		"from-literal",
		[]string{},
		"Specify a key and literal value to insert in configmap (i.e. mykey=somevalue)")
	cmd.Flags().StringVar(
		&flags.EnvFileSource,
		"from-env-file",
		"",
		"Specify the path to a file to read lines of key=val pairs to create a configmap (i.e. a Docker .env file).")

	return cmd
}

// addConfigMap adds a configmap to a kustomization file.
// Note: error may leave kustomization file in an undefined state.
// Suggest passing a copy of kustomization file.
func addConfigMap(
	ldr ifc.Loader,
	k *types.Kustomization,
	flags flagsAndArgs, kf ifc.KunstructuredFactory) error {
	args := findOrMakeConfigMapArgs(k, flags.Name)
	mergeFlagsIntoCmArgs(args, flags)
	// Validate by trying to create corev1.configmap.
	_, err := kf.MakeConfigMap(ldr, k.GeneratorOptions, args)
	if err != nil {
		return err
	}
	return nil
}

func findOrMakeConfigMapArgs(m *types.Kustomization, name string) *types.ConfigMapArgs {
	for i, v := range m.ConfigMapGenerator {
		if name == v.Name {
			return &m.ConfigMapGenerator[i]
		}
	}
	// config map not found, create new one and add it to the kustomization file.
	cm := &types.ConfigMapArgs{GeneratorArgs: types.GeneratorArgs{Name: name}}
	m.ConfigMapGenerator = append(m.ConfigMapGenerator, *cm)
	return &m.ConfigMapGenerator[len(m.ConfigMapGenerator)-1]
}

func mergeFlagsIntoCmArgs(args *types.ConfigMapArgs, flags flagsAndArgs) {
	if len(flags.LiteralSources) > 0 {
		args.LiteralSources = append(
			args.LiteralSources, flags.LiteralSources...)
	}
	if len(flags.FileSources) > 0 {
		args.FileSources = append(
			args.FileSources, flags.FileSources...)
	}
	if flags.EnvFileSource != "" {
		args.EnvSources = append(
			args.EnvSources, flags.EnvFileSource)
	}
}
