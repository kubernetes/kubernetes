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

// newCmdAddSecret returns a new command.
func newCmdAddSecret(
	fSys fs.FileSystem,
	ldr ifc.Loader,
	kf ifc.KunstructuredFactory) *cobra.Command {
	var flags flagsAndArgs
	cmd := &cobra.Command{
		Use:   "secret NAME [--from-file=[key=]source] [--from-literal=key1=value1] [--type=Opaque|kubernetes.io/tls]",
		Short: "Adds a secret to the kustomization file.",
		Long:  "",
		Example: `
	# Adds a secret to the kustomization file (with a specified key)
	kustomize edit add secret my-secret --from-file=my-key=file/path --from-literal=my-literal=12345

	# Adds a secret to the kustomization file (key is the filename)
	kustomize edit add secret my-secret --from-file=file/path

	# Adds a secret from env-file
	kustomize edit add secret my-secret --from-env-file=env/path.env
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
			err = addSecret(ldr, kustomization, flags, kf)
			if err != nil {
				return err
			}

			// Write out the kustomization file with added secret.
			return mf.Write(kustomization)
		},
	}

	cmd.Flags().StringSliceVar(
		&flags.FileSources,
		"from-file",
		[]string{},
		"Key file can be specified using its file path, in which case file basename will be used as secret "+
			"key, or optionally with a key and file path, in which case the given key will be used.  Specifying a "+
			"directory will iterate each named file in the directory whose basename is a valid secret key.")
	cmd.Flags().StringArrayVar(
		&flags.LiteralSources,
		"from-literal",
		[]string{},
		"Specify a key and literal value to insert in secret (i.e. mykey=somevalue)")
	cmd.Flags().StringVar(
		&flags.EnvFileSource,
		"from-env-file",
		"",
		"Specify the path to a file to read lines of key=val pairs to create a secret (i.e. a Docker .env file).")
	cmd.Flags().StringVar(
		&flags.Type,
		"type",
		"Opaque",
		"Specify the secret type this can be 'Opaque' (default), or 'kubernetes.io/tls'")

	return cmd
}

// addSecret adds a secret to a kustomization file.
// Note: error may leave kustomization file in an undefined state.
// Suggest passing a copy of kustomization file.
func addSecret(
	ldr ifc.Loader,
	k *types.Kustomization,
	flags flagsAndArgs, kf ifc.KunstructuredFactory) error {
	args := findOrMakeSecretArgs(k, flags.Name, flags.Type)
	mergeFlagsIntoGeneratorArgs(&args.GeneratorArgs, flags)
	// Validate by trying to create corev1.secret.
	_, err := kf.MakeSecret(ldr, k.GeneratorOptions, args)
	if err != nil {
		return err
	}
	return nil
}

func findOrMakeSecretArgs(m *types.Kustomization, name, secretType string) *types.SecretArgs {
	for i, v := range m.SecretGenerator {
		if name == v.Name {
			return &m.SecretGenerator[i]
		}
	}
	// secret not found, create new one and add it to the kustomization file.
	secret := &types.SecretArgs{
		GeneratorArgs: types.GeneratorArgs{Name: name},
		Type:          secretType}
	m.SecretGenerator = append(m.SecretGenerator, *secret)
	return &m.SecretGenerator[len(m.SecretGenerator)-1]
}

func mergeFlagsIntoGeneratorArgs(args *types.GeneratorArgs, flags flagsAndArgs) {
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
