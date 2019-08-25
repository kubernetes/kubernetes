// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/ifc/transformer"
	"sigs.k8s.io/kustomize/pkg/loader"
	"sigs.k8s.io/kustomize/pkg/pgmconfig"
	"sigs.k8s.io/kustomize/pkg/plugins"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/target"
	"sigs.k8s.io/kustomize/plugin/builtin"
	"sigs.k8s.io/yaml"
)

// Options contain the options for running a build
type Options struct {
	kustomizationPath string
	outputPath        string
	loadRestrictor    loader.LoadRestrictorFunc
	outOrder          reorderOutput
}

// NewOptions creates a Options object
func NewOptions(p, o string) *Options {
	return &Options{
		kustomizationPath: p,
		outputPath:        o,
		loadRestrictor:    loader.RestrictionRootOnly,
	}
}

var examples = `
To generate the resources specified in 'someDir/kustomization.yaml', run

  kustomize build someDir

The default argument to 'build' is '.' (the current working directory).

The argument can be a URL resolving to a directory
with a kustomization.yaml file, e.g.

  kustomize build \
    github.com/kubernetes-sigs/kustomize//examples/multibases/dev/?ref=v1.0.6

The URL should be formulated as described at
https://github.com/hashicorp/go-getter#url-format
`

// NewCmdBuild creates a new build command.
func NewCmdBuild(
	out io.Writer, fSys fs.FileSystem,
	v ifc.Validator, rf *resmap.Factory,
	ptf transformer.Factory) *cobra.Command {
	var o Options

	pluginConfig := plugins.DefaultPluginConfig()
	pl := plugins.NewLoader(pluginConfig, rf)

	cmd := &cobra.Command{
		Use:          "build {path}",
		Short:        "Print configuration per contents of " + pgmconfig.KustomizationFileNames[0],
		Example:      examples,
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			err := o.Validate(args)
			if err != nil {
				return err
			}
			return o.RunBuild(out, v, fSys, rf, ptf, pl)
		},
	}

	cmd.Flags().StringVarP(
		&o.outputPath,
		"output", "o", "",
		"If specified, write the build output to this path.")
	loader.AddFlagLoadRestrictor(cmd.Flags())
	plugins.AddFlagEnablePlugins(
		cmd.Flags(), &pluginConfig.Enabled)
	addFlagReorderOutput(cmd.Flags())
	cmd.AddCommand(NewCmdBuildPrune(out, v, fSys, rf, ptf, pl))
	return cmd
}

// Validate validates build command.
func (o *Options) Validate(args []string) (err error) {
	if len(args) > 1 {
		return errors.New(
			"specify one path to " + pgmconfig.KustomizationFileNames[0])
	}
	if len(args) == 0 {
		o.kustomizationPath = loader.CWD
	} else {
		o.kustomizationPath = args[0]
	}
	o.loadRestrictor, err = loader.ValidateFlagLoadRestrictor()
	if err != nil {
		return err
	}
	o.outOrder, err = validateFlagReorderOutput()
	return
}

// RunBuild runs build command.
func (o *Options) RunBuild(
	out io.Writer, v ifc.Validator, fSys fs.FileSystem,
	rf *resmap.Factory, ptf transformer.Factory,
	pl *plugins.Loader) error {
	ldr, err := loader.NewLoader(
		o.loadRestrictor, v, o.kustomizationPath, fSys)
	if err != nil {
		return err
	}
	defer ldr.Cleanup()
	kt, err := target.NewKustTarget(ldr, rf, ptf, pl)
	if err != nil {
		return err
	}
	m, err := kt.MakeCustomizedResMap()
	if err != nil {
		return err
	}
	return o.emitResources(out, fSys, m)
}

func (o *Options) RunBuildPrune(
	out io.Writer, v ifc.Validator, fSys fs.FileSystem,
	rf *resmap.Factory, ptf transformer.Factory,
	pl *plugins.Loader) error {
	ldr, err := loader.NewLoader(
		o.loadRestrictor, v, o.kustomizationPath, fSys)
	if err != nil {
		return err
	}
	defer ldr.Cleanup()
	kt, err := target.NewKustTarget(ldr, rf, ptf, pl)
	if err != nil {
		return err
	}
	m, err := kt.MakePruneConfigMap()
	if err != nil {
		return err
	}
	return o.emitResources(out, fSys, m)
}

func (o *Options) emitResources(
	out io.Writer, fSys fs.FileSystem, m resmap.ResMap) error {
	if o.outputPath != "" && fSys.IsDir(o.outputPath) {
		return writeIndividualFiles(fSys, o.outputPath, m)
	}
	if o.outOrder == legacy {
		// Done this way just to show how overall sorting
		// can be performed by a plugin.  This particular
		// plugin doesn't require configuration; just make
		// it and call transform.
		builtin.NewLegacyOrderTransformerPlugin().Transform(m)
	}
	res, err := m.AsYaml()
	if err != nil {
		return err
	}
	if o.outputPath != "" {
		return fSys.WriteFile(o.outputPath, res)
	}
	_, err = out.Write(res)
	return err
}

func NewCmdBuildPrune(
	out io.Writer, v ifc.Validator, fSys fs.FileSystem,
	rf *resmap.Factory, ptf transformer.Factory,
	pl *plugins.Loader) *cobra.Command {
	var o Options

	cmd := &cobra.Command{
		Use:          "alpha-inventory [path]",
		Short:        "Print the inventory object which contains a list of all other objects",
		Example:      examples,
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			err := o.Validate(args)
			if err != nil {
				return err
			}
			return o.RunBuildPrune(out, v, fSys, rf, ptf, pl)
		},
	}
	return cmd
}

func writeIndividualFiles(
	fSys fs.FileSystem, folderPath string, m resmap.ResMap) error {
	for _, res := range m.Resources() {
		filename := filepath.Join(
			folderPath,
			fmt.Sprintf(
				"%s_%s.yaml",
				strings.ToLower(res.GetGvk().String()),
				strings.ToLower(res.GetName()),
			),
		)
		out, err := yaml.Marshal(res.Map())
		if err != nil {
			return err
		}
		err = fSys.WriteFile(filename, out)
		if err != nil {
			return err
		}
	}
	return nil
}
