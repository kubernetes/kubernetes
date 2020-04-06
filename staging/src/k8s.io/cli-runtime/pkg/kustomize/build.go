// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kustomize

import (
	"io"
	"log"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/krusty"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/yaml"
)

// Options contain the options for running a build
type Options struct {
	kustomizationPath string
	outputPath        string
	outOrder          reorderOutput
}

// NewOptions creates a Options object
func NewOptions(p, o string) *Options {
	return &Options{
		kustomizationPath: p,
		outputPath:        o,
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
func NewCmdBuild(out io.Writer) *cobra.Command {
	var o Options
	cmd := &cobra.Command{
		Use: "build {path}",
		Short: "Print configuration per contents of " +
			konfig.DefaultKustomizationFileName(),
		Example:      examples,
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			err := o.Validate(args)
			if err != nil {
				return err
			}
			return o.RunBuild(out)
		},
	}

	cmd.Flags().StringVarP(
		&o.outputPath,
		"output", "o", "",
		"If specified, write the build output to this path.")
	addFlagLoadRestrictor(cmd.Flags())
	addFlagEnablePlugins(cmd.Flags())
	addFlagReorderOutput(cmd.Flags())
	cmd.AddCommand(NewCmdBuildPrune(out))
	return cmd
}

// Validate validates build command.
func (o *Options) Validate(args []string) (err error) {
	if len(args) > 1 {
		return errors.New(
			"specify one path to " +
				konfig.DefaultKustomizationFileName())
	}
	if len(args) == 0 {
		o.kustomizationPath = filesys.SelfDir
	} else {
		o.kustomizationPath = args[0]
	}
	err = validateFlagLoadRestrictor()
	if err != nil {
		return err
	}
	o.outOrder, err = validateFlagReorderOutput()
	return
}

func (o *Options) makeOptions() *krusty.Options {
	opts := &krusty.Options{
		DoLegacyResourceSort: o.outOrder == legacy,
		LoadRestrictions:     getFlagLoadRestrictorValue(),
		DoPrune:              false,
	}
	if isFlagEnablePluginsSet() {
		c, err := konfig.EnabledPluginConfig()
		if err != nil {
			log.Fatal(err)
		}
		opts.PluginConfig = c
	} else {
		opts.PluginConfig = konfig.DisabledPluginConfig()
	}
	return opts
}

func (o *Options) RunBuild(out io.Writer) error {
	fSys := filesys.MakeFsOnDisk()
	k := krusty.MakeKustomizer(fSys, o.makeOptions())
	m, err := k.Run(o.kustomizationPath)
	if err != nil {
		return err
	}
	return o.emitResources(out, fSys, m)
}

func (o *Options) RunBuildPrune(out io.Writer) error {
	fSys := filesys.MakeFsOnDisk()
	opts := o.makeOptions()
	opts.DoPrune = true
	k := krusty.MakeKustomizer(fSys, opts)
	m, err := k.Run(o.kustomizationPath)
	if err != nil {
		return err
	}
	return o.emitResources(out, fSys, m)
}

func (o *Options) emitResources(
	out io.Writer, fSys filesys.FileSystem, m resmap.ResMap) error {
	if o.outputPath != "" && fSys.IsDir(o.outputPath) {
		return writeIndividualFiles(fSys, o.outputPath, m)
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

func NewCmdBuildPrune(out io.Writer) *cobra.Command {
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
			return o.RunBuildPrune(out)
		},
	}
	return cmd
}

func writeIndividualFiles(
	fSys filesys.FileSystem, folderPath string, m resmap.ResMap) error {
	byNamespace := m.GroupedByCurrentNamespace()
	for namespace, resList := range byNamespace {
		for _, res := range resList {
			fName := fileName(res)
			if len(byNamespace) > 1 {
				fName = strings.ToLower(namespace) + "_" + fName
			}
			err := writeFile(fSys, folderPath, fName, res)
			if err != nil {
				return err
			}
		}
	}
	for _, res := range m.NonNamespaceable() {
		err := writeFile(fSys, folderPath, fileName(res), res)
		if err != nil {
			return err
		}
	}
	return nil
}

func fileName(res *resource.Resource) string {
	return strings.ToLower(res.GetGvk().String()) +
		"_" + strings.ToLower(res.GetName()) + ".yaml"
}

func writeFile(
	fSys filesys.FileSystem, path, fName string, res *resource.Resource) error {
	out, err := yaml.Marshal(res.Map())
	if err != nil {
		return err
	}
	return fSys.WriteFile(filepath.Join(path, fName), out)
}
