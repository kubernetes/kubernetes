// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"fmt"
	"io"
	"log"

	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/krusty"
	"sigs.k8s.io/kustomize/api/types"
)

var theArgs struct {
	kustomizationPath string
}

var theFlags struct {
	outputPath string
	enable     struct {
		resourceIdChanges bool
		plugins           bool
		managedByLabel    bool
	}
	loadRestrictor string
	reorderOutput  string
	fnOptions      types.FnPluginLoadingOptions
}

type Help struct {
	Use     string
	Short   string
	Long    string
	Example string
}

func MakeHelp(pgmName, cmdName string) *Help {
	fN := konfig.DefaultKustomizationFileName()
	return &Help{
		Use:   cmdName + " DIR",
		Short: "Build a kustomization target from a directory or URL.",
		Long: fmt.Sprintf(`Build a set of KRM resources using a '%s' file.
The DIR argument must be a path to a directory containing
'%s', or a git repository URL with a path suffix
specifying same with respect to the repository root.
If DIR is omitted, '.' is assumed.
`, fN, fN),
		Example: fmt.Sprintf(`# Build the current working directory
  %s %s

# Build some shared configuration directory
  %s %s /home/config/production

# Build from github
  %s %s https://github.com/kubernetes-sigs/kustomize.git/examples/helloWorld?ref=v1.0.6
`, pgmName, cmdName, pgmName, cmdName, pgmName, cmdName),
	}
}

// NewCmdBuild creates a new build command.
func NewCmdBuild(
	fSys filesys.FileSystem, help *Help, writer io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:          help.Use,
		Short:        help.Short,
		Long:         help.Long,
		Example:      help.Example,
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := Validate(args); err != nil {
				return err
			}
			k := krusty.MakeKustomizer(
				HonorKustomizeFlags(krusty.MakeDefaultOptions()),
			)
			m, err := k.Run(fSys, theArgs.kustomizationPath)
			if err != nil {
				return err
			}
			if theFlags.outputPath != "" && fSys.IsDir(theFlags.outputPath) {
				// Ignore writer; write to o.outputPath directly.
				return MakeWriter(fSys).WriteIndividualFiles(
					theFlags.outputPath, m)
			}
			yml, err := m.AsYaml()
			if err != nil {
				return err
			}
			if theFlags.outputPath != "" {
				// Ignore writer; write to o.outputPath directly.
				return fSys.WriteFile(theFlags.outputPath, yml)
			}
			_, err = writer.Write(yml)
			return err
		},
	}
	AddFlagOutputPath(cmd.Flags())
	AddFunctionBasicsFlags(cmd.Flags())
	AddFlagLoadRestrictor(cmd.Flags())
	AddFlagEnablePlugins(cmd.Flags())
	AddFlagReorderOutput(cmd.Flags())
	AddFlagEnableManagedbyLabel(cmd.Flags())
	AddFlagAllowResourceIdChanges(cmd.Flags())
	return cmd
}

// Validate validates build command args and flags.
func Validate(args []string) error {
	if len(args) > 1 {
		return fmt.Errorf(
			"specify one path to " +
				konfig.DefaultKustomizationFileName())
	}
	if len(args) == 0 {
		theArgs.kustomizationPath = filesys.SelfDir
	} else {
		theArgs.kustomizationPath = args[0]
	}
	if err := validateFlagLoadRestrictor(); err != nil {
		return err
	}
	return validateFlagReorderOutput()
}

// HonorKustomizeFlags feeds command line data to the krusty options.
// Flags and such are held in private package variables.
func HonorKustomizeFlags(kOpts *krusty.Options) *krusty.Options {
	kOpts.DoLegacyResourceSort = getFlagReorderOutput() == legacy
	kOpts.LoadRestrictions = getFlagLoadRestrictorValue()
	if theFlags.enable.plugins {
		c, err := konfig.EnabledPluginConfig(types.BploUseStaticallyLinked)
		if err != nil {
			log.Fatal(err)
		}
		c.FnpLoadingOptions = theFlags.fnOptions
		kOpts.PluginConfig = c
	}
	kOpts.AddManagedbyLabel = isManagedByLabelEnabled()
	kOpts.AllowResourceIdChanges = theFlags.enable.resourceIdChanges
	return kOpts
}
