// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"fmt"
	"io"
	"log"

	"github.com/spf13/cobra"
	flag "github.com/spf13/pflag"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/krusty"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/filesys"
)

var theArgs struct {
	kustomizationPath string
}

var theFlags struct {
	outputPath string
	enable     struct {
		plugins        bool
		managedByLabel bool
		helm           bool
	}
	helmCommand    string
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
		Short: "Build a kustomization target from a directory or URL",
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
				HonorKustomizeFlags(krusty.MakeDefaultOptions(), cmd.Flags()),
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
	msg := "Error marking flag '%s' as deprecated: %v"
	err := cmd.Flags().MarkDeprecated(flagReorderOutputName,
		"use the new 'sortOptions' field in kustomization.yaml instead.")
	if err != nil {
		log.Fatalf(msg, flagReorderOutputName, err)
	}
	err = cmd.Flags().MarkDeprecated(managedByFlag,
		"The flag `enable-managedby-label` has been deprecated. Use the `managedByLabel` option in the `buildMetadata` field instead.")
	if err != nil {
		log.Fatalf(msg, managedByFlag, err)
	}

	AddFlagEnableHelm(cmd.Flags())
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
func HonorKustomizeFlags(kOpts *krusty.Options, flags *flag.FlagSet) *krusty.Options {
	kOpts.Reorder = getFlagReorderOutput(flags)
	kOpts.LoadRestrictions = getFlagLoadRestrictorValue()
	if theFlags.enable.plugins {
		c := types.EnabledPluginConfig(types.BploUseStaticallyLinked)
		c.FnpLoadingOptions = theFlags.fnOptions
		kOpts.PluginConfig = c
	} else {
		kOpts.PluginConfig.HelmConfig.Enabled = theFlags.enable.helm
	}
	kOpts.PluginConfig.HelmConfig.Command = theFlags.helmCommand
	kOpts.AddManagedbyLabel = isManagedByLabelEnabled()
	return kOpts
}
