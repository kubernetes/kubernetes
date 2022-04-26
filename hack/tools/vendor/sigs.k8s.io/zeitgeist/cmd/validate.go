/*
Copyright 2020 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"sigs.k8s.io/zeitgeist/dependencies"
)

type ValidateOptions struct {
	local    bool
	remote   bool
	config   string
	basePath string
}

var validateOpts = &ValidateOptions{}

const defaultConfigFile = "dependencies.yaml"

// validateCmd is a subcommand which invokes RunValidate()
var validateCmd = &cobra.Command{
	Use:           "validate",
	Short:         "Check dependencies locally and against upstream versions",
	SilenceUsage:  true,
	SilenceErrors: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		return RunValidate(validateOpts)
	},
}

func init() {
	// Submit types
	validateCmd.PersistentFlags().BoolVar(
		&validateOpts.local,
		"local",
		false,
		"validate dependencies locally",
	)

	validateCmd.PersistentFlags().BoolVar(
		&validateOpts.remote,
		"remote",
		false,
		"validate dependencies against specified upstreams",
	)

	validateCmd.PersistentFlags().StringVar(
		&validateOpts.config,
		"config",
		defaultConfigFile,
		"location of zeitgeist configuration file",
	)

	validateCmd.PersistentFlags().StringVar(
		&validateOpts.basePath,
		"base-path",
		"",
		"base path where will the start point to find the dependencies files to check. Defaults to where the program is called.",
	)

	rootCmd.AddCommand(validateCmd)
}

// RunValidate is the function invoked by 'krel gcbmgr', responsible for
// submitting release jobs to GCB
func RunValidate(opts *ValidateOptions) error {
	if err := opts.SetAndValidate(); err != nil {
		return errors.Wrap(err, "validating zeitgeist options")
	}

	client := dependencies.NewClient()

	if opts.remote {
		updates, err := client.RemoteCheck(opts.config)
		if err != nil {
			return errors.Wrap(err, "check remote dependencies")
		}

		for _, update := range updates {
			fmt.Println(update)
		}
	}

	return client.LocalCheck(opts.config, opts.basePath)
}

// SetAndValidate sets some default options and verifies if options are valid
func (o *ValidateOptions) SetAndValidate() error {
	logrus.Info("Validating zeitgeist options...")

	if o.basePath != "" {
		if _, err := os.Stat(o.basePath); os.IsNotExist(err) {
			return err
		}
	} else {
		dir, err := filepath.Abs(filepath.Dir(os.Args[0]))
		if err != nil {
			return err
		}
		o.basePath = dir
	}

	return nil
}
