/*
Copyright 2016 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func NewCmdToken(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "token",
		Short: "Manage tokens used by init/join",

		// Without this callback, if a user runs just the "token"
		// command without a subcommand, or with an invalid subcommand,
		// cobra will print usage information, but still exit cleanly.
		// We want to return an error code in these cases so that the
		// user knows that their command was invalid.
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) < 1 {
				return errors.New("missing subcommand; 'token' is not meant to be run on its own")
			} else {
				return fmt.Errorf("invalid subcommand: %s", args[0])
			}
		},
	}

	cmd.AddCommand(NewCmdTokenGenerate(out))
	return cmd
}

func NewCmdTokenGenerate(out io.Writer) *cobra.Command {
	return &cobra.Command{
		Use:   "generate",
		Short: "Generate and print a token suitable for use with init/join",
		Long: dedent.Dedent(`
			This command will print out a randomly-generated token that you can use with
			the "init" and "join" commands.

			You don't have to use this command in order to generate a token, you can do so
			yourself as long as it's in the format "<6 characters>.<16 characters>". This
			command is provided for convenience to generate tokens in that format.

			You can also use "kubeadm init" without specifying a token, and it will
			generate and print one for you.
		`),
		Run: func(cmd *cobra.Command, args []string) {
			err := RunGenerateToken(out)
			kubeadmutil.CheckErr(err)
		},
	}
}

func RunGenerateToken(out io.Writer) error {
	s := &kubeadmapi.Secrets{}
	err := util.GenerateToken(s)
	if err != nil {
		return err
	}

	fmt.Fprintln(out, s.GivenToken)
	return nil
}
