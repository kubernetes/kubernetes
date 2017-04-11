// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package ctlv3 contains the main entry point for the etcdctl for v3 API.
package ctlv3

import (
	"time"

	"github.com/coreos/etcd/etcdctl/ctlv3/command"
	"github.com/spf13/cobra"
)

const (
	cliName        = "etcdctl"
	cliDescription = "A simple command line client for etcd3."

	defaultDialTimeout    = 2 * time.Second
	defaultCommandTimeOut = 5 * time.Second
)

var (
	globalFlags = command.GlobalFlags{}
)

var (
	rootCmd = &cobra.Command{
		Use:        cliName,
		Short:      cliDescription,
		SuggestFor: []string{"etcdctl"},
	}
)

func init() {
	rootCmd.PersistentFlags().StringSliceVar(&globalFlags.Endpoints, "endpoints", []string{"127.0.0.1:2379"}, "gRPC endpoints")

	rootCmd.PersistentFlags().StringVarP(&globalFlags.OutputFormat, "write-out", "w", "simple", "set the output format (simple, json, etc..)")
	rootCmd.PersistentFlags().BoolVar(&globalFlags.IsHex, "hex", false, "print byte strings as hex encoded strings")

	rootCmd.PersistentFlags().DurationVar(&globalFlags.DialTimeout, "dial-timeout", defaultDialTimeout, "dial timeout for client connections")
	rootCmd.PersistentFlags().DurationVar(&globalFlags.CommandTimeOut, "command-timeout", defaultCommandTimeOut, "timeout for short running command (excluding dial timeout)")

	// TODO: secure by default when etcd enables secure gRPC by default.
	rootCmd.PersistentFlags().BoolVar(&globalFlags.Insecure, "insecure-transport", true, "disable transport security for client connections")
	rootCmd.PersistentFlags().BoolVar(&globalFlags.InsecureSkipVerify, "insecure-skip-tls-verify", false, "skip server certificate verification")
	rootCmd.PersistentFlags().StringVar(&globalFlags.TLS.CertFile, "cert", "", "identify secure client using this TLS certificate file")
	rootCmd.PersistentFlags().StringVar(&globalFlags.TLS.KeyFile, "key", "", "identify secure client using this TLS key file")
	rootCmd.PersistentFlags().StringVar(&globalFlags.TLS.CAFile, "cacert", "", "verify certificates of TLS-enabled secure servers using this CA bundle")
	rootCmd.PersistentFlags().StringVar(&globalFlags.User, "user", "", "username[:password] for authentication (prompt if password is not supplied)")

	rootCmd.AddCommand(
		command.NewGetCommand(),
		command.NewPutCommand(),
		command.NewDelCommand(),
		command.NewTxnCommand(),
		command.NewCompactionCommand(),
		command.NewAlarmCommand(),
		command.NewDefragCommand(),
		command.NewEndpointCommand(),
		command.NewWatchCommand(),
		command.NewVersionCommand(),
		command.NewLeaseCommand(),
		command.NewMemberCommand(),
		command.NewSnapshotCommand(),
		command.NewMakeMirrorCommand(),
		command.NewMigrateCommand(),
		command.NewLockCommand(),
		command.NewElectCommand(),
		command.NewAuthCommand(),
		command.NewUserCommand(),
		command.NewRoleCommand(),
	)
}

func init() {
	cobra.EnablePrefixMatching = true
}

func Start() {
	rootCmd.SetUsageFunc(usageFunc)

	// Make help just show the usage
	rootCmd.SetHelpTemplate(`{{.UsageString}}`)

	if err := rootCmd.Execute(); err != nil {
		command.ExitWithError(command.ExitError, err)
	}
}
