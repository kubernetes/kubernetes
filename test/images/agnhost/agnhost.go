/*
Copyright 2019 The Kubernetes Authors.

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

package main

import (
	"flag"

	"github.com/spf13/cobra"

	"k8s.io/klog"
	"k8s.io/kubernetes/test/images/agnhost/net"
	"k8s.io/kubernetes/test/images/agnhost/netexec"
	"k8s.io/kubernetes/test/images/agnhost/nettest"
	"k8s.io/kubernetes/test/images/agnhost/webhook"
)

func main() {
	cmdDNSSuffix := &cobra.Command{
		Use:   "dns-suffix",
		Short: "Prints the host's DNS suffix list",
		Long:  `Prints the DNS suffixes of this host.`,
		Args:  cobra.MaximumNArgs(0),
		Run:   printDNSSuffixList,
	}

	cmdDNSServerList := &cobra.Command{
		Use:   "dns-server-list",
		Short: "Prints the host's DNS Server list",
		Long:  `Prints the DNS Server list of this host.`,
		Args:  cobra.MaximumNArgs(0),
		Run:   printDNSServerList,
	}

	cmdEtcHosts := &cobra.Command{
		Use:   "etc-hosts",
		Short: "Prints the host's /etc/hosts file",
		Long:  `Prints the "hosts" file of this host."`,
		Args:  cobra.MaximumNArgs(0),
		Run:   printHostsFile,
	}

	cmdPause := &cobra.Command{
		Use:   "pause",
		Short: "Pauses",
		Long:  `Pauses the execution. Useful for keeping the containers running, so other commands can be executed.`,
		Args:  cobra.MaximumNArgs(0),
		Run:   pause,
	}

	rootCmd := &cobra.Command{Use: "app"}
	rootCmd.AddCommand(cmdDNSSuffix)
	rootCmd.AddCommand(cmdDNSServerList)
	rootCmd.AddCommand(cmdEtcHosts)
	rootCmd.AddCommand(cmdPause)

	rootCmd.AddCommand(net.CmdNet)
	rootCmd.AddCommand(netexec.CmdNetexec)
	rootCmd.AddCommand(nettest.CmdNettest)
	rootCmd.AddCommand(webhook.CmdWebhook)

	// NOTE(claudiub): Some tests are passing logging related flags, so we need to be able to
	// accept them. This will also include them in the printed help.
	loggingFlags := &flag.FlagSet{}
	klog.InitFlags(loggingFlags)
	rootCmd.PersistentFlags().AddGoFlagSet(loggingFlags)
	rootCmd.Execute()
}
