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
	"github.com/spf13/cobra"
)

func main() {
	cmdDNSSuffix := &cobra.Command{
		Use:   "dns-suffix",
		Short: "Prints the host's DNS suffix list.",
		Long:  `prints the DNS suffixes of this host.`,
		Args:  cobra.MaximumNArgs(0),
		Run:   printDNSSuffixList,
	}

	rootCmd := &cobra.Command{Use: "app"}
	rootCmd.AddCommand(cmdDNSSuffix)
	rootCmd.Execute()
}
