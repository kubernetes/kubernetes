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

package dns

import (
	"fmt"
	"io/ioutil"
	"strings"

	"github.com/spf13/cobra"
)

// CmdDNSSuffix is used by agnhost Cobra.
var CmdDNSSuffix = &cobra.Command{
	Use:   "dns-suffix",
	Short: "Prints the host's DNS suffix list",
	Long:  `Prints the DNS suffixes of this host.`,
	Args:  cobra.MaximumNArgs(0),
	Run:   printDNSSuffixList,
}

// CmdDNSServerList is used by agnhost Cobra.
var CmdDNSServerList = &cobra.Command{
	Use:   "dns-server-list",
	Short: "Prints the host's DNS Server list",
	Long:  `Prints the DNS Server list of this host.`,
	Args:  cobra.MaximumNArgs(0),
	Run:   printDNSServerList,
}

// CmdEtcHosts is used by agnhost Cobra.
var CmdEtcHosts = &cobra.Command{
	Use:   "etc-hosts",
	Short: "Prints the host's /etc/hosts file",
	Long:  `Prints the "hosts" file of this host."`,
	Args:  cobra.MaximumNArgs(0),
	Run:   printHostsFile,
}

func printDNSSuffixList(cmd *cobra.Command, args []string) {
	dnsSuffixList := GetDNSSuffixList()
	fmt.Println(strings.Join(dnsSuffixList, ","))
}

func printDNSServerList(cmd *cobra.Command, args []string) {
	dnsServerList := getDNSServerList()
	fmt.Println(strings.Join(dnsServerList, ","))
}

func printHostsFile(cmd *cobra.Command, args []string) {
	fmt.Println(readFile(etcHostsFile))
}

func readFile(fileName string) string {
	fileData, err := ioutil.ReadFile(fileName)
	if err != nil {
		panic(err)
	}

	return string(fileData)
}
