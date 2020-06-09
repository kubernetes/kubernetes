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
	"bytes"
	"os/exec"
	"strings"
)

const etcHostsFile = "C:/Windows/System32/drivers/etc/hosts"

func getDNSSuffixList() []string {
	output := runCommand("powershell", "-Command", "(Get-DnsClient)[0].SuffixSearchList")
	if len(output) > 0 {
		return strings.Split(output, "\r\n")
	}

	panic("Could not find DNS search list!")
}

func getDNSServerList() []string {
	output := runCommand("powershell", "-Command", "(Get-DnsClientServerAddress).ServerAddresses")
	if len(output) > 0 {
		return strings.Split(output, "\r\n")
	}

	panic("Could not find DNS Server list!")
}

func runCommand(name string, arg ...string) string {
	var out bytes.Buffer
	cmd := exec.Command(name, arg...)
	cmd.Stdout = &out

	err := cmd.Run()
	if err != nil {
		panic(err)
	}

	return strings.TrimSpace(out.String())
}
