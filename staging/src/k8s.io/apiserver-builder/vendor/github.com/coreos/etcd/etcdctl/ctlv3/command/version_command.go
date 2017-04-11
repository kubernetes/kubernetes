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

package command

import (
	"fmt"

	"github.com/coreos/etcd/version"
	"github.com/coreos/go-semver/semver"
	"github.com/spf13/cobra"
)

// NewVersionCommand prints out the version of etcd.
func NewVersionCommand() *cobra.Command {
	return &cobra.Command{
		Use:   "version",
		Short: "Prints the version of etcdctl",
		Run:   versionCommandFunc,
	}
}

func versionCommandFunc(cmd *cobra.Command, args []string) {
	fmt.Println("etcdctl version:", version.Version)
	ver, err := semver.NewVersion(version.Version)
	var vs string
	if err == nil {
		vs = fmt.Sprintf("%d.%d", ver.Major, ver.Minor)
	} else {
		vs = "unknown"
	}
	fmt.Println("API version:", vs)
}
