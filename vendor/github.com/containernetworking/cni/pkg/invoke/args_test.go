// Copyright 2017 CNI authors
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

package invoke_test

import (
	"os"

	"github.com/containernetworking/cni/pkg/invoke"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Args", func() {
	Describe("AsEnv", func() {
		It("places the CNI_ environment variables ahead of any ambient variables", func() {
			args := invoke.Args{
				Command:     "ADD",
				ContainerID: "some-container-id",
				NetNS:       "/some/netns/path",
				PluginArgs: [][2]string{
					[2]string{"KEY1", "VALUE1"},
					[2]string{"KEY2", "VALUE2"},
				},
				IfName: "eth7",
				Path:   "/some/cni/path",
			}
			const numCNIEnvVars = 6

			latentVars := os.Environ()

			cniEnv := args.AsEnv()
			Expect(cniEnv).To(HaveLen(len(latentVars) + numCNIEnvVars))
			Expect(cniEnv[0:numCNIEnvVars]).To(Equal([]string{
				"CNI_COMMAND=ADD",
				"CNI_CONTAINERID=some-container-id",
				"CNI_NETNS=/some/netns/path",
				"CNI_ARGS=KEY1=VALUE1;KEY2=VALUE2",
				"CNI_IFNAME=eth7",
				"CNI_PATH=/some/cni/path",
			}))

			for i := range latentVars {
				Expect(cniEnv[numCNIEnvVars+i]).To(Equal(latentVars[i]))
			}
		})
	})
})
