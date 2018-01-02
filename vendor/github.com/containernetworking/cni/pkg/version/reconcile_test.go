// Copyright 2016 CNI authors
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

package version_test

import (
	"github.com/containernetworking/cni/pkg/version"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Reconcile versions of net config with versions supported by plugins", func() {
	var (
		reconciler *version.Reconciler
		pluginInfo version.PluginInfo
	)

	BeforeEach(func() {
		reconciler = &version.Reconciler{}
		pluginInfo = version.PluginSupports("1.2.3", "4.3.2")
	})

	It("succeeds if the config version is supported by the plugin", func() {
		err := reconciler.Check("4.3.2", pluginInfo)
		Expect(err).NotTo(HaveOccurred())
	})

	Context("when the config version is not supported by the plugin", func() {
		It("returns a helpful error", func() {
			err := reconciler.Check("0.1.0", pluginInfo)

			Expect(err).To(Equal(&version.ErrorIncompatible{
				Config:    "0.1.0",
				Supported: []string{"1.2.3", "4.3.2"},
			}))

			Expect(err.Error()).To(Equal(`incompatible CNI versions: config is "0.1.0", plugin supports ["1.2.3" "4.3.2"]`))
		})
	})
})
