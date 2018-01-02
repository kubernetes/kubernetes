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

var _ = Describe("Decoding the version of network config", func() {
	var (
		decoder     *version.ConfigDecoder
		configBytes []byte
	)

	BeforeEach(func() {
		decoder = &version.ConfigDecoder{}
		configBytes = []byte(`{ "cniVersion": "4.3.2" }`)
	})

	Context("when the version is explict", func() {
		It("returns the version", func() {
			version, err := decoder.Decode(configBytes)
			Expect(err).NotTo(HaveOccurred())

			Expect(version).To(Equal("4.3.2"))
		})
	})

	Context("when the version is not present in the config", func() {
		BeforeEach(func() {
			configBytes = []byte(`{ "not-a-version-field": "foo" }`)
		})

		It("assumes the config is version 0.1.0", func() {
			version, err := decoder.Decode(configBytes)
			Expect(err).NotTo(HaveOccurred())

			Expect(version).To(Equal("0.1.0"))
		})
	})

	Context("when the config data is malformed", func() {
		BeforeEach(func() {
			configBytes = []byte(`{{{`)
		})

		It("returns a useful error", func() {
			_, err := decoder.Decode(configBytes)
			Expect(err).To(MatchError(HavePrefix(
				"decoding version from network config: invalid character",
			)))
		})
	})
})
