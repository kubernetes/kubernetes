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

package types_test

import (
	"reflect"

	. "github.com/containernetworking/cni/pkg/types"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/extensions/table"
	. "github.com/onsi/gomega"
)

var _ = Describe("UnmarshallableBool UnmarshalText", func() {
	DescribeTable("string to bool detection should succeed in all cases",
		func(inputs []string, expected bool) {
			for _, s := range inputs {
				var ub UnmarshallableBool
				err := ub.UnmarshalText([]byte(s))
				Expect(err).ToNot(HaveOccurred())
				Expect(ub).To(Equal(UnmarshallableBool(expected)))
			}
		},
		Entry("parse to true", []string{"True", "true", "1"}, true),
		Entry("parse to false", []string{"False", "false", "0"}, false),
	)

	Context("When passed an invalid value", func() {
		It("should result in an error", func() {
			var ub UnmarshallableBool
			err := ub.UnmarshalText([]byte("invalid"))
			Expect(err).To(HaveOccurred())
		})
	})
})

var _ = Describe("UnmarshallableString UnmarshalText", func() {
	DescribeTable("string to string detection should succeed in all cases",
		func(inputs []string, expected string) {
			for _, s := range inputs {
				var us UnmarshallableString
				err := us.UnmarshalText([]byte(s))
				Expect(err).ToNot(HaveOccurred())
				Expect(string(us)).To(Equal(expected))
			}
		},
		Entry("parse empty string", []string{""}, ""),
		Entry("parse non-empty string", []string{"notempty"}, "notempty"),
	)
})

var _ = Describe("GetKeyField", func() {
	type testcontainer struct {
		Valid string `json:"valid,omitempty"`
	}
	var (
		container          = testcontainer{Valid: "valid"}
		containerInterface = func(i interface{}) interface{} { return i }(&container)
		containerValue     = reflect.ValueOf(containerInterface)
	)
	Context("When a valid field is provided", func() {
		It("should return the correct field", func() {
			field := GetKeyField("Valid", containerValue)
			Expect(field.String()).To(Equal("valid"))
		})
	})
})

var _ = Describe("LoadArgs", func() {
	Context("When no arguments are passed", func() {
		It("LoadArgs should succeed", func() {
			err := LoadArgs("", struct{}{})
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("When unknown arguments are passed and ignored", func() {
		It("LoadArgs should succeed", func() {
			ca := CommonArgs{}
			err := LoadArgs("IgnoreUnknown=True;Unk=nown", &ca)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("When unknown arguments are passed and not ignored", func() {
		It("LoadArgs should fail", func() {
			ca := CommonArgs{}
			err := LoadArgs("Unk=nown", &ca)
			Expect(err).To(HaveOccurred())
		})
	})

	Context("When unknown arguments are passed and explicitly not ignored", func() {
		It("LoadArgs should fail", func() {
			ca := CommonArgs{}
			err := LoadArgs("IgnoreUnknown=0, Unk=nown", &ca)
			Expect(err).To(HaveOccurred())
		})
	})

	Context("When known arguments are passed", func() {
		It("LoadArgs should succeed", func() {
			ca := CommonArgs{}
			err := LoadArgs("IgnoreUnknown=1", &ca)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Context("When known arguments are passed and cannot be unmarshalled", func() {
		It("LoadArgs should fail", func() {
			conf := struct {
				IP IPNet
			}{}
			err := LoadArgs("IP=10.0.0.0/24", &conf)
			Expect(err).To(HaveOccurred())

		})
	})
})
