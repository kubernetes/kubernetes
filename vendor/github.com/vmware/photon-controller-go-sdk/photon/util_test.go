// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package photon

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type options struct {
	A int    `urlParam:"a"`
	B string `urlParam:"b"`
}

var _ = Describe("Utils", func() {
	It("GetQueryString", func() {
		opts := &options{5, "a test"}
		query := getQueryString(opts)
		Expect(query).Should(Equal("?a=5&b=a+test"))
	})
})
