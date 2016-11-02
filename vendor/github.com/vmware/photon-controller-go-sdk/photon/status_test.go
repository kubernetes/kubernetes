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
	"github.com/vmware/photon-controller-go-sdk/photon/internal/mocks"
)

var _ = Describe("Status", func() {
	var (
		server *mocks.Server
		client *Client
	)

	BeforeEach(func() {
		server, client = testSetup()
	})

	AfterEach(func() {
		server.Close()
	})

	// Simple preliminary test. Make sure status API correctly deserializes the response
	It("GetStatus200", func() {
		expectedStruct := Status{"READY", []Component{{"chairman", "", "READY"}, {"housekeeper", "", "READY"}}}
		server.SetResponseJson(200, expectedStruct)

		status, err := client.Status.Get()
		GinkgoT().Log(err)
		Expect(err).Should(BeNil())
		Expect(status.Status).Should(Equal(expectedStruct.Status))
		Expect(status.Components).ShouldNot(HaveLen(1))
	})
})
