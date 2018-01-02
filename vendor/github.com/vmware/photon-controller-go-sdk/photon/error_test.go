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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/vmware/photon-controller-go-sdk/photon/internal/mocks"
)

var _ = Describe("ErrorTesting", func() {
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

	It("TaskError", func() {
		// Unit test only
		if isIntegrationTest() {
			return
		}
		task := &Task{ID: "fake-id", State: "ERROR", Operation: "fake-op"}
		server.SetResponseJson(200, task)
		task, err := client.Tasks.Wait(task.ID)
		taskErr, ok := err.(TaskError)
		Expect(ok).ShouldNot(BeNil())
		Expect(taskErr.ID).Should(Equal(task.ID))
	})

	It("TaskTimeoutError", func() {
		// Unit test only
		if isIntegrationTest() {
			return
		}
		client.options.TaskPollTimeout = 1 * time.Second
		task := &Task{ID: "fake-id", State: "QUEUED", Operation: "fake-op"}
		server.SetResponseJson(200, task)
		task, err := client.Tasks.Wait(task.ID)
		taskErr, ok := err.(TaskTimeoutError)
		Expect(ok).ShouldNot(BeNil())
		Expect(taskErr.ID).Should(Equal(task.ID))
	})

	It("HttpError", func() {
		// Unit test only
		if isIntegrationTest() {
			return
		}
		client.options.TaskPollTimeout = 1 * time.Second
		task := &Task{ID: "fake-id", State: "QUEUED", Operation: "fake-op"}
		server.SetResponseJson(500, "server error")
		task, err := client.Tasks.Wait(task.ID)
		taskErr, ok := err.(HttpError)
		Expect(ok).ShouldNot(BeNil())
		Expect(taskErr.StatusCode).Should(Equal(500))
	})
})
