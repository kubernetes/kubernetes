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

var _ = Describe("ResourceTicket", func() {
	var (
		server   *mocks.Server
		client   *Client
		tenantID string
	)

	BeforeEach(func() {
		server, client = testSetup()
		tenantID = createTenant(server, client)
	})

	AfterEach(func() {
		cleanTenants(client)
		server.Close()
	})

	Describe("GetResourceTicketTasks", func() {
		It("GetTasks returns a completed task", func() {
			mockTask := createMockTask("CREATE_RESOURCE_TICKET", "COMPLETED")
			mockTask.Entity.ID = "mock-task-id"
			server.SetResponseJson(200, mockTask)
			spec := &ResourceTicketCreateSpec{
				Name:   randomString(10),
				Limits: []QuotaLineItem{QuotaLineItem{Unit: "GB", Value: 16, Key: "vm.memory"}},
			}
			task, err := client.Tenants.CreateResourceTicket(tenantID, spec)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_RESOURCE_TICKET"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTasksPage := createMockTasksPage(*mockTask)
			server.SetResponseJson(200, mockTasksPage)
			taskList, err := client.ResourceTickets.GetTasks(task.Entity.ID, &TaskGetOptions{})
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(taskList).ShouldNot(BeNil())
			Expect(taskList.Items).Should(ContainElement(*task))
		})
	})

	Describe("GetResourceTicket", func() {
		It("Get succeeds", func() {
			mockTask := createMockTask("CREATE_RESOURCE_TICKET", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			spec := &ResourceTicketCreateSpec{
				Name:   randomString(10),
				Limits: []QuotaLineItem{QuotaLineItem{Unit: "GB", Value: 16, Key: "vm.memory"}},
			}
			task, err := client.Tenants.CreateResourceTicket(tenantID, spec)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_RESOURCE_TICKET"))
			Expect(task.State).Should(Equal("COMPLETED"))

			server.SetResponseJson(200, ResourceTicket{TenantId: tenantID, Name: spec.Name,
				Limits: spec.Limits})
			resourceTicket, err := client.ResourceTickets.Get(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(resourceTicket.Name).Should(Equal(spec.Name))
			Expect(resourceTicket.TenantId).Should(Equal(tenantID))
			Expect(resourceTicket.Limits).Should(Equal(spec.Limits))
			Expect(resourceTicket.ID).Should(Equal(task.Entity.ID))
		})
	})
})
