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

var _ = Describe("Flavor", func() {
	var (
		server     *mocks.Server
		client     *Client
		flavorSpec *FlavorCreateSpec
	)

	BeforeEach(func() {
		server, client = testSetup()
		flavorSpec = &FlavorCreateSpec{
			Name: randomString(10, "go-sdk-flavor-"),
			Kind: "vm",
			Cost: []QuotaLineItem{QuotaLineItem{"GB", 16, "vm.memory"}},
		}
	})

	AfterEach(func() {
		cleanFlavors(client)
		server.Close()
	})

	Describe("CreateGetAndDeleteFlavor", func() {
		It("Flavor create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_FLAVOR", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			flavorSpec := &FlavorCreateSpec{
				Name: randomString(10, "go-sdk-flavor-"),
				Kind: "vm",
				Cost: []QuotaLineItem{QuotaLineItem{"GB", 16, "vm.memory"}},
			}
			task, err := client.Flavors.Create(flavorSpec)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_FLAVOR"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_FLAVOR", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Flavors.Delete(task.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_FLAVOR"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("GetFlavor", func() {
		var (
			flavorName string
			flavorID   string
		)

		BeforeEach(func() {
			flavorName, flavorID = createFlavor(server, client)
		})

		It("Get flavor succeeds", func() {
			server.SetResponseJson(200, Flavor{Name: flavorName})
			flavor, err := client.Flavors.Get(flavorID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(flavor).ShouldNot(BeNil())
			Expect(flavor.ID).Should(Equal(flavorID))
			Expect(flavor.Name).Should(Equal(flavorName))

			mockTask := createMockTask("DELETE_FLAVOR", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			_, err = client.Flavors.Delete(flavorID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		It("Get all flavor succeeds", func() {
			mockFlavorsPage := createMockFlavorsPage(Flavor{Name: flavorName})
			server.SetResponseJson(200, mockFlavorsPage)
			flavorList, err := client.Flavors.GetAll(&FlavorGetOptions{})
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(flavorList).ShouldNot(BeNil())

			var found bool
			for _, flavor := range flavorList.Items {
				if flavor.Name == flavorName && flavor.ID == flavorID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask := createMockTask("DELETE_FLAVOR", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			_, err = client.Flavors.Delete(flavorID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetTasks", func() {
		It("GetTasks returns a completed task", func() {
			mockTask := createMockTask("CREATE_FLAVOR", "COMPLETED")
			mockTask.Entity.ID = "mock-task-id"
			server.SetResponseJson(200, mockTask)

			task, err := client.Flavors.Create(flavorSpec)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTasksPage := createMockTasksPage(*mockTask)
			server.SetResponseJson(200, mockTasksPage)
			taskList, err := client.Flavors.GetTasks(task.Entity.ID, &TaskGetOptions{})

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(taskList).ShouldNot(BeNil())
			Expect(taskList.Items).Should(ContainElement(*task))

			mockTask = createMockTask("DELETE_FLAVOR", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			_, err = client.Flavors.Delete(task.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})
})
