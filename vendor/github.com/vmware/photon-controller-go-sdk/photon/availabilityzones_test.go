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

var _ = Describe("AvailabilityZone", func() {
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

	Describe("CreateAndDeleteAvailabilityZone", func() {
		It("AvailabilityZone create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_AVAILABILITYZONE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			availabilityzoneSpec := &AvailabilityZoneCreateSpec{Name: randomString(10, "go-sdk-availabilityzone-")}
			task, err := client.AvailabilityZones.Create(availabilityzoneSpec)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_AVAILABILITYZONE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_AVAILABILITYZONE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.AvailabilityZones.Delete(task.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_AVAILABILITYZONE"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})

		It("AvailabilityZone create fails", func() {
			availabilityzoneSpec := &AvailabilityZoneCreateSpec{}
			task, err := client.AvailabilityZones.Create(availabilityzoneSpec)

			Expect(err).ShouldNot(BeNil())
			Expect(task).Should(BeNil())
		})
	})

	Describe("GetAvailabilityZone", func() {
		It("Get returns availabilityzone", func() {
			mockTask := createMockTask("CREATE_AVAILABILITYZONE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			availabilityzoneName := randomString(10, "go-sdk-availabilityzone-")
			availabilityzoneSpec := &AvailabilityZoneCreateSpec{Name: availabilityzoneName}
			task, err := client.AvailabilityZones.Create(availabilityzoneSpec)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_AVAILABILITYZONE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			server.SetResponseJson(200, AvailabilityZone{Name: availabilityzoneName})
			availabilityzone, err := client.AvailabilityZones.Get(task.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(availabilityzone).ShouldNot(BeNil())

			var found bool
			if availabilityzone.Name == availabilityzoneName && availabilityzone.ID == task.Entity.ID {
				found = true
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_AVAILABILITYZONE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			_, err = client.AvailabilityZones.Delete(task.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		It("Get all returns availabilityzones", func() {
			mockTask := createMockTask("CREATE_AVAILABILITYZONE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			availabilityzoneName := randomString(10, "go-sdk-availabilityzone-")
			availabilityzoneSpec := &AvailabilityZoneCreateSpec{Name: availabilityzoneName}
			task, err := client.AvailabilityZones.Create(availabilityzoneSpec)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_AVAILABILITYZONE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			availZonePage := createMockAvailZonesPage(AvailabilityZone{Name: availabilityzoneName})
			server.SetResponseJson(200, availZonePage)
			availabilityzones, err := client.AvailabilityZones.GetAll()

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(availabilityzones).ShouldNot(BeNil())

			var found bool
			for _, availabilityzone := range availabilityzones.Items {
				if availabilityzone.Name == availabilityzoneName && availabilityzone.ID == task.Entity.ID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_AVAILABILITYZONE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			_, err = client.AvailabilityZones.Delete(task.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetAvailabilityZoneTasks", func() {
		var (
			option string
		)

		Context("no extra options for GetTask", func() {
			BeforeEach(func() {
				option = ""
			})

			It("GetTasks returns a completed task", func() {
				mockTask := createMockTask("CREATE_AVAILABILITYZONE", "COMPLETED")
				mockTask.Entity.ID = "mock-task-id"
				server.SetResponseJson(200, mockTask)
				availabilityzoneSpec := &AvailabilityZoneCreateSpec{Name: randomString(10, "go-sdk-availabilityzone-")}
				task, err := client.AvailabilityZones.Create(availabilityzoneSpec)

				GinkgoT().Log(err)
				Expect(err).Should(BeNil())
				Expect(task).ShouldNot(BeNil())
				Expect(task.Operation).Should(Equal("CREATE_AVAILABILITYZONE"))
				Expect(task.State).Should(Equal("COMPLETED"))

				mockTasksPage := createMockTasksPage(*mockTask)
				server.SetResponseJson(200, mockTasksPage)
				taskList, err := client.AvailabilityZones.GetTasks(task.Entity.ID, &TaskGetOptions{State: option})
				GinkgoT().Log(err)
				Expect(err).Should(BeNil())
				Expect(taskList).ShouldNot(BeNil())
				Expect(taskList.Items).Should(ContainElement(*task))

				mockTask = createMockTask("DELETE_AVAILABILITYZONE", "COMPLETED")
				server.SetResponseJson(200, mockTask)
				_, err = client.AvailabilityZones.Delete(task.Entity.ID)

				GinkgoT().Log(err)
				Expect(err).Should(BeNil())
			})
		})

		Context("Searching COMPLETED state for GetTask", func() {
			BeforeEach(func() {
				option = "COMPLETED"
			})

			It("GetTasks returns a completed task", func() {
				mockTask := createMockTask("CREATE_AVAILABILITYZONE", "COMPLETED")
				mockTask.Entity.ID = "mock-task-id"
				server.SetResponseJson(200, mockTask)
				availabilityzoneSpec := &AvailabilityZoneCreateSpec{Name: randomString(10, "go-sdk-availabilityzone-")}
				task, err := client.AvailabilityZones.Create(availabilityzoneSpec)

				GinkgoT().Log(err)
				Expect(err).Should(BeNil())
				Expect(task).ShouldNot(BeNil())
				Expect(task.Operation).Should(Equal("CREATE_AVAILABILITYZONE"))
				Expect(task.State).Should(Equal("COMPLETED"))

				mockTasksPage := createMockTasksPage(*mockTask)
				server.SetResponseJson(200, mockTasksPage)
				taskList, err := client.AvailabilityZones.GetTasks(task.Entity.ID, &TaskGetOptions{State: option})
				GinkgoT().Log(err)
				Expect(err).Should(BeNil())
				Expect(taskList).ShouldNot(BeNil())
				Expect(taskList.Items).Should(ContainElement(*task))

				mockTask = createMockTask("DELETE_AVAILABILITYZONE", "COMPLETED")
				server.SetResponseJson(200, mockTask)
				_, err = client.AvailabilityZones.Delete(task.Entity.ID)

				GinkgoT().Log(err)
				Expect(err).Should(BeNil())
			})
		})
	})
})
