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
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/vmware/photon-controller-go-sdk/photon/internal/mocks"
)

var _ = Describe("Project", func() {
	var (
		server     *mocks.Server
		client     *Client
		tenantID   string
		resName    string
		projID     string
		flavorName string
		flavorID   string
	)

	BeforeEach(func() {
		server, client = testSetup()
		tenantID = createTenant(server, client)
		resName = createResTicket(server, client, tenantID)
		projID = createProject(server, client, tenantID, resName)
		flavorName, flavorID = createFlavor(server, client)

	})

	AfterEach(func() {
		cleanDisks(client, projID)
		cleanFlavors(client)
		cleanTenants(client)
		server.Close()
	})

	Describe("GetProjectTasks", func() {
		It("GetTasks returns a completed task", func() {
			mockTask := createMockTask("CREATE_DISK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			diskSpec := &DiskCreateSpec{
				Flavor:     flavorName,
				Kind:       "persistent-disk",
				CapacityGB: 2,
				Name:       randomString(10, "go-sdk-disk-"),
			}

			task, err := client.Projects.CreateDisk(projID, diskSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTasksPage := createMockTasksPage(*mockTask)
			server.SetResponseJson(200, mockTasksPage)
			taskList, err := client.Projects.GetTasks(projID, &TaskGetOptions{})
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(taskList).ShouldNot(BeNil())
			Expect(taskList.Items).Should(ContainElement(*task))

			// Clean disk
			mockTask = createMockTask("DELETE_DISK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Disks.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetProjectDisks", func() {
		It("GetAll returns disk", func() {
			mockTask := createMockTask("CREATE_DISK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			diskSpec := &DiskCreateSpec{
				Flavor:     flavorName,
				Kind:       "persistent-disk",
				CapacityGB: 2,
				Name:       randomString(10, "go-sdk-disk-"),
			}

			task, err := client.Projects.CreateDisk(projID, diskSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			diskMock := PersistentDisk{
				Name:       diskSpec.Name,
				Flavor:     diskSpec.Flavor,
				CapacityGB: diskSpec.CapacityGB,
				Kind:       diskSpec.Kind,
			}
			server.SetResponseJson(200, &DiskList{[]PersistentDisk{diskMock}})
			diskList, err := client.Projects.GetDisks(projID, &DiskGetOptions{})
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(diskList).ShouldNot(BeNil())

			var found bool
			for _, disk := range diskList.Items {
				if disk.Name == diskSpec.Name && disk.ID == task.Entity.ID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_DISK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Disks.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetProjectVms", func() {
		var (
			imageID      string
			flavorSpec   *FlavorCreateSpec
			vmFlavorSpec *FlavorCreateSpec
		)

		BeforeEach(func() {
			imageID = createImage(server, client)
			flavorSpec = &FlavorCreateSpec{
				[]QuotaLineItem{QuotaLineItem{"COUNT", 1, "ephemeral-disk.cost"}},
				"ephemeral-disk",
				randomString(10, "go-sdk-flavor-"),
			}

			_, err := client.Flavors.Create(flavorSpec)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			vmFlavorSpec = &FlavorCreateSpec{
				Name: randomString(10, "go-sdk-flavor-"),
				Kind: "vm",
				Cost: []QuotaLineItem{
					QuotaLineItem{"GB", 2, "vm.memory"},
					QuotaLineItem{"COUNT", 4, "vm.cpu"},
				},
			}
			_, err = client.Flavors.Create(vmFlavorSpec)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		AfterEach(func() {
			cleanVMs(client, projID)
		})

		It("GetAll returns vm", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			vmSpec := &VmCreateSpec{
				Flavor:        vmFlavorSpec.Name,
				SourceImageID: imageID,
				AttachedDisks: []AttachedDisk{
					AttachedDisk{
						CapacityGB: 1,
						Flavor:     flavorSpec.Name,
						Kind:       "ephemeral-disk",
						Name:       randomString(10),
						State:      "STARTED",
						BootDisk:   true,
					},
				},
				Name: randomString(10, "go-sdk-vm-"),
			}

			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockVm := VM{Name: vmSpec.Name}
			server.SetResponseJson(200, createMockVmsPage(mockVm))
			vmList, err := client.Projects.GetVMs(projID, &VmGetOptions{})
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(vmList).ShouldNot(BeNil())

			var found bool
			for _, vm := range vmList.Items {
				if vm.Name == vmSpec.Name && vm.ID == task.Entity.ID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetProjectServices", func() {
		It("GetAll returns service", func() {
			if isIntegrationTest() {
				Skip("Skipping service test on integration mode. Need to set extendedProperties to use real IPs and masks")
			}
			mockTask := createMockTask("CREATE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			serviceSpec := &ServiceCreateSpec{
				Name:               randomString(10, "go-sdk-service-"),
				Type:               "KUBERNETES",
				WorkerCount:        50,
				BatchSizeWorker:    5,
				ExtendedProperties: map[string]string{},
			}

			task, err := client.Projects.CreateService(projID, serviceSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockService := Service{Name: serviceSpec.Name}
			server.SetResponseJson(200, createMockServicesPage(mockService))
			serviceList, err := client.Projects.GetServices(projID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(serviceList).ShouldNot(BeNil())

			var found bool
			for _, service := range serviceList.Items {
				if service.Name == serviceSpec.Name && service.ID == task.Entity.ID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("SecurityGroups", func() {
		It("sets security groups for a project", func() {
			// Set security groups for the project
			expected := &Tenant{
				SecurityGroups: []SecurityGroup{
					{randomString(10), false},
					{randomString(10), false},
				},
			}
			mockTask := createMockTask("SET_TENANT_SECURITY_GROUPS", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			securityGroups := &SecurityGroupsSpec{
				[]string{expected.SecurityGroups[0].Name, expected.SecurityGroups[1].Name},
			}
			createTask, err := client.Projects.SetSecurityGroups(projID, securityGroups)
			createTask, err = client.Tasks.Wait(createTask.ID)
			Expect(err).Should(BeNil())

			// Get the security groups for the project
			server.SetResponseJson(200, expected)
			project, err := client.Projects.Get(projID)
			Expect(err).Should(BeNil())
			fmt.Fprintf(GinkgoWriter, "Got project: %+v", project)
			Expect(expected.SecurityGroups).Should(Equal(project.SecurityGroups))
		})
	})

})
