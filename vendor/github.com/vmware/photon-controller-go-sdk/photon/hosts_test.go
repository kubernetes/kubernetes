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

var _ = Describe("Host", func() {
	var (
		server   *mocks.Server
		client   *Client
		hostSpec *HostCreateSpec
	)

	BeforeEach(func() {
		if isIntegrationTest() {
			Skip("Skipping Host test on integration mode. Unable to prevent address host collision")
		}
		server, client = testSetup()
		hostSpec = &HostCreateSpec{
			Username: randomString(10),
			Password: randomString(10),
			Address:  randomAddress(),
			Tags:     []string{"CLOUD"},
			Metadata: map[string]string{"Test": "go-sdk-host"},
		}
	})

	AfterEach(func() {
		cleanHosts(client)
		server.Close()
	})

	Describe("CreateAndDeleteHost", func() {
		It("host create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Hosts.Create(hostSpec, "deployment-Id")
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_HOST"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err = client.Hosts.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_HOST"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("ProvisionHost", func() {
		It("host provisioning succeeds", func() {
			mockTask := createMockTask("CREATE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Hosts.Create(hostSpec, "deployment-Id")
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("PROVISION_HOST", "COMPLETED")
			server.SetResponseJson(202, mockTask)

			task, err = client.Hosts.Provision(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("PROVISION_HOST"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Hosts.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

		})
	})

	Describe("GetHosts", func() {
		It("GetHosts succeeds", func() {
			mockTask := createMockTask("CREATE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Hosts.Create(hostSpec, "deployment-Id")
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, Host{Tags: hostSpec.Tags, ID: task.Entity.ID})
			host, err := client.Hosts.Get(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(host.ID).Should(Equal(task.Entity.ID))
			Expect(host.Tags).Should(Equal(hostSpec.Tags))

			mockTask = createMockTask("DELETE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Hosts.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		It("GetAll returns a host", func() {
			mockTask := createMockTask("CREATE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Hosts.Create(hostSpec, "deployment-Id")
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, &Hosts{[]Host{Host{Tags: hostSpec.Tags, ID: task.Entity.ID}}})
			hostList, err := client.Hosts.GetAll()
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(hostList).ShouldNot(BeNil())

			var found bool
			for _, host := range hostList.Items {
				if host.ID == task.Entity.ID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Hosts.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("SetHostAvailabilityZone", func() {
		It("set host's availability zone", func() {
			mockTask := createMockTask("SET_AVAILABILITYZONE", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			hostSetAvailabilityZoneOperation := &HostSetAvailabilityZoneOperation{AvailabilityZoneId: "availability-zone-Id"}
			task, err := client.Hosts.SetAvailabilityZone("host-Id", hostSetAvailabilityZoneOperation)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("SET_AVAILABILITYZONE"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("GetTasks", func() {
		It("GetTasks returns a completed task", func() {
			mockTask := createMockTask("CREATE_HOST", "COMPLETED")
			mockTask.Entity.ID = "mock-task-id"
			server.SetResponseJson(200, mockTask)

			task, err := client.Hosts.Create(hostSpec, "deployment-Id")
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTasksPage := createMockTasksPage(*mockTask)
			server.SetResponseJson(200, mockTasksPage)
			taskList, err := client.Hosts.GetTasks(task.Entity.ID, &TaskGetOptions{})

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(taskList).ShouldNot(BeNil())
			Expect(taskList.Items).Should(ContainElement(*task))

			mockTask = createMockTask("DELETE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Hosts.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetVms", func() {
		var (
			tenantID     string
			resName      string
			projID       string
			imageID      string
			flavorSpec   *FlavorCreateSpec
			vmFlavorSpec *FlavorCreateSpec
			vmSpec       *VmCreateSpec
		)

		BeforeEach(func() {
			tenantID = createTenant(server, client)
			resName = createResTicket(server, client, tenantID)
			projID = createProject(server, client, tenantID, resName)
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

			vmSpec = &VmCreateSpec{
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
		})

		AfterEach(func() {
			cleanVMs(client, projID)
			cleanImages(client)
			cleanFlavors(client)
			cleanTenants(client)
		})

		It("GetVms returns a list of vms", func() {
			mockTask := createMockTask("CREATE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			hostTask, err := client.Hosts.Create(hostSpec, "deployment-Id")
			hostTask, err = client.Tasks.Wait(hostTask.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			vmTask, err := client.Projects.CreateVM(projID, vmSpec)
			vmTask, err = client.Tasks.Wait(vmTask.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, &VMs{[]VM{VM{Name: vmSpec.Name}}})
			vmList, err := client.Hosts.GetVMs(hostTask.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			var found bool
			for _, vm := range vmList.Items {
				if vm.Name == vmSpec.Name && vm.ID == vmTask.Entity.ID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			vmTask, err = client.VMs.Delete(vmTask.Entity.ID)
			vmTask, err = client.Tasks.Wait(vmTask.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("DELETE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			hostTask, err = client.Hosts.Delete(hostTask.Entity.ID)
			hostTask, err = client.Tasks.Wait(hostTask.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("SuspendHostAndResumeHost", func() {
		It("Suspend Host and Resume Host succeeds", func() {
			mockTask := createMockTask("CREATE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Hosts.Create(hostSpec, "deployment-Id")
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("SUSPEND_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err = client.Hosts.Suspend(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("SUSPEND_HOST"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("RESUME_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err = client.Hosts.Resume(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("RESUME_HOST"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("EnterAndExitMaintenanceMode", func() {
		var (
			hostID string
		)

		BeforeEach(func() {
			hostID = ""

			// Create host
			mockTask := createMockTask("CREATE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Hosts.Create(hostSpec, "deployment-Id")
			Expect(err).Should(BeNil())

			task, err = client.Tasks.Wait(task.ID)
			if task != nil {
				hostID = task.Entity.ID
			}
			Expect(err).Should(BeNil())

			// Suspend host
			mockTask = createMockTask("SUSPEND_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Hosts.Suspend(hostID)
			task, err = client.Tasks.Wait(task.ID)
			Expect(err).Should(BeNil())
		})

		AfterEach(func() {
			// Delete host
			if len(hostID) > 0 {
				mockTask := createMockTask("DELETE_HOST", "COMPLETED")
				server.SetResponseJson(200, mockTask)
				task, err := client.Hosts.Delete(hostID)
				task, err = client.Tasks.Wait(task.ID)
				if err != nil {
					GinkgoT().Log(err)
				}
			}
		})

		It("Host Enter and Exit Maintenance Mode succeeds", func() {
			mockTask := createMockTask("ENTER_MAINTENANCE_MODE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Hosts.EnterMaintenanceMode(hostID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("ENTER_MAINTENANCE_MODE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("EXIT_MAINTENANCE_MODE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Hosts.ExitMaintenanceMode(hostID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("EXIT_MAINTENANCE_MODE"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})
})
