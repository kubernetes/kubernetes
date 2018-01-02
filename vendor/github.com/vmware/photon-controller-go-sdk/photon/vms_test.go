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
	"os"
)

var _ = Describe("VM", func() {
	var (
		server       *mocks.Server
		client       *Client
		tenantID     string
		resName      string
		projID       string
		imageID      string
		flavorSpec   *FlavorCreateSpec
		vmFlavorSpec *FlavorCreateSpec
		vmSpec       *VmCreateSpec
	)

	BeforeEach(func() {
		server, client = testSetup()
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
		server.Close()
	})

	Describe("CreateAndDeleteVm", func() {
		It("Vm create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_VM"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_VM"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("GetVms", func() {
		It("Set vm metadata and get vm returns a vm ID with metadata", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			// Set VM metadata
			data := map[string]string{"key1": "value1"}
			metadata := &VmMetadata{Metadata: data}
			task, err = client.VMs.SetMetadata(task.Entity.ID, metadata)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockVm := &VM{Name: vmSpec.Name, Metadata: data}
			server.SetResponseJson(200, mockVm)
			vm, err := client.VMs.Get(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(vm).ShouldNot(BeNil())
			Expect(vm.Name).Should(Equal(vmSpec.Name))
			Expect(vm.Metadata).Should(Equal(data))

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("AttachDetach", func() {
		AfterEach(func() {
			cleanDisks(client, projID)
		})

		It("Attach detach disk succeeds", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			// Create persistent flavor
			persistentFlavorName, _ := createFlavor(server, client)

			// Create persistent disk
			diskSpec := &DiskCreateSpec{
				Flavor:     persistentFlavorName,
				Kind:       "persistent-disk",
				CapacityGB: 1,
				Name:       randomString(10, "go-sdk-disk-"),
				Affinities: []LocalitySpec{LocalitySpec{Kind: "vm", ID: task.Entity.ID}},
			}
			diskTask, err := client.Projects.CreateDisk(projID, diskSpec)
			diskTask, err = client.Tasks.Wait(diskTask.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, createMockTask("ATTACH_DISK", "QUEUED"))
			attachTask, err := client.VMs.AttachDisk(task.Entity.ID, &VmDiskOperation{DiskID: diskTask.Entity.ID})
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(attachTask.Operation).Should(Equal("ATTACH_DISK"))
			Expect(attachTask.State).Should(Equal("QUEUED"))

			mockTask = createMockTask("ATTACH_DISK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			attachTask, err = client.Tasks.Wait(attachTask.ID)
			GinkgoT().Log("Attach disk task: ", attachTask)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(attachTask.Operation).Should(Equal("ATTACH_DISK"))
			Expect(attachTask.State).Should(Equal("COMPLETED"))

			server.SetResponseJson(200, createMockTask("DETACH_DISK", "QUEUED"))
			detachTask, err := client.VMs.DetachDisk(task.Entity.ID, &VmDiskOperation{DiskID: diskTask.Entity.ID})
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(detachTask.Operation).Should(Equal("DETACH_DISK"))
			Expect(detachTask.State).Should(Equal("QUEUED"))

			mockTask = createMockTask("DETACH_DISK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			detachTask, err = client.Tasks.Wait(detachTask.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(detachTask.Operation).Should(Equal("DETACH_DISK"))
			Expect(detachTask.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_DISK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			diskTask, err = client.Disks.Delete(diskTask.Entity.ID)
			diskTask, err = client.Tasks.Wait(diskTask.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		It("Attach detach iso", func() {
			if isIntegrationTest() && !isRealAgent() {
				Skip("Skipping attach/detach ISO test unless REAL_AGENT env var is set")
			}

			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, createMockTask("ATTACH_ISO", "COMPLETED"))

			isoPath := "../testdata/ttylinux-pc_i486-16.1.iso"
			file, err := os.Open(isoPath)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			attachIsoTask, err := client.VMs.AttachISO(task.Entity.ID, file, "ttylinux-pc_i486-16.1.iso")
			attachIsoTask, err = client.Tasks.Wait(attachIsoTask.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(attachIsoTask.Operation).Should(Equal("ATTACH_ISO"))
			Expect(attachIsoTask.State).Should(Equal("COMPLETED"))

			server.SetResponseJson(200, createMockTask("DETACH_ISO", "COMPLETED"))
			detachIsoTask, err := client.VMs.DetachISO(task.Entity.ID)
			detachIsoTask, err = client.Tasks.Wait(detachIsoTask.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(detachIsoTask.Operation).Should(Equal("DETACH_ISO"))
			Expect(detachIsoTask.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("PowerOnOff Vms", func() {
		It("Power on, off succeeds", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			// Power on
			server.SetResponseJson(200, createMockTask("START_VM", "QUEUED"))
			powerOnTask, err := client.VMs.Start(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(powerOnTask.Operation).Should(Equal("START_VM"))
			Expect(powerOnTask.State).Should(Equal("QUEUED"))

			// Wait for power on
			server.SetResponseJson(200, createMockTask("START_VM", "COMPLETED"))
			powerOnTask, err = client.Tasks.Wait(powerOnTask.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(powerOnTask.Operation).Should(Equal("START_VM"))
			Expect(powerOnTask.State).Should(Equal("COMPLETED"))

			// Power off
			server.SetResponseJson(200, createMockTask("STOP_VM", "QUEUED"))
			powerOffTask, err := client.VMs.Stop(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(powerOffTask.Operation).Should(Equal("STOP_VM"))
			Expect(powerOffTask.State).Should(Equal("QUEUED"))

			// Wait for power off
			server.SetResponseJson(200, createMockTask("STOP_VM", "COMPLETED"))
			powerOffTask, err = client.Tasks.Wait(powerOffTask.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(powerOffTask.Operation).Should(Equal("STOP_VM"))
			Expect(powerOffTask.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetTasks", func() {
		It("GetTasks returns a completed task", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTasksPage := createMockTasksPage(*mockTask)
			server.SetResponseJson(200, mockTasksPage)
			taskList, err := client.VMs.GetTasks(task.Entity.ID, &TaskGetOptions{})

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(taskList).ShouldNot(BeNil())
			Expect(taskList.Items).Should(ContainElement(*task))

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetNetworks", func() {
		It("GetNetworks returns a completed task", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("GET_NETWORKS", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.GetNetworks(task.Entity.ID)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("GET_NETWORKS"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("AcquireFloatingIp", func() {
		It("Acquire floating IP for VM", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("ACQUIRE_FLOATING_IP", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			acquireFloatingIpSpec := &VmFloatingIpSpec{
				NetworkId: "networkId",
			}
			task, err = client.VMs.AcquireFloatingIp(task.Entity.ID, acquireFloatingIpSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("ReleaseFloatingIp", func() {
		It("Acquire floating IP for VM", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("RELEASE_FLOATING_IP", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err = client.VMs.ReleaseFloatingIp(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetMKSTicket", func() {
		It("GetMKSTicket returns a completed task", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			// Power on
			server.SetResponseJson(200, createMockTask("START_VM", "QUEUED"))
			powerOnTask, err := client.VMs.Start(task.Entity.ID)
			Expect(err).Should(BeNil())
			Expect(powerOnTask).ShouldNot(BeNil())
			server.SetResponseJson(200, createMockTask("START_VM", "COMPLETED"))
			powerOnTask, err = client.Tasks.Wait(powerOnTask.ID)
			Expect(err).Should(BeNil())
			Expect(powerOnTask).ShouldNot(BeNil())

			mockTask = createMockTask("GET_MKS_TICKET", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.GetMKSTicket(task.Entity.ID)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("GET_MKS_TICKET"))
			Expect(task.State).Should(Equal("COMPLETED"))

			// Power off
			server.SetResponseJson(200, createMockTask("STOP_VM", "QUEUED"))
			powerOffTask, err := client.VMs.Stop(task.Entity.ID)
			Expect(err).Should(BeNil())
			Expect(powerOffTask).ShouldNot(BeNil())
			server.SetResponseJson(200, createMockTask("STOP_VM", "COMPLETED"))
			powerOffTask, err = client.Tasks.Wait(powerOffTask.ID)
			Expect(err).Should(BeNil())
			Expect(powerOffTask).ShouldNot(BeNil())

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("SetTag", func() {
		It("SetTag returns a completed task", func() {
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateVM(projID, vmSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			// Set VM tag
			tag := "namespace:predicate=value"
			vmTags := &VmTag{Tag: tag}
			mockTask = createMockTask("ADD_TAG", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.SetTag(task.Entity.ID, vmTags)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("ADD_TAG"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockVm := &VM{Name: vmSpec.Name, Tags: []string{tag}}
			server.SetResponseJson(200, mockVm)
			vm, err := client.VMs.Get(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(vm).ShouldNot(BeNil())
			Expect(vm.Name).Should(Equal(vmSpec.Name))
			Expect(vm.Tags).Should(Equal([]string{tag}))

			mockTask = createMockTask("DELETE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.VMs.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("CreateImage", func() {
		var (
			vmID    string
			imageID string
		)

		BeforeEach(func() {
			vmID = ""
			imageID = ""

			// Create VM
			mockTask := createMockTask("CREATE_VM", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateVM(projID, vmSpec)
			Expect(err).Should(BeNil())

			task, err = client.Tasks.Wait(task.ID)
			if task != nil {
				vmID = task.Entity.ID
			}
			Expect(err).Should(BeNil())
		})

		AfterEach(func() {
			// Delete image
			if len(imageID) > 0 {
				mockTask := createMockTask("DELETE_IMAGE", "COMPLETED")
				server.SetResponseJson(200, mockTask)
				task, err := client.Images.Delete(imageID)
				task, err = client.Tasks.Wait(task.ID)
				if err != nil {
					GinkgoT().Log(err)
				}
			}

			// Delete VM
			if len(vmID) > 0 {
				mockTask := createMockTask("DELETE_VM", "COMPLETED")
				server.SetResponseJson(200, mockTask)
				task, err := client.VMs.Delete(vmID)
				task, err = client.Tasks.Wait(task.ID)
				if err != nil {
					GinkgoT().Log(err)
				}
			}
		})

		It("CreateImage succeeds", func() {
			// Create image from VM
			imageName := randomString(10, "go-sdk-image-")
			imageCreateOptions := &ImageCreateSpec{Name: imageName, ReplicationType: "ON_DEMAND"}
			mockTask := createMockTask("CREATE_VM_IMAGE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			createTask, err := client.VMs.CreateImage(vmID, imageCreateOptions)
			createTask, err = client.Tasks.Wait(createTask.ID)
			if createTask != nil {
				imageID = createTask.Entity.ID
			}

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(createTask).ShouldNot(BeNil())
			Expect(createTask.Operation).Should(Equal("CREATE_VM_IMAGE"))
			Expect(createTask.State).Should(Equal("COMPLETED"))

			// Check image created as expected
			server.SetResponseJson(200, Image{Name: imageName})
			image, err := client.Images.Get(createTask.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(image).ShouldNot(BeNil())
			Expect(image.ID).Should(Equal(createTask.Entity.ID))
			Expect(image.Name).Should(Equal(imageName))
		})
	})
})
