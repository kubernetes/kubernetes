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

var _ = Describe("Service", func() {
	var (
		server                *mocks.Server
		client                *Client
		kubernetesServiceSpec *ServiceCreateSpec
		mesosServiceSpec      *ServiceCreateSpec
		tenantID              string
		resName               string
		projID                string
	)

	BeforeEach(func() {
		if isIntegrationTest() {
			Skip("Skipping service test on integration mode. Need to set extendedProperties to use real IPs and masks")
		}
		server, client = testSetup()
		tenantID = createTenant(server, client)
		resName = createResTicket(server, client, tenantID)
		projID = createProject(server, client, tenantID, resName)
		kubernetesMap := map[string]string{"dns": "1.1.1.1", "gateway": "1.1.1.2", "netmask": "255.255.255.128",
			"master_ip": "1.1.1.3", "container_network": "1.2.0.0/16"}
		kubernetesServiceSpec = &ServiceCreateSpec{
			Name:               randomString(10, "go-sdk-service-"),
			Type:               "KUBERNETES",
			WorkerCount:        2,
			BatchSizeWorker:    1,
			ExtendedProperties: kubernetesMap,
		}
		mesosMap := map[string]string{"dns": "1.1.1.1", "gateway": "1.1.1.2", "netmask": "255.255.255.128",
			"zookeeper_ip1": "1.1.1.4", "zookeeper_ip2": "1.1.1.5", "zookeeper_ip3": "1.1.1.6"}
		mesosServiceSpec = &ServiceCreateSpec{
			Name:               randomString(10, "go-sdk-service-"),
			Type:               "MESOS",
			WorkerCount:        2,
			BatchSizeWorker:    1,
			ExtendedProperties: mesosMap,
		}
	})

	AfterEach(func() {
		cleanServices(client, projID)
		cleanTenants(client)
		server.Close()
	})

	Describe("CreateDeleteService", func() {
		It("Kubernetes service create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateService(projID, kubernetesServiceSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_SERVICE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_SERVICE"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})

		It("Mesos s create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateService(projID, mesosServiceSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_SERVICE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_SERVICE"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("GetService", func() {
		It("Get service succeeds", func() {
			mockTask := createMockTask("CREATE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateService(projID, kubernetesServiceSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, Service{Name: kubernetesServiceSpec.Name})
			service, err := client.Services.Get(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(service).ShouldNot(BeNil())
			Expect(service.Name).Should(Equal(kubernetesServiceSpec.Name))

			mockTask = createMockTask("DELETE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetVMs", func() {
		It("Get vms succeeds", func() {
			serviceVMName := "MasterVM"
			mockTask := createMockTask("CREATE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateService(projID, kubernetesServiceSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockVm := VM{Name: serviceVMName}
			mockVmsPage := createMockVmsPage(mockVm)
			server.SetResponseJson(200, mockVmsPage)
			vmList, err := client.Services.GetVMs(projID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(vmList).ShouldNot(BeNil())

			var found bool
			for _, vm := range vmList.Items {
				if vm.Name == serviceVMName {
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

	Describe("Resize service", func() {
		It("Resize succeeds", func() {
			mockTask := createMockTask("CREATE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateService(projID, kubernetesServiceSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			serviceResize := &ServiceResizeOperation{NewWorkerCount: 3}
			mockTask = createMockTask("RESIZE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.Resize(task.Entity.ID, serviceResize)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("RESIZE_SERVICE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockService := &Service{Name: kubernetesServiceSpec.Name, WorkerCount: serviceResize.NewWorkerCount}
			server.SetResponseJson(200, mockService)
			service, err := client.Services.Get(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(service).ShouldNot(BeNil())
			Expect(service.Name).Should(Equal(kubernetesServiceSpec.Name))
			Expect(service.WorkerCount).Should(Equal(serviceResize.NewWorkerCount))

			mockTask = createMockTask("DELETE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("Trigger service maintenance", func() {
		It("Trigger service maintenance succeeds", func() {
			mockTask := createMockTask("CREATE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateService(projID, kubernetesServiceSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("TRIGGER_SERVICE_MAINTENANCE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.TriggerMaintenance(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("TRIGGER_SERVICE_MAINTENANCE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("Change version service", func() {
		It("Resize succeeds", func() {
			imageID := createImage(server, client)
			mockTask := createMockTask("CREATE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateService(projID, kubernetesServiceSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			changeVersion := &ServiceChangeVersionOperation{NewImageID: imageID}
			mockTask = createMockTask("CHANGE_VERSION_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.ChangeVersion(task.Entity.ID, changeVersion)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CHANGE_VERSION_SERVICE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Services.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})
})
