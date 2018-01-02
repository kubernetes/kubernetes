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
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"github.com/vmware/photon-controller-go-sdk/photon/internal/mocks"
)

var _ = ginkgo.Describe("VirtualSubnet", func() {
	var (
		server      *mocks.Server
		client      *Client
		networkSpec *VirtualSubnetCreateSpec
	)

	var projectId = "project1"

	ginkgo.BeforeEach(func() {
		server, client = testSetup()
		networkSpec = &VirtualSubnetCreateSpec{
			Name:                 randomString(10, "go-sdk-virtual-network-"),
			Description:          "a test virtual network",
			RoutingType:          "ROUTED",
			Size:                 256,
			ReservedStaticIpSize: 20,
		}
	})

	ginkgo.AfterEach(func() {
		cleanVirtualSubnets(client, projectId)
		server.Close()
	})

	ginkgo.Describe("CreateDeleteVirtualSubnet", func() {
		ginkgo.It("Virtual subnet create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_VIRTUAL_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.VirtualSubnets.Create(projectId, networkSpec)
			task, err = client.Tasks.Wait(task.ID)
			ginkgo.GinkgoT().Log(err)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(task).ShouldNot(gomega.BeNil())
			gomega.Expect(task.Operation).Should(gomega.Equal("CREATE_VIRTUAL_NETWORK"))
			gomega.Expect(task.State).Should(gomega.Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_VIRTUAL_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			ginkgo.GinkgoT().Log(err)

			task, err = client.VirtualSubnets.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(task).ShouldNot(gomega.BeNil())
			gomega.Expect(task.Operation).Should(gomega.Equal("DELETE_VIRTUAL_NETWORK"))
			gomega.Expect(task.State).Should(gomega.Equal("COMPLETED"))
		})
	})

	ginkgo.Describe("GetVirtualSubnet", func() {
		ginkgo.It("Get virtual subnet succeeds", func() {
			mockTask := createMockTask("CREATE_VIRTUAL_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.VirtualSubnets.Create(projectId, networkSpec)
			task, err = client.Tasks.Wait(task.ID)
			ginkgo.GinkgoT().Log(err)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(task).ShouldNot(gomega.BeNil())
			gomega.Expect(task.Operation).Should(gomega.Equal("CREATE_VIRTUAL_NETWORK"))
			gomega.Expect(task.State).Should(gomega.Equal("COMPLETED"))

			server.SetResponseJson(200, VirtualSubnet{Name: networkSpec.Name})
			subnet, err := client.VirtualSubnets.Get(task.Entity.ID)
			ginkgo.GinkgoT().Log(err)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(subnet).ShouldNot(gomega.BeNil())
			gomega.Expect(subnet.Name).Should(gomega.Equal(networkSpec.Name))

			mockTask = createMockTask("DELETE_VIRTUAL_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			ginkgo.GinkgoT().Log(err)

			task, err = client.VirtualSubnets.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(task).ShouldNot(gomega.BeNil())
			gomega.Expect(task.Operation).Should(gomega.Equal("DELETE_VIRTUAL_NETWORK"))
			gomega.Expect(task.State).Should(gomega.Equal("COMPLETED"))
		})

		ginkgo.It("Get all virtual subnets succeeds", func() {
			mockTask := createMockTask("CREATE_VIRTUAL_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.VirtualSubnets.Create(projectId, networkSpec)
			task, err = client.Tasks.Wait(task.ID)
			ginkgo.GinkgoT().Log(err)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(task).ShouldNot(gomega.BeNil())
			gomega.Expect(task.Operation).Should(gomega.Equal("CREATE_VIRTUAL_NETWORK"))
			gomega.Expect(task.State).Should(gomega.Equal("COMPLETED"))

			server.SetResponseJson(200, createMockVirtualSubnetsPage(VirtualSubnet{Name: networkSpec.Name}))
			subnets, err := client.VirtualSubnets.GetAll(projectId, &VirtualSubnetGetOptions{})
			ginkgo.GinkgoT().Log(err)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(subnets).ShouldNot(gomega.BeNil())

			var found bool
			for _, subnet := range subnets.Items {
				if subnet.Name == networkSpec.Name && subnet.ID == task.Entity.ID {
					found = true
					break
				}
			}
			gomega.Expect(found).Should(gomega.BeTrue())

			mockTask = createMockTask("DELETE_VIRTUAL_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			ginkgo.GinkgoT().Log(err)

			task, err = client.VirtualSubnets.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(task).ShouldNot(gomega.BeNil())
			gomega.Expect(task.Operation).Should(gomega.Equal("DELETE_VIRTUAL_NETWORK"))
			gomega.Expect(task.State).Should(gomega.Equal("COMPLETED"))
		})

		ginkgo.It("Get virtual subnet with the given name succeeds", func() {
			mockTask := createMockTask("CREATE_VIRTUAL_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.VirtualSubnets.Create(projectId, networkSpec)
			task, err = client.Tasks.Wait(task.ID)
			ginkgo.GinkgoT().Log(err)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(task).ShouldNot(gomega.BeNil())
			gomega.Expect(task.Operation).Should(gomega.Equal("CREATE_VIRTUAL_NETWORK"))
			gomega.Expect(task.State).Should(gomega.Equal("COMPLETED"))

			server.SetResponseJson(200, createMockVirtualSubnetsPage(VirtualSubnet{Name: networkSpec.Name}))
			subnets, err := client.VirtualSubnets.GetAll(projectId, &VirtualSubnetGetOptions{Name: networkSpec.Name})
			ginkgo.GinkgoT().Log(err)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(subnets).ShouldNot(gomega.BeNil())

			var found bool
			for _, subnet := range subnets.Items {
				if subnet.Name == networkSpec.Name && subnet.ID == task.Entity.ID {
					found = true
					break
				}
			}
			gomega.Expect(found).Should(gomega.BeTrue())

			mockTask = createMockTask("DELETE_VIRTUAL_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			ginkgo.GinkgoT().Log(err)

			task, err = client.VirtualSubnets.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(task).ShouldNot(gomega.BeNil())
			gomega.Expect(task.Operation).Should(gomega.Equal("DELETE_VIRTUAL_NETWORK"))
			gomega.Expect(task.State).Should(gomega.Equal("COMPLETED"))
		})
	})

	ginkgo.Describe("Set default subnet", func() {
		ginkgo.It("Set default virtual subnet succeeds", func() {
			mockTask := createMockTask("SET_DEFAULT_VIRTUAL_SUBNET", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.VirtualSubnets.SetDefault("subnetId")
			ginkgo.GinkgoT().Log(err)

			gomega.Expect(err).Should(gomega.BeNil())
			gomega.Expect(task).ShouldNot(gomega.BeNil())
			gomega.Expect(task.Operation).Should(gomega.Equal("SET_DEFAULT_VIRTUAL_SUBNET"))
			gomega.Expect(task.State).Should(gomega.Equal("COMPLETED"))
		})
	})
})
