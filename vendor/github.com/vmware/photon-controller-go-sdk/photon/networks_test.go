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

var _ = Describe("Subnet", func() {
	var (
		server     *mocks.Server
		client     *Client
		subnetSpec *SubnetCreateSpec
	)

	BeforeEach(func() {
		server, client = testSetup()
		subnetSpec = &SubnetCreateSpec{
			Name:       randomString(10, "go-sdk-subnet-"),
			PortGroups: []string{"portGroup"},
		}
	})

	AfterEach(func() {
		cleanSubnets(client)
		server.Close()
	})

	Describe("CreateDeleteSubnet", func() {
		It("Subnet create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Subnets.Create(subnetSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_NETWORK"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Subnets.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_NETWORK"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("GetSubnet", func() {
		It("Get subnet succeeds", func() {
			mockTask := createMockTask("CREATE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Subnets.Create(subnetSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, Subnet{Name: subnetSpec.Name})
			subnet, err := client.Subnets.Get(task.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(subnet).ShouldNot(BeNil())
			Expect(subnet.Name).Should(Equal(subnetSpec.Name))

			mockTask = createMockTask("DELETE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Subnets.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		It("GetAll Subnet succeeds", func() {
			mockTask := createMockTask("CREATE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Subnets.Create(subnetSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, createMockSubnetsPage(Subnet{Name: subnetSpec.Name}))
			subnets, err := client.Subnets.GetAll(&SubnetGetOptions{})

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(subnets).ShouldNot(BeNil())

			var found bool
			for _, subnet := range subnets.Items {
				if subnet.Name == subnetSpec.Name && subnet.ID == task.Entity.ID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Subnets.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		It("GetAll Subnet with optional name succeeds", func() {
			mockTask := createMockTask("CREATE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Subnets.Create(subnetSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, createMockSubnetsPage(Subnet{Name: subnetSpec.Name}))
			subnets, err := client.Subnets.GetAll(&SubnetGetOptions{Name: subnetSpec.Name})

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(subnets).ShouldNot(BeNil())

			var found bool
			for _, subnet := range subnets.Items {
				if subnet.Name == subnetSpec.Name && subnet.ID == task.Entity.ID {
					found = true
					break
				}
			}
			Expect(len(subnets.Items)).Should(Equal(1))
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Subnets.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("SetDefaultSubnet", func() {
		It("Set default subnet succeeds", func() {
			mockTask := createMockTask("SET_DEFAULT_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Subnets.SetDefault("subnetId")
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("SET_DEFAULT_NETWORK"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})
})
