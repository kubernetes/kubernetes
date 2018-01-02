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

var _ = Describe("Network", func() {
	var (
		server      *mocks.Server
		client      *Client
		networkSpec *NetworkCreateSpec
	)

	BeforeEach(func() {
		server, client = testSetup()
		networkSpec = &NetworkCreateSpec{
			Name:       randomString(10, "go-sdk-network-"),
			PortGroups: []string{"portGroup"},
		}
	})

	AfterEach(func() {
		cleanNetworks(client)
		server.Close()
	})

	Describe("CreateDeleteNetwork", func() {
		It("Network create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Networks.Create(networkSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_NETWORK"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Networks.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_NETWORK"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("GetNetwork", func() {
		It("Get network succeeds", func() {
			mockTask := createMockTask("CREATE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Networks.Create(networkSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, Network{Name: networkSpec.Name})
			network, err := client.Networks.Get(task.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(network).ShouldNot(BeNil())
			Expect(network.Name).Should(Equal(networkSpec.Name))

			mockTask = createMockTask("DELETE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Networks.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		It("GetAll Network succeeds", func() {
			mockTask := createMockTask("CREATE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Networks.Create(networkSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, createMockNetworksPage(Network{Name: networkSpec.Name}))
			networks, err := client.Networks.GetAll(&NetworkGetOptions{})

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(networks).ShouldNot(BeNil())

			var found bool
			for _, network := range networks.Items {
				if network.Name == networkSpec.Name && network.ID == task.Entity.ID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Networks.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		It("GetAll Network with optional name succeeds", func() {
			mockTask := createMockTask("CREATE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Networks.Create(networkSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, createMockNetworksPage(Network{Name: networkSpec.Name}))
			networks, err := client.Networks.GetAll(&NetworkGetOptions{Name: networkSpec.Name})

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(networks).ShouldNot(BeNil())

			var found bool
			for _, network := range networks.Items {
				if network.Name == networkSpec.Name && network.ID == task.Entity.ID {
					found = true
					break
				}
			}
			Expect(len(networks.Items)).Should(Equal(1))
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Networks.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("SetDefaultNetwork", func() {
		It("Set default network succeeds", func() {
			mockTask := createMockTask("SET_DEFAULT_NETWORK", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Networks.SetDefault("networkId")
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("SET_DEFAULT_NETWORK"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})
})
