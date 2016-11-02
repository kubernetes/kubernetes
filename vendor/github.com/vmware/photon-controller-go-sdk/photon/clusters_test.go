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

var _ = Describe("Cluster", func() {
	var (
		server                *mocks.Server
		client                *Client
		kubernetesClusterSpec *ClusterCreateSpec
		mesosClusterSpec      *ClusterCreateSpec
		tenantID              string
		resName               string
		projID                string
	)

	BeforeEach(func() {
		if isIntegrationTest() {
			Skip("Skipping cluster test on integration mode. Need to set extendedProperties to use real IPs and masks")
		}
		server, client = testSetup()
		tenantID = createTenant(server, client)
		resName = createResTicket(server, client, tenantID)
		projID = createProject(server, client, tenantID, resName)
		kubernetesMap := map[string]string{"dns": "1.1.1.1", "gateway": "1.1.1.2", "netmask": "255.255.255.128",
			"master_ip": "1.1.1.3", "container_network": "1.2.0.0/16"}
		kubernetesClusterSpec = &ClusterCreateSpec{
			Name:               randomString(10, "go-sdk-cluster-"),
			Type:               "KUBERNETES",
			WorkerCount:        2,
			BatchSizeWorker:    1,
			ExtendedProperties: kubernetesMap,
		}
		mesosMap := map[string]string{"dns": "1.1.1.1", "gateway": "1.1.1.2", "netmask": "255.255.255.128",
			"zookeeper_ip1": "1.1.1.4", "zookeeper_ip2": "1.1.1.5", "zookeeper_ip3": "1.1.1.6"}
		mesosClusterSpec = &ClusterCreateSpec{
			Name:               randomString(10, "go-sdk-cluster-"),
			Type:               "MESOS",
			WorkerCount:        2,
			BatchSizeWorker:    1,
			ExtendedProperties: mesosMap,
		}
	})

	AfterEach(func() {
		cleanClusters(client, projID)
		cleanTenants(client)
		server.Close()
	})

	Describe("CreateDeleteCluster", func() {
		It("Kubernetes cluster create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateCluster(projID, kubernetesClusterSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_CLUSTER"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Clusters.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_CLUSTER"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})

		It("Mesos cluster create and delete succeeds", func() {
			mockTask := createMockTask("CREATE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateCluster(projID, mesosClusterSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_CLUSTER"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Clusters.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_CLUSTER"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("GetCluster", func() {
		It("Get cluster succeeds", func() {
			mockTask := createMockTask("CREATE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateCluster(projID, kubernetesClusterSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, Cluster{Name: kubernetesClusterSpec.Name})
			cluster, err := client.Clusters.Get(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(cluster).ShouldNot(BeNil())
			Expect(cluster.Name).Should(Equal(kubernetesClusterSpec.Name))

			mockTask = createMockTask("DELETE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Clusters.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("GetVMs", func() {
		It("Get vms succeeds", func() {
			clusterVMName := "MasterVM"
			mockTask := createMockTask("CREATE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Projects.CreateCluster(projID, kubernetesClusterSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockVm := VM{Name: clusterVMName}
			mockVmsPage := createMockVmsPage(mockVm)
			server.SetResponseJson(200, mockVmsPage)
			vmList, err := client.Clusters.GetVMs(projID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(vmList).ShouldNot(BeNil())

			var found bool
			for _, vm := range vmList.Items {
				if vm.Name == clusterVMName {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Clusters.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("Resize cluster", func() {
		It("Resize succeeds", func() {
			mockTask := createMockTask("CREATE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateCluster(projID, kubernetesClusterSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			clusterResize := &ClusterResizeOperation{NewWorkerCount: 3}
			mockTask = createMockTask("RESIZE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Clusters.Resize(task.Entity.ID, clusterResize)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("RESIZE_CLUSTER"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockCluster := &Cluster{Name: kubernetesClusterSpec.Name, WorkerCount: clusterResize.NewWorkerCount}
			server.SetResponseJson(200, mockCluster)
			cluster, err := client.Clusters.Get(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(cluster).ShouldNot(BeNil())
			Expect(cluster.Name).Should(Equal(kubernetesClusterSpec.Name))
			Expect(cluster.WorkerCount).Should(Equal(clusterResize.NewWorkerCount))

			mockTask = createMockTask("DELETE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Clusters.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})

	Describe("Trigger cluster maintenance", func() {
		It("Trigger cluster maintenance succeeds", func() {
			mockTask := createMockTask("CREATE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err := client.Projects.CreateCluster(projID, kubernetesClusterSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("TRIGGER_CLUSTER_MAINTENANCE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Clusters.TriggerMaintenance(task.Entity.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("TRIGGER_CLUSTER_MAINTENANCE"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_CLUSTER", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Clusters.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})
	})
})
