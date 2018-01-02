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
	"reflect"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/vmware/photon-controller-go-sdk/photon/internal/mocks"
)

var _ = Describe("Deployment", func() {
	var (
		server         *mocks.Server
		client         *Client
		deploymentSpec *DeploymentCreateSpec
	)

	BeforeEach(func() {
		if isIntegrationTest() {
			Skip("Skipping deployment test on integration mode. Need undeployed environment")
		}
		server, client = testSetup()
		deploymentSpec = &DeploymentCreateSpec{
			ImageDatastores:         []string{randomString(10, "go-sdk-deployment-")},
			UseImageDatastoreForVms: true,
			Auth: &AuthInfo{},
		}
	})

	AfterEach(func() {
		server.Close()
	})

	Describe("CreateGetAndDeleteDeployment", func() {
		It("Deployment create and delete succeeds", func() {
			if isIntegrationTest() {
				Skip("Skipping deployment test on integration mode. Need undeployed environment")
			}
			mockTask := createMockTask("CREATE_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Deployments.Create(deploymentSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("CREATE_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockDeployment := Deployment{
				ImageDatastores:         deploymentSpec.ImageDatastores,
				UseImageDatastoreForVms: deploymentSpec.UseImageDatastoreForVms,
				Auth:                 &AuthInfo{},
				NetworkConfiguration: &NetworkConfiguration{Enabled: false},
			}
			server.SetResponseJson(200, mockDeployment)
			deployment, err := client.Deployments.Get(task.Entity.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(deployment).ShouldNot(BeNil())
			Expect(deployment.ImageDatastores).Should(Equal(deploymentSpec.ImageDatastores))
			Expect(deployment.UseImageDatastoreForVms).Should(Equal(deploymentSpec.UseImageDatastoreForVms))
			Expect(deployment.ID).Should(Equal(task.Entity.ID))

			server.SetResponseJson(200, &Deployments{[]Deployment{mockDeployment}})
			deploymentList, err := client.Deployments.GetAll()
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(deploymentList).ShouldNot(BeNil())

			var found bool
			for _, d := range deploymentList.Items {
				if reflect.DeepEqual(d.ImageDatastores, deploymentSpec.ImageDatastores) {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Deployments.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("DELETE_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("GetDeployment", func() {
		It("Get Hosts succeeds", func() {
			mockTask := createMockTask("CREATE_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Deployments.Create(deploymentSpec)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			mockTask = createMockTask("CREATE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			hostSpec := &HostCreateSpec{
				Username: randomString(10),
				Password: randomString(10),
				Address:  randomAddress(),
				Tags:     []string{"CLOUD"},
				Metadata: map[string]string{"test": "go-sdk-host"},
			}

			mockTask = createMockTask("CREATE_HOST", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			hostTask, err := client.Hosts.Create(hostSpec, task.Entity.ID)
			hostTask, err = client.Tasks.Wait(hostTask.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())

			server.SetResponseJson(200, createMockHostsPage(Host{Tags: hostSpec.Tags, ID: task.Entity.ID}))
			hostList, err := client.Deployments.GetHosts(task.Entity.ID)

			var found bool
			for _, host := range hostList.Items {
				if host.ID == hostTask.Entity.ID {
					found = true
					break
				}
			}
			Expect(found).Should(BeTrue())

			mockTask = createMockTask("DELETE_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Deployments.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
		})

		Context("GetVM needs a vm", func() {
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

			It("Get VMs succeeds", func() {
				mockTask := createMockTask("CREATE_DEPLOYMENT", "COMPLETED")
				server.SetResponseJson(200, mockTask)

				task, err := client.Deployments.Create(deploymentSpec)
				task, err = client.Tasks.Wait(task.ID)
				GinkgoT().Log(err)
				Expect(err).Should(BeNil())

				mockTask = createMockTask("CREATE_VM", "COMPLETED")
				server.SetResponseJson(200, mockTask)
				vmTask, err := client.Projects.CreateVM(projID, vmSpec)
				vmTask, err = client.Tasks.Wait(vmTask.ID)
				GinkgoT().Log(err)
				Expect(err).Should(BeNil())

				server.SetResponseJson(200, createMockVmsPage(VM{Name: vmSpec.Name}))
				vmList, err := client.Deployments.GetVms(task.Entity.ID)
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
				task, err = client.VMs.Delete(vmTask.Entity.ID)
				task, err = client.Tasks.Wait(task.ID)

				GinkgoT().Log(err)
				Expect(err).Should(BeNil())

				mockTask = createMockTask("DELETE_DEPLOYMENT", "COMPLETED")
				server.SetResponseJson(200, mockTask)
				task, err = client.Deployments.Delete(task.Entity.ID)
				task, err = client.Tasks.Wait(task.ID)
				GinkgoT().Log(err)
				Expect(err).Should(BeNil())
			})
		})

	})

	Describe("DeployAndDestroyDeployment", func() {
		It("Deploy and Destroy a deployment succeeds", func() {
			if isIntegrationTest() {
				Skip("Skipping deployment test on integration mode. Need undeployed environment")
			}
			mockTask := createMockTask("PERFORM_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Deployments.Deploy("deployment-ID",
				&DeploymentDeployOperation{})
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("PERFORM_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("PERFORM_DELETE_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Deployments.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("PERFORM_DELETE_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("DeployAndDestroyDeploymentWithDesiredState", func() {
		It("Deploy and Destroy a deployment succeeds", func() {
			if isIntegrationTest() {
				Skip("Skipping deployment test on integration mode. Need undeployed environment")
			}
			mockTask := createMockTask("PERFORM_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Deployments.Deploy("deployment-ID",
				&DeploymentDeployOperation{
					DesiredState: "PAUSED",
				})
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("PERFORM_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("PERFORM_DELETE_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Deployments.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("PERFORM_DELETE_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("InitializeAndFinalizeMigrateDeployment", func() {
		It("Initialize and Finalize deployment migration succeeds", func() {
			if isIntegrationTest() {
				Skip("Skipping deployment test on integration mode. Need undeployed environment")
			}
			mockTask := createMockTask("PERFORM_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Deployments.Deploy("deployment-ID", &DeploymentDeployOperation{})
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("PERFORM_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("INITIALIZE_MIGRATE_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			initializeMigrationOperation := &InitializeMigrationOperation{SourceNodeGroupReference: "sourceAddr"}
			task, err = client.Deployments.InitializeDeploymentMigration(initializeMigrationOperation, "deployment-ID")
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("INITIALIZE_MIGRATE_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("FINALIZE_MIGRATE_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			finalizeMigrationOperation := &FinalizeMigrationOperation{SourceNodeGroupReference: "sourceAddr"}
			task, err = client.Deployments.FinalizeDeploymentMigration(finalizeMigrationOperation, "deployment-ID")
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("FINALIZE_MIGRATE_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("PERFORM_DELETE_DEPLOYMENT", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			task, err = client.Deployments.Delete(task.Entity.ID)
			task, err = client.Tasks.Wait(task.ID)
			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("PERFORM_DELETE_DEPLOYMENT"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("SetSecurityGroups", func() {
		It("sets security groups for a project", func() {
			mockTask := createMockTask("SET_SECURITY_GROUPS", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			// Set security groups for the project
			expected := &Deployment{
				Auth: &AuthInfo{
					SecurityGroups: []string{
						randomString(10),
						randomString(10),
					},
				},
			}

			payload := SecurityGroupsSpec{
				Items: expected.Auth.SecurityGroups,
			}
			updateTask, err := client.Deployments.SetSecurityGroups("deployment-ID", &payload)
			updateTask, err = client.Tasks.Wait(updateTask.ID)
			Expect(err).Should(BeNil())

			// Get the security groups for the project
			server.SetResponseJson(200, expected)
			deployment, err := client.Deployments.Get("deployment-ID")
			Expect(err).Should(BeNil())
			Expect(deployment.Auth.SecurityGroups).To(ContainElement(payload.Items[0]))
			Expect(deployment.Auth.SecurityGroups).To(ContainElement(payload.Items[1]))
		})
	})

	Describe("SetImageDatastores", func() {
		It("Succeeds", func() {
			mockTask := createMockTask("UPDATE_IMAGE_DATASTORES", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			imageDatastores := &ImageDatastores{
				[]string{"imageDatastore1", "imageDatastore2"},
			}
			createdTask, err := client.Deployments.SetImageDatastores("deploymentId", imageDatastores)
			createdTask, err = client.Tasks.Wait(createdTask.ID)

			Expect(err).Should(BeNil())
			Expect(createdTask.Operation).Should(Equal("UPDATE_IMAGE_DATASTORES"))
			Expect(createdTask.State).Should(Equal("COMPLETED"))
		})

		It("Fails", func() {
			mockApiError := createMockApiError("INVALID_IMAGE_DATASTORES", "Not a super set", 400)
			server.SetResponseJson(400, mockApiError)

			imageDatastores := &ImageDatastores{
				[]string{"imageDatastore1", "imageDatastore2"},
			}
			createdTask, err := client.Deployments.SetImageDatastores("deploymentId", imageDatastores)

			Expect(err).Should(Equal(*mockApiError))
			Expect(createdTask).Should(BeNil())
		})
	})

	Describe("SyncHostsConfig", func() {
		It("Sync Hosts Config succeeds", func() {
			mockTask := createMockTask("SYNC_HOSTS_CONFIG", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Deployments.SyncHostsConfig("deploymentId")
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("SYNC_HOSTS_CONFIG"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("PauseSystemAndPauseBackgroundTasks", func() {
		It("Pause System and Resume System succeeds", func() {
			mockTask := createMockTask("PAUSE_SYSTEM", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Deployments.PauseSystem("deploymentId")
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("PAUSE_SYSTEM"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("RESUME_SYSTEM", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err = client.Deployments.PauseBackgroundTasks("deploymentId")
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("RESUME_SYSTEM"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})

		It("Pause Background Tasks and Resume System succeeds", func() {
			mockTask := createMockTask("PAUSE_BACKGROUND_TASKS", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err := client.Deployments.PauseBackgroundTasks("deploymentId")
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("PAUSE_BACKGROUND_TASKS"))
			Expect(task.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("RESUME_SYSTEM", "COMPLETED")
			server.SetResponseJson(200, mockTask)

			task, err = client.Deployments.PauseBackgroundTasks("deploymentId")
			task, err = client.Tasks.Wait(task.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(task).ShouldNot(BeNil())
			Expect(task.Operation).Should(Equal("RESUME_SYSTEM"))
			Expect(task.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("EnableAndDisableServiceType", func() {
		It("Enable And Disable Service Type", func() {
			serviceType := "SWARM"
			serviceImageId := "testImageId"
			serviceConfigSpec := &ServiceConfigurationSpec{
				Type:    serviceType,
				ImageID: serviceImageId,
			}

			mockTask := createMockTask("CONFIGURE_SERVICE", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			enableTask, err := client.Deployments.EnableServiceType("deploymentId", serviceConfigSpec)
			enableTask, err = client.Tasks.Wait(enableTask.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(enableTask).ShouldNot(BeNil())
			Expect(enableTask.Operation).Should(Equal("CONFIGURE_SERVICE"))
			Expect(enableTask.State).Should(Equal("COMPLETED"))

			mockTask = createMockTask("DELETE_SERVICE_CONFIGURATION", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			disableTask, err := client.Deployments.DisableServiceType("deploymentId", serviceConfigSpec)
			disableTask, err = client.Tasks.Wait(disableTask.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(disableTask).ShouldNot(BeNil())
			Expect(disableTask.Operation).Should(Equal("DELETE_SERVICE_CONFIGURATION"))
			Expect(disableTask.State).Should(Equal("COMPLETED"))
		})
	})

	Describe("ConfigureNsx", func() {
		It("Configure NSX", func() {
			nsxAddress := "nsxAddress"
			nsxUsername := "nsxUsername"
			nsxPassword := "nsxPassword"

			nsxConfigSpec := &NsxConfigurationSpec{
				NsxAddress:  nsxAddress,
				NsxUsername: nsxUsername,
				NsxPassword: nsxPassword,
			}

			mockTask := createMockTask("CONFIGURE_NSX", "COMPLETED")
			server.SetResponseJson(200, mockTask)
			enableTask, err := client.Deployments.ConfigureNsx("deploymentId", nsxConfigSpec)
			enableTask, err = client.Tasks.Wait(enableTask.ID)

			GinkgoT().Log(err)
			Expect(err).Should(BeNil())
			Expect(enableTask).ShouldNot(BeNil())
			Expect(enableTask.Operation).Should(Equal("CONFIGURE_NSX"))
			Expect(enableTask.State).Should(Equal("COMPLETED"))
		})
	})
})
