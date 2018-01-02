//
// Copyright (c) 2016 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package cmds

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"

	client "github.com/heketi/heketi/client/api/go-client"
	"github.com/heketi/heketi/pkg/db"
	"github.com/heketi/heketi/pkg/glusterfs/api"
	"github.com/spf13/cobra"

	kubeapi "k8s.io/kubernetes/pkg/api/v1"
	batch "k8s.io/kubernetes/pkg/apis/batch/v1"
)

type KubeList struct {
	APIVersion string        `json:"apiVersion"`
	Kind       string        `json:"kind"`
	Items      []interface{} `json:"items"`
}

const (
	HeketiStorageJobName      = "heketi-storage-copy-job"
	HeketiStorageEndpointName = "heketi-storage-endpoints"
	HeketiStorageSecretName   = "heketi-storage-secret"
	HeketiStorageVolTagName   = "heketi-storage"

	HeketiStorageVolumeSize    = 2
	HeketiStorageVolumeSizeStr = "2Gi"
)

var (
	HeketiStorageJobContainer string
	heketiStorageListFilename string
)

func init() {
	RootCmd.AddCommand(setupHeketiStorageCommand)
	setupHeketiStorageCommand.Flags().StringVar(&heketiStorageListFilename,
		"listfile",
		"heketi-storage.json",
		"Filename to contain list of objects")
	setupHeketiStorageCommand.Flags().StringVar(&HeketiStorageJobContainer,
		"image",
		"heketi/heketi:dev",
		"container image to run this job")
}

func saveJson(i interface{}, filename string) error {

	// Open File
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Marshal struct to JSON
	data, err := json.MarshalIndent(i, "", "  ")
	if err != nil {
		return err
	}

	// Save data to file
	_, err = f.Write(data)
	if err != nil {
		return err
	}

	return nil
}

func createHeketiStorageVolume(c *client.Client) (*api.VolumeInfoResponse, error) {

	// Make sure the volume does not already exist on any cluster
	clusters, err := c.ClusterList()
	if err != nil {
		return nil, err
	}

	// Go through all the clusters checking volumes
	for _, clusterId := range clusters.Clusters {
		cluster, err := c.ClusterInfo(clusterId)
		if err != nil {
			return nil, err
		}

		// Go through all the volumes checking the names
		for _, volumeId := range cluster.Volumes {
			volume, err := c.VolumeInfo(volumeId)
			if err != nil {
				return nil, err
			}

			// Check volume name
			if volume.Name == db.HeketiStorageVolumeName {
				return nil, fmt.Errorf("Volume %v alreay exists", db.HeketiStorageVolumeName)
			}
		}
	}

	// Create request
	req := &api.VolumeCreateRequest{}
	req.Size = HeketiStorageVolumeSize
	req.Durability.Type = api.DurabilityReplicate
	req.Durability.Replicate.Replica = 3
	req.Name = db.HeketiStorageVolumeName

	// Create volume
	volume, err := c.VolumeCreate(req)
	if err != nil {
		return nil, err
	}

	return volume, nil
}

func createHeketiSecretFromDb(c *client.Client) (*kubeapi.Secret, error) {
	var dbfile bytes.Buffer

	// Save db
	err := c.BackupDb(&dbfile)
	if err != nil {
		return nil, fmt.Errorf("ERROR: %v\nUnable to get database from Heketi server", err.Error())
	}

	// Create Secret
	secret := &kubeapi.Secret{}
	secret.Kind = "Secret"
	secret.APIVersion = "v1"
	secret.ObjectMeta.Name = HeketiStorageSecretName
	secret.ObjectMeta.Labels = map[string]string{
		"deploy-heketi": "support",
	}
	secret.Data = make(map[string][]byte)
	secret.Data["heketi.db"] = dbfile.Bytes()

	return secret, nil
}

func createHeketiEndpointService() *kubeapi.Service {

	service := &kubeapi.Service{}
	service.Kind = "Service"
	service.APIVersion = "v1"
	service.ObjectMeta.Name = HeketiStorageEndpointName
	service.Spec.Ports = []kubeapi.ServicePort{
		kubeapi.ServicePort{
			Port: 1,
		},
	}

	return service
}

func createHeketiStorageEndpoints(c *client.Client,
	volume *api.VolumeInfoResponse) *kubeapi.Endpoints {

	endpoint := &kubeapi.Endpoints{}
	endpoint.Kind = "Endpoints"
	endpoint.APIVersion = "v1"
	endpoint.ObjectMeta.Name = HeketiStorageEndpointName

	// Initialize slices
	endpoint.Subsets = make([]kubeapi.EndpointSubset,
		len(volume.Mount.GlusterFS.Hosts))

	// Save all nodes in the endpoints
	for n, host := range volume.Mount.GlusterFS.Hosts {

		// Set Hostname/IP
		endpoint.Subsets[n].Addresses = []kubeapi.EndpointAddress{
			kubeapi.EndpointAddress{
				IP: host,
			},
		}

		// Set to port 1
		endpoint.Subsets[n].Ports = []kubeapi.EndpointPort{
			kubeapi.EndpointPort{
				Port: 1,
			},
		}
	}

	return endpoint
}

func createHeketiCopyJob(volume *api.VolumeInfoResponse) *batch.Job {
	job := &batch.Job{}
	job.Kind = "Job"
	job.APIVersion = "batch/v1"
	job.ObjectMeta.Name = HeketiStorageJobName
	job.ObjectMeta.Labels = map[string]string{
		"deploy-heketi": "support",
	}

	var (
		p int32 = 1
		c int32 = 1
	)
	job.Spec.Parallelism = &p
	job.Spec.Completions = &c
	job.Spec.Template.ObjectMeta.Name = HeketiStorageJobName
	job.Spec.Template.Spec.Volumes = []kubeapi.Volume{
		kubeapi.Volume{
			Name: HeketiStorageVolTagName,
			VolumeSource: kubeapi.VolumeSource{
				Glusterfs: &kubeapi.GlusterfsVolumeSource{
					EndpointsName: HeketiStorageEndpointName,
					Path:          volume.Name,
				},
			},
		},
		kubeapi.Volume{
			Name: HeketiStorageSecretName,
			VolumeSource: kubeapi.VolumeSource{
				Secret: &kubeapi.SecretVolumeSource{
					SecretName: HeketiStorageSecretName,
				},
			},
		},
	}

	job.Spec.Template.Spec.Containers = []kubeapi.Container{
		kubeapi.Container{
			Name:  "heketi",
			Image: HeketiStorageJobContainer,
			Command: []string{
				"cp",
				"/db/heketi.db",
				"/heketi",
			},
			VolumeMounts: []kubeapi.VolumeMount{
				kubeapi.VolumeMount{
					Name:      HeketiStorageVolTagName,
					MountPath: "/heketi",
				},
				kubeapi.VolumeMount{
					Name:      HeketiStorageSecretName,
					MountPath: "/db",
				},
			},
		},
	}
	job.Spec.Template.Spec.RestartPolicy = kubeapi.RestartPolicyNever

	return job
}

var setupHeketiStorageCommand = &cobra.Command{
	Use:   "setup-openshift-heketi-storage",
	Short: "Setup OpenShift/Kubernetes persistent storage for Heketi",
	Long: "Creates a dedicated GlusterFS volume for Heketi.\n" +
		"Once the volume is created, a Kubernetes/OpenShift\n" +
		"list object is created to configure the volume.\n",
	RunE: func(cmd *cobra.Command, args []string) (e error) {

		// Initialize Kubernetes List object
		list := &KubeList{}
		list.APIVersion = "v1"
		list.Kind = "List"
		list.Items = make([]interface{}, 0)

		// Create client
		c := client.NewClient(options.Url, options.User, options.Key)

		// Create volume
		volume, err := createHeketiStorageVolume(c)
		if err != nil {
			return err
		}

		// Cleanup volume on error
		defer func() {
			if e != nil {
				fmt.Fprintln(stderr, "Cleaning up")
				c.VolumeDelete(volume.Id)
			}
		}()

		// Create secret
		secret, err := createHeketiSecretFromDb(c)
		if err != nil {
			return err
		}
		list.Items = append(list.Items, secret)

		// Create endpoints
		endpoints := createHeketiStorageEndpoints(c, volume)
		list.Items = append(list.Items, endpoints)

		// Create service for the endpoints
		service := createHeketiEndpointService()
		list.Items = append(list.Items, service)

		// Create Job which copies db
		job := createHeketiCopyJob(volume)
		list.Items = append(list.Items, job)

		// Save list
		fmt.Fprintf(stdout, "Saving %v\n", heketiStorageListFilename)
		err = saveJson(list, heketiStorageListFilename)
		if err != nil {
			return err
		}

		return nil
	},
}
