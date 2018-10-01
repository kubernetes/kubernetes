// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package monitoredresource

import (
	"fmt"
	"os"
	"sync"
)

// Interface is a type that represent monitor resource that satisfies monitoredresource.Interface
type Interface interface {

	// MonitoredResource returns the resource type and resource labels.
	MonitoredResource() (resType string, labels map[string]string)
}

// GKEContainer represents gke_container type monitored resource.
// For definition refer to
// https://cloud.google.com/monitoring/api/resources#tag_gke_container
type GKEContainer struct {

	// ProjectID is the identifier of the GCP project associated with this resource, such as "my-project".
	ProjectID string

	// InstanceID is the numeric VM instance identifier assigned by Compute Engine.
	InstanceID string

	// ClusterName is the name for the cluster the container is running in.
	ClusterName string

	// ContainerName is the name of the container.
	ContainerName string

	// NamespaceID is the identifier for the cluster namespace the container is running in
	NamespaceID string

	// PodID is the identifier for the pod the container is running in.
	PodID string

	// Zone is the Compute Engine zone in which the VM is running.
	Zone string
}

// MonitoredResource returns resource type and resource labels for GKEContainer
func (gke *GKEContainer) MonitoredResource() (resType string, labels map[string]string) {
	labels = map[string]string{
		"project_id":     gke.ProjectID,
		"instance_id":    gke.InstanceID,
		"zone":           gke.Zone,
		"cluster_name":   gke.ClusterName,
		"container_name": gke.ContainerName,
		"namespace_id":   gke.NamespaceID,
		"pod_id":         gke.PodID,
	}
	return "gke_container", labels
}

// GCEInstance represents gce_instance type monitored resource.
// For definition refer to
// https://cloud.google.com/monitoring/api/resources#tag_gce_instance
type GCEInstance struct {

	// ProjectID is the identifier of the GCP project associated with this resource, such as "my-project".
	ProjectID string

	// InstanceID is the numeric VM instance identifier assigned by Compute Engine.
	InstanceID string

	// Zone is the Compute Engine zone in which the VM is running.
	Zone string
}

// MonitoredResource returns resource type and resource labels for GCEInstance
func (gce *GCEInstance) MonitoredResource() (resType string, labels map[string]string) {
	labels = map[string]string{
		"project_id":  gce.ProjectID,
		"instance_id": gce.InstanceID,
		"zone":        gce.Zone,
	}
	return "gce_instance", labels
}

// AWSEC2Instance represents aws_ec2_instance type monitored resource.
// For definition refer to
// https://cloud.google.com/monitoring/api/resources#tag_aws_ec2_instance
type AWSEC2Instance struct {

	// AWSAccount is the AWS account number for the VM.
	AWSAccount string

	// InstanceID is the instance id of the instance.
	InstanceID string

	// Region is the AWS region for the VM. The format of this field is "aws:{region}",
	// where supported values for {region} are listed at
	// http://docs.aws.amazon.com/general/latest/gr/rande.html.
	Region string
}

// MonitoredResource returns resource type and resource labels for AWSEC2Instance
func (aws *AWSEC2Instance) MonitoredResource() (resType string, labels map[string]string) {
	labels = map[string]string{
		"aws_account": aws.AWSAccount,
		"instance_id": aws.InstanceID,
		"region":      aws.Region,
	}
	return "aws_ec2_instance", labels
}

// Autodetect auto detects monitored resources based on
// the environment where the application is running.
// It supports detection of following resource types
// 1. gke_container:
// 2. gce_instance:
// 3. aws_ec2_instance:
//
// Returns MonitoredResInterface which implements getLabels() and getType()
// For resource definition go to https://cloud.google.com/monitoring/api/resources
func Autodetect() Interface {
	return func() Interface {
		var autoDetected Interface
		var awsIdentityDoc *awsIdentityDocument
		var gcpMetadata *gcpMetadata
		detectOnce.Do(func() {

			// First attempts to retrieve AWS Identity Doc and GCP metadata.
			// It then determines the resource type
			// In GCP and AWS environment both func finishes quickly. However,
			// in an environment other than those (e.g local laptop) it
			// takes 2 seconds for GCP and 5-6 for AWS.
			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				awsIdentityDoc = retrieveAWSIdentityDocument()
			}()
			go func() {
				defer wg.Done()
				gcpMetadata = retrieveGCPMetadata()
			}()

			wg.Wait()
			autoDetected = detectResourceType(awsIdentityDoc, gcpMetadata)
		})
		return autoDetected
	}()

}

// createAWSEC2InstanceMonitoredResource creates a aws_ec2_instance monitored resource
// awsIdentityDoc contains AWS EC2 specific attributes.
func createAWSEC2InstanceMonitoredResource(awsIdentityDoc *awsIdentityDocument) *AWSEC2Instance {
	awsInstance := AWSEC2Instance{
		AWSAccount: awsIdentityDoc.accountID,
		InstanceID: awsIdentityDoc.instanceID,
		Region:     fmt.Sprintf("aws:%s", awsIdentityDoc.region),
	}
	return &awsInstance
}

// createGCEInstanceMonitoredResource creates a gce_instance monitored resource
// gcpMetadata contains GCP (GKE or GCE) specific attributes.
func createGCEInstanceMonitoredResource(gcpMetadata *gcpMetadata) *GCEInstance {
	gceInstance := GCEInstance{
		ProjectID:  gcpMetadata.projectID,
		InstanceID: gcpMetadata.instanceID,
		Zone:       gcpMetadata.zone,
	}
	return &gceInstance
}

// createGKEContainerMonitoredResource creates a gke_container monitored resource
// gcpMetadata contains GCP (GKE or GCE) specific attributes.
func createGKEContainerMonitoredResource(gcpMetadata *gcpMetadata) *GKEContainer {
	gkeContainer := GKEContainer{
		ProjectID:     gcpMetadata.projectID,
		InstanceID:    gcpMetadata.instanceID,
		Zone:          gcpMetadata.zone,
		ContainerName: gcpMetadata.containerName,
		ClusterName:   gcpMetadata.clusterName,
		NamespaceID:   gcpMetadata.namespaceID,
		PodID:         gcpMetadata.podID,
	}
	return &gkeContainer
}

// detectOnce is used to make sure GCP and AWS metadata detect function executes only once.
var detectOnce sync.Once

// detectResourceType determines the resource type.
// awsIdentityDoc contains AWS EC2 attributes. nil if it is not AWS EC2 environment
// gcpMetadata contains GCP (GKE or GCE) specific attributes.
func detectResourceType(awsIdentityDoc *awsIdentityDocument, gcpMetadata *gcpMetadata) Interface {
	if os.Getenv("KUBERNETES_SERVICE_HOST") != "" &&
		gcpMetadata != nil && gcpMetadata.instanceID != "" {
		return createGKEContainerMonitoredResource(gcpMetadata)
	} else if gcpMetadata != nil && gcpMetadata.instanceID != "" {
		return createGCEInstanceMonitoredResource(gcpMetadata)
	} else if awsIdentityDoc != nil {
		return createAWSEC2InstanceMonitoredResource(awsIdentityDoc)
	}
	return nil
}
