/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gce

import (
	"fmt"
	"net/http"
	"os/exec"
	"regexp"
	"strings"
	"time"

	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/test/e2e/framework"
)

func init() {
	framework.RegisterProvider("gce", factory)
	framework.RegisterProvider("gke", factory)
}

func factory() (framework.ProviderInterface, error) {
	framework.Logf("Fetching cloud provider for %q\r", framework.TestContext.Provider)
	zone := framework.TestContext.CloudConfig.Zone
	region := framework.TestContext.CloudConfig.Region

	var err error
	if region == "" {
		region, err = gcecloud.GetGCERegion(zone)
		if err != nil {
			return nil, fmt.Errorf("error parsing GCE/GKE region from zone %q: %v", zone, err)
		}
	}
	managedZones := []string{} // Manage all zones in the region
	if !framework.TestContext.CloudConfig.MultiZone {
		managedZones = []string{zone}
	}

	gceCloud, err := gcecloud.CreateGCECloud(&gcecloud.CloudConfig{
		APIEndpoint:        framework.TestContext.CloudConfig.ApiEndpoint,
		ProjectID:          framework.TestContext.CloudConfig.ProjectID,
		Region:             region,
		Zone:               zone,
		ManagedZones:       managedZones,
		NetworkName:        "", // TODO: Change this to use framework.TestContext.CloudConfig.Network?
		SubnetworkName:     "",
		NodeTags:           nil,
		NodeInstancePrefix: "",
		TokenSource:        nil,
		UseMetadataServer:  false,
		AlphaFeatureGate:   gcecloud.NewAlphaFeatureGate([]string{}),
	})

	if err != nil {
		return nil, fmt.Errorf("Error building GCE/GKE provider: %v", err)
	}

	// Arbitrarily pick one of the zones we have nodes in
	if framework.TestContext.CloudConfig.Zone == "" && framework.TestContext.CloudConfig.MultiZone {
		zones, err := gceCloud.GetAllZonesFromCloudProvider()
		if err != nil {
			return nil, err
		}

		framework.TestContext.CloudConfig.Zone, _ = zones.PopAny()
	}

	return NewProvider(gceCloud), nil
}

func NewProvider(gceCloud *gcecloud.Cloud) framework.ProviderInterface {
	return &Provider{
		gceCloud: gceCloud,
	}
}

type Provider struct {
	framework.NullProvider
	gceCloud *gcecloud.Cloud
}

func (p *Provider) ResizeGroup(group string, size int32) error {
	// TODO: make this hit the compute API directly instead of shelling out to gcloud.
	// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
	zone, err := getGCEZoneForGroup(group)
	if err != nil {
		return err
	}
	output, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "resize",
		group, fmt.Sprintf("--size=%v", size),
		"--project="+framework.TestContext.CloudConfig.ProjectID, "--zone="+zone).CombinedOutput()
	if err != nil {
		return fmt.Errorf("Failed to resize node instance group %s: %s", group, output)
	}
	return nil
}

func (p *Provider) GetGroupNodes(group string) ([]string, error) {
	// TODO: make this hit the compute API directly instead of shelling out to gcloud.
	// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
	zone, err := getGCEZoneForGroup(group)
	if err != nil {
		return nil, err
	}
	output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
		"list-instances", group, "--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+zone).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("Failed to get nodes in instance group %s: %s", group, output)
	}
	re := regexp.MustCompile(".*RUNNING")
	lines := re.FindAllString(string(output), -1)
	for i, line := range lines {
		lines[i] = line[:strings.Index(line, " ")]
	}
	return lines, nil
}

func (p *Provider) GroupSize(group string) (int, error) {
	// TODO: make this hit the compute API directly instead of shelling out to gcloud.
	// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
	zone, err := getGCEZoneForGroup(group)
	if err != nil {
		return -1, err
	}
	output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
		"list-instances", group, "--project="+framework.TestContext.CloudConfig.ProjectID,
		"--zone="+zone).CombinedOutput()
	if err != nil {
		return -1, fmt.Errorf("Failed to get group size for group %s: %s", group, output)
	}
	re := regexp.MustCompile("RUNNING")
	return len(re.FindAllString(string(output), -1)), nil
}

func (p *Provider) EnsureLoadBalancerResourcesDeleted(ip, portRange string) error {
	project := framework.TestContext.CloudConfig.ProjectID
	region, err := gcecloud.GetGCERegion(framework.TestContext.CloudConfig.Zone)
	if err != nil {
		return fmt.Errorf("could not get region for zone %q: %v", framework.TestContext.CloudConfig.Zone, err)
	}

	return wait.Poll(10*time.Second, 5*time.Minute, func() (bool, error) {
		service := p.gceCloud.ComputeServices().GA
		list, err := service.ForwardingRules.List(project, region).Do()
		if err != nil {
			return false, err
		}
		for _, item := range list.Items {
			if item.PortRange == portRange && item.IPAddress == ip {
				framework.Logf("found a load balancer: %v", item)
				return false, nil
			}
		}
		return true, nil
	})
}

func getGCEZoneForGroup(group string) (string, error) {
	zone := framework.TestContext.CloudConfig.Zone
	if framework.TestContext.CloudConfig.MultiZone {
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "list",
			"--project="+framework.TestContext.CloudConfig.ProjectID, "--format=value(zone)", "--filter=name="+group).CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("Failed to get zone for node group %s: %s", group, output)
		}
		zone = strings.TrimSpace(string(output))
	}
	return zone, nil
}

func (p *Provider) CreatePD(zone string) (string, error) {
	pdName := fmt.Sprintf("%s-%s", framework.TestContext.Prefix, string(uuid.NewUUID()))

	if zone == "" && framework.TestContext.CloudConfig.MultiZone {
		zones, err := p.gceCloud.GetAllZonesFromCloudProvider()
		if err != nil {
			return "", err
		}
		zone, _ = zones.PopAny()
	}

	tags := map[string]string{}
	if err := p.gceCloud.CreateDisk(pdName, gcecloud.DiskTypeStandard, zone, 2 /* sizeGb */, tags); err != nil {
		return "", err
	}
	return pdName, nil
}

func (p *Provider) DeletePD(pdName string) error {
	err := p.gceCloud.DeleteDisk(pdName)

	if err != nil {
		if gerr, ok := err.(*googleapi.Error); ok && len(gerr.Errors) > 0 && gerr.Errors[0].Reason == "notFound" {
			// PD already exists, ignore error.
			return nil
		}

		framework.Logf("error deleting PD %q: %v", pdName, err)
	}
	return err
}

func (p *Provider) CreatePVSource(zone, diskName string) (*v1.PersistentVolumeSource, error) {
	return &v1.PersistentVolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   diskName,
			FSType:   "ext3",
			ReadOnly: false,
		},
	}, nil
}

func (p *Provider) DeletePVSource(pvSource *v1.PersistentVolumeSource) error {
	return framework.DeletePDWithRetry(pvSource.GCEPersistentDisk.PDName)
}

// CleanupResources cleans up GCE Service Type=LoadBalancer resources with
// the given name. The name is usually the UUID of the Service prefixed with an
// alpha-numeric character ('a') to work around cloudprovider rules.
func (p *Provider) CleanupServiceResources(c clientset.Interface, loadBalancerName, region, zone string) {
	if pollErr := wait.Poll(5*time.Second, framework.LoadBalancerCleanupTimeout, func() (bool, error) {
		if err := p.cleanupGCEResources(c, loadBalancerName, region, zone); err != nil {
			framework.Logf("Still waiting for glbc to cleanup: %v", err)
			return false, nil
		}
		return true, nil
	}); pollErr != nil {
		framework.Failf("Failed to cleanup service GCE resources.")
	}
}

func (p *Provider) cleanupGCEResources(c clientset.Interface, loadBalancerName, region, zone string) (retErr error) {
	if region == "" {
		// Attempt to parse region from zone if no region is given.
		var err error
		region, err = gcecloud.GetGCERegion(zone)
		if err != nil {
			return fmt.Errorf("error parsing GCE/GKE region from zone %q: %v", zone, err)
		}
	}
	if err := p.gceCloud.DeleteFirewall(gcecloud.MakeFirewallName(loadBalancerName)); err != nil &&
		!IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
		retErr = err
	}
	if err := p.gceCloud.DeleteRegionForwardingRule(loadBalancerName, region); err != nil &&
		!IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
		retErr = fmt.Errorf("%v\n%v", retErr, err)

	}
	if err := p.gceCloud.DeleteRegionAddress(loadBalancerName, region); err != nil &&
		!IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
		retErr = fmt.Errorf("%v\n%v", retErr, err)
	}
	clusterID, err := GetClusterID(c)
	if err != nil {
		retErr = fmt.Errorf("%v\n%v", retErr, err)
		return
	}
	hcNames := []string{gcecloud.MakeNodesHealthCheckName(clusterID)}
	hc, getErr := p.gceCloud.GetHTTPHealthCheck(loadBalancerName)
	if getErr != nil && !IsGoogleAPIHTTPErrorCode(getErr, http.StatusNotFound) {
		retErr = fmt.Errorf("%v\n%v", retErr, getErr)
		return
	}
	if hc != nil {
		hcNames = append(hcNames, hc.Name)
	}
	if err := p.gceCloud.DeleteExternalTargetPoolAndChecks(&v1.Service{}, loadBalancerName, region, clusterID, hcNames...); err != nil &&
		!IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
		retErr = fmt.Errorf("%v\n%v", retErr, err)
	}
	return
}

func (p *Provider) LoadBalancerSrcRanges() []string {
	return gcecloud.LoadBalancerSrcRanges()
}

func (p *Provider) EnableAndDisableInternalLB() (enable, disable func(svc *v1.Service)) {
	enable = func(svc *v1.Service) {
		svc.ObjectMeta.Annotations = map[string]string{gcecloud.ServiceAnnotationLoadBalancerType: string(gcecloud.LBTypeInternal)}
	}
	disable = func(svc *v1.Service) {
		delete(svc.ObjectMeta.Annotations, gcecloud.ServiceAnnotationLoadBalancerType)
	}
	return
}

// GetInstanceTags gets tags from GCE instance with given name.
func GetInstanceTags(cloudConfig framework.CloudConfig, instanceName string) *compute.Tags {
	gceCloud := cloudConfig.Provider.(*Provider).gceCloud
	res, err := gceCloud.ComputeServices().GA.Instances.Get(cloudConfig.ProjectID, cloudConfig.Zone,
		instanceName).Do()
	if err != nil {
		framework.Failf("Failed to get instance tags for %v: %v", instanceName, err)
	}
	return res.Tags
}

// SetInstanceTags sets tags on GCE instance with given name.
func SetInstanceTags(cloudConfig framework.CloudConfig, instanceName, zone string, tags []string) []string {
	gceCloud := cloudConfig.Provider.(*Provider).gceCloud
	// Re-get instance everytime because we need the latest fingerprint for updating metadata
	resTags := GetInstanceTags(cloudConfig, instanceName)
	_, err := gceCloud.ComputeServices().GA.Instances.SetTags(
		cloudConfig.ProjectID, zone, instanceName,
		&compute.Tags{Fingerprint: resTags.Fingerprint, Items: tags}).Do()
	if err != nil {
		framework.Failf("failed to set instance tags: %v", err)
	}
	framework.Logf("Sent request to set tags %v on instance: %v", tags, instanceName)
	return resTags.Items
}

// GetNodeTags gets k8s node tag from one of the nodes
func GetNodeTags(c clientset.Interface, cloudConfig framework.CloudConfig) []string {
	nodes := framework.GetReadySchedulableNodesOrDie(c)
	if len(nodes.Items) == 0 {
		framework.Logf("GetNodeTags: Found 0 node.")
		return []string{}
	}
	return GetInstanceTags(cloudConfig, nodes.Items[0].Name).Items
}

// IsHTTPErrorCode returns true if the error is a google api
// error matching the corresponding HTTP error code.
func IsGoogleAPIHTTPErrorCode(err error, code int) bool {
	apiErr, ok := err.(*googleapi.Error)
	return ok && apiErr.Code == code
}

func GetGCECloud() (*gcecloud.Cloud, error) {
	p, ok := framework.TestContext.CloudConfig.Provider.(*Provider)
	if !ok {
		return nil, fmt.Errorf("failed to convert CloudConfig.Provider to GCE provider: %#v", framework.TestContext.CloudConfig.Provider)
	}
	return p.gceCloud, nil
}

func GetClusterID(c clientset.Interface) (string, error) {
	cm, err := c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(gcecloud.UIDConfigMapName, metav1.GetOptions{})
	if err != nil || cm == nil {
		return "", fmt.Errorf("error getting cluster ID: %v", err)
	}
	clusterID, clusterIDExists := cm.Data[gcecloud.UIDCluster]
	providerID, providerIDExists := cm.Data[gcecloud.UIDProvider]
	if !clusterIDExists {
		return "", fmt.Errorf("cluster ID not set")
	}
	if providerIDExists {
		return providerID, nil
	}
	return clusterID, nil
}
