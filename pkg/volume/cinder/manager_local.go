/*
Copyright 2016 The Kubernetes Authors.

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

package cinder

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"go/build"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/volumeactions"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/cinder/drivers"
	"k8s.io/kubernetes/pkg/volume/util"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type cdManagerLocal struct {
	plugin *cinderPlugin
}

func (cdl *cdManagerLocal) GetName() string {
	return "local"
}

// Attaches the volume specified by the given spec to the node with the given Name.
// On success, returns the device path where the device was attached on the
// node.
func (cdl *cdManagerLocal) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	volumeSource, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	globalPDPath := makeGlobalPDName(cdl.plugin.host, volumeSource.VolumeID)
	glog.V(4).Infof("Cinder (%v): Attach %v", cdl.GetName(), volumeSource.VolumeID)

	client, err := newCinderClientFromSecret(volumeSource.SecretRef, cdl.plugin.host)
	if err != nil {
		glog.Errorf("Cinder (%v): Attach %v: Failed to create Cinder client: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return "", err
	}

	// Query Cinder for the volume's information.
	vol, err := volumes.Get(client, volumeSource.VolumeID).Extract()
	if err != nil {
		glog.Errorf("Cinder (%v): Attach %v: Unable to retrieve volume information: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return "", err
	}

	// Verify if the volume is already attached to this node.
	isAttached, _ := isVolumeAttachedToNode(vol, string(nodeName))
	if isAttached {
		glog.V(2).Infof("Cinder (%v): Attach %v: Volume is already attached locally, skipping", cdl.GetName(), volumeSource.VolumeID)
		return globalPDPath, nil
	}

	// Verify if the volume can be attached at all.
	isVolumeAttachableErr := isVolumeAttachable(vol)
	if isVolumeAttachableErr != nil {
		glog.V(2).Infof("Cinder (%v): Attach %v: %v", cdl.GetName(), volumeSource.VolumeID, isVolumeAttachableErr)
		return "", err
	}

	// Reserve the volume, marking it as "Attaching" in Cinder.
	if err = volumeactions.Reserve(client, volumeSource.VolumeID).Err; err != nil {
		glog.Errorf("Cinder (%v): Attach %v: Reservation failed: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return "", fmt.Errorf("Cinder reservation failed: %v", err)
	}

	// Actually attach the volume.
	attachOpts := volumeactions.AttachOpts{
		MountPoint: globalPDPath,
		HostName:   string(nodeName),
		Mode:       volumeactions.ReadWrite,
	}
	if readOnly {
		attachOpts.Mode = volumeactions.ReadOnly
	}

	if err = volumeactions.Attach(client, volumeSource.VolumeID, &attachOpts).Err; err != nil {
		volumeactions.Unreserve(client, volumeSource.VolumeID)
		glog.Errorf("Cinder (%v): Attach %v: failed to attach volume: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return "", err
	}

	glog.V(4).Infof("Cinder (%v): Attach %v: success", cdl.GetName(), volumeSource.VolumeID)

	return globalPDPath, nil
}

// VolumesAreAttached checks whether the list of volumes still attached to the specified
// the node. It returns a map which maps from the volume spec to the checking result.
// If an error is occurred during checking, the error will be returned.
func (cdl *cdManagerLocal) VolumesAreAttached(specs []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
	results := make(map[*volume.Spec]bool)
	var attachedIDs []string
	var detachedIDs []string

	for _, spec := range specs {
		volumeSource, _, err := getVolumeSource(spec)
		if err != nil {
			return nil, err
		}

		// Create Cinder client.
		client, err := newCinderClientFromSecret(volumeSource.SecretRef, cdl.plugin.host)
		if err != nil {
			return nil, err
		}

		// Query Cinder for the volume's information and attachment status.
		vol, err := volumes.Get(client, volumeSource.VolumeID).Extract()
		attached := false
		if err != nil {
			glog.Warningf("Cinder (%v): Attach %v: Unable to retrieve volume information: %v", cdl.GetName(), volumeSource.VolumeID, err)
		} else {
			attached, _ = isVolumeAttachedToNode(vol, string(nodeName))
		}

		results[spec] = attached
		if attached {
			attachedIDs = append(attachedIDs, volumeSource.VolumeID)
		} else {
			detachedIDs = append(detachedIDs, volumeSource.VolumeID)
		}
	}

	glog.V(2).Infof("Cinder (%v): Currently attached volumes: %v", cdl.GetName(), attachedIDs)
	glog.V(2).Infof("Cinder (%v): No longer attached volumes: %v", cdl.GetName(), detachedIDs)

	return results, nil
}

// WaitForAttach blocks until the device is attached to this
// node. If it successfully attaches, the path to the device
// is returned. Otherwise, if the device does not attach after
// the given timeout period, an error will be returned.
func (cdl *cdManagerLocal) WaitForAttach(spec *volume.Spec, _ string, _ time.Duration) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}
	glog.V(4).Infof("Cinder (%v): WaitForAttach %v", cdl.GetName(), volumeSource.VolumeID)

	client, err := newCinderClientFromSecret(volumeSource.SecretRef, cdl.plugin.host)
	if err != nil {
		glog.Errorf("Cinder (%v): WaitForAttach %v: Failed to create Cinder client: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return "", err
	}

	// Initialize the connection with Cinder.
	connInfo, err := initializeConnection(client, volumeSource.VolumeID)
	if err != nil {
		glog.Errorf("Cinder (%v): WaitForAttach %v: initializeConnection failed: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return "", err
	}

	// Get the adequate driver.
	driver := drivers.GetDriver(connInfo.DriverVolumeType)
	if driver == nil {
		glog.Errorf("Cinder (%v): WaitForAttach %v: Unsupported volume type: %v", cdl.GetName(), volumeSource.VolumeID, connInfo.DriverVolumeType)
		return "", fmt.Errorf("Unsupported Cinder volume type: %v", connInfo.DriverVolumeType)
	}

	// Attach the volume using the driver.
	devicePath, err := driver.AttachDisk(connInfo)
	if err != nil {
		glog.Errorf("Cinder (%v): WaitForAttach %v: Driver failed to attach disk of type %v: %v", cdl.GetName(), volumeSource.VolumeID, connInfo.DriverVolumeType, err)
		return "", err
	}

	glog.V(4).Infof("Cinder (%v): WaitForAttach %v: success", cdl.GetName(), volumeSource.VolumeID)

	return devicePath, nil
}

// Detach the given device from the node with the given Name.
func (cdl *cdManagerLocal) Detach(spec *volume.Spec, deviceName string, nodeName types.NodeName) error {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return err
	}
	glog.V(4).Infof("Cinder (%v): Detach %v", cdl.GetName(), volumeSource.VolumeID)

	hostname, err := os.Hostname()
	if err != nil {
		glog.Errorf("Cinder (%v): Attach %v: Unable to retrieve hostname: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return err
	}

	client, err := newCinderClientFromSecret(volumeSource.SecretRef, cdl.plugin.host)
	if err != nil {
		glog.Errorf("Cinder (%v): Detach %v: Failed to create Cinder client: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return err
	}

	// Query Cinder for the volume's information.
	vol, err := volumes.Get(client, volumeSource.VolumeID).Extract()
	if err != nil {
		glog.Errorf("Cinder (%v): Detach %v: Unable to retrieve volume information: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return err
	}

	// Verify that the volume is attached and get its attachment ID.
	isAttached, attachmentID := isVolumeAttachedToNode(vol, string(nodeName))
	if !isAttached {
		glog.V(2).Infof("Cinder (%v): Detach %v: Volume is not attached, skipping")
		return nil
	}

	// Begin the Detaching process on Cinder.
	if err = volumeactions.BeginDetaching(client, volumeSource.VolumeID).Err; err != nil {
		glog.Errorf("Cinder (%v): Detach %v: BeginDetaching failed: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return err
	}

	// Execute Cinder's TerminateConnection.
	connOpts := volumeactions.TerminateConnectionOpts{
		Host:     hostname,
		Platform: build.Default.GOARCH,
		OSType:   build.Default.GOOS,
	}
	if err := volumeactions.TerminateConnection(client, volumeSource.VolumeID, &connOpts).Err; err != nil {
		glog.Errorf("Cinder (%v): Detach %v: TerminateConnection failed: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return err
	}

	// Execute Cinder's Detach.
	detachOpts := volumeactions.DetachOpts{AttachmentID: attachmentID}
	if err := volumeactions.Detach(client, volumeSource.VolumeID, &detachOpts).Err; err != nil {
		glog.Errorf("Cinder (%v): Detach %v: Failed to detach: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return err
	}

	glog.V(4).Infof("Cinder (%v): Detach %v: success", cdl.GetName(), volumeSource.VolumeID)

	return nil
}

// WaitForDetach blocks until the device is detached from this
// node. If the device does not detach within the given timeout
// period an error is returned.
func (cdl *cdManagerLocal) WaitForDetach(_ *volume.Spec, _ string, _ time.Duration) error {
	// Unused by the operation executor.
	return nil
}

// UnmountDevice unmounts the global mount of the disk. This
// should only be called once all bind mounts have been
// unmounted.
func (cdl *cdManagerLocal) UnmountDevice(spec *volume.Spec, _ string, mounter mount.Interface) error {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return err
	}
	glog.V(4).Infof("Cinder (%v): UnmountDevice %v", cdl.GetName(), volumeSource.VolumeID)

	client, err := newCinderClientFromSecret(volumeSource.SecretRef, cdl.plugin.host)
	if err != nil {
		glog.Errorf("Cinder (%v): UnmountDevice %v: Failed to create Cinder client: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return err
	}

	// Initialize the connection with Cinder.
	connInfo, err := initializeConnection(client, volumeSource.VolumeID)
	if err != nil {
		glog.Errorf("Cinder (%v): UnmountDevice %v: initializeConnection failed: %v", cdl.GetName(), volumeSource.VolumeID, err)
		return err
	}

	// Unmount the volume from its global path.
	globalPDPath := makeGlobalPDName(cdl.plugin.host, volumeSource.VolumeID)
	if err = volumeutil.UnmountPath(globalPDPath, cdl.plugin.host.GetMounter()); err != nil {
		glog.Errorf("Cinder (%v): UnmountDevice %v: Could not unmount global mountpoint %v: %v", cdl.GetName(), volumeSource.VolumeID, globalPDPath, err)
		return err
	}

	// Get the adequate driver.
	driver := drivers.GetDriver(connInfo.DriverVolumeType)
	if driver == nil {
		glog.Errorf("Cinder (%v): UnmountDevice %v: Unsupported volume type: %v", cdl.GetName(), volumeSource.VolumeID, connInfo.DriverVolumeType)
		return fmt.Errorf("Unsupported Cinder volume type: %v", connInfo.DriverVolumeType)
	}

	// Detach disk using the driver.
	if err = driver.DetachDisk(connInfo); err != nil {
		glog.Warningf("Cinder (%v): UnmountDevice %v: Driver failed to detach disk of type %v: %v", volumeSource.VolumeID, connInfo.DriverVolumeType, err)
		return fmt.Errorf("Could not detach disk: %v", err)
	}

	// Remove the global mountpoint.
	if err := os.RemoveAll(globalPDPath); err != nil {
		glog.Warningf("Cinder (%v): UnmountDevice %v: Could not remove global mountpoint %v: %v", cdl.GetName(), volumeSource.VolumeID, globalPDPath, err)
		return fmt.Errorf("Could not remove %s: %v", globalPDPath, err)
	}

	glog.V(4).Infof("Cinder (%v): UnmountDevice %v: success", cdl.GetName(), volumeSource.VolumeID)

	return nil
}

func (cdl *cdManagerLocal) CreateVolume(p *cinderVolumeProvisioner) (volumeID string, volumeSizeGB int, secretRef *v1.LocalObjectReference, err error) {
	// Initialize options structure for Cinder volume creation.
	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	createOpts := volumes.CreateOpts{
		Size: int(volume.RoundUpSize(capacity.Value(), 1024*1024*1024)),
		Name: volume.GenerateVolumeName(p.options.ClusterName, p.options.PVName, 255),
	}
	if p.options.CloudTags != nil {
		createOpts.Metadata = *p.options.CloudTags
	}

	// Parse ProvisionerParameters (case-insensitive).
	for k, v := range p.options.Parameters {
		switch strings.ToLower(k) {
		case "type":
			createOpts.VolumeType = v
		case "availability":
			createOpts.AvailabilityZone = v
		case "secretref":
			secretRef = new(v1.LocalObjectReference)
			secretRef.Name = v
		default:
			return "", 0, nil, fmt.Errorf("invalid option %q for volume plugin %s", k, p.plugin.GetPluginName())
		}
	}

	// TODO: implement PVC.Selector parsing
	if p.options.PVC.Spec.Selector != nil {
		return "", 0, nil, errors.New("claim.Spec.Selector is not supported by Cinder for dynamic provisioning")
	}

	glog.V(4).Infof("Cinder (%v): CreateVolume with options %#v", cdl.GetName(), createOpts)

	// Create Cinder client using the credentials specified in the given secret.
	client, err := newCinderClientFromSecret(secretRef, cdl.plugin.host)
	if err != nil {
		glog.Errorf("Cinder (%v): CreateVolume: Failed to create Cinder client: %v", cdl.GetName(), err)
		return "", 0, nil, err
	}

	// Create volume.
	res := volumes.Create(client, &createOpts)
	if res.Err != nil {
		glog.Errorf("Cinder (%v): CreateVolume: Failed to create volume with options %#v: %v", cdl.GetName(), createOpts, err)
		return "", 0, nil, fmt.Errorf("Cinder volume creation failed: %v", res.Err)
	}
	vol, err := res.Extract()
	if err != nil {
		glog.Errorf("Cinder (%v): CreateVolume: Unexpected results when creating volume with options %#v: %v", cdl.GetName(), createOpts, err)
		return "", 0, nil, fmt.Errorf("Cinder volume creation returned unexpected results: %v", err)
	}

	glog.V(4).Infof("Cinder (%v): CreateVolume with options %#v: success", cdl.GetName(), createOpts)

	return vol.ID, createOpts.Size, secretRef, nil
}

func (cdl *cdManagerLocal) DeleteVolume(d *cinderVolumeDeleter) error {
	client, err := newCinderClientFromSecret(d.secretRef, cdl.plugin.host)
	if err != nil {
		glog.Errorf("Cinder (%v): DeleteVolume %v: Failed to create Cinder client: %v", cdl.GetName(), d.pdName, err)
		return err
	}

	// Delete the volume.
	res := volumes.Delete(client, d.pdName)
	if res.Err != nil {
		glog.Errorf("Cinder (%v): DeleteVolume %v: Failed to delete volume: %v", cdl.GetName(), d.pdName, err)
		return fmt.Errorf("Cinder volume deletion failed: %v", res.Err)
	}

	return nil
}

func newCinderClientFromSecret(secretRef *v1.LocalObjectReference, host volume.VolumeHost) (*gophercloud.ServiceClient, error) {
	// Extract auth options from the specified secret.
	secretNamespace, secretName := splitSecretName(secretRef.Name)
	secretMap, err := util.GetSecretForPV(secretNamespace, secretName, cinderVolumePluginName, host.GetKubeClient())
	if err != nil {
		return nil, err
	}

	authOpts, httpClient, err := newAuthOptionsFromSecret(secretMap)
	if err != nil {
		return nil, err
	}

	// Create an OpenStack client and authenticate to Keystone using the extracted
	// credentials and an optional http.Client that may carry TLS configuration
	// (CA certificate, client certificate). Note that if provided, the specified
	// TLS configuration will be used for both Keystone and Cinder.
	provider, err := openstack.NewClient(authOpts.IdentityEndpoint)
	if err != nil {
		return nil, err
	}
	if httpClient != nil {
		provider.HTTPClient = *httpClient
	}
	err = openstack.Authenticate(provider, authOpts)
	if err != nil {
		return nil, err
	}

	// Create the Cinder v2 client from the authenticated OpenStack client.
	client, err := openstack.NewBlockStorageV2(provider, gophercloud.EndpointOpts{})
	if err != nil {
		return nil, err
	}

	return client, nil
}

func splitSecretName(secretName string) (string, string) {
	parts := strings.SplitN(secretName, "/", 2)
	if len(parts) == 1 {
		return "default", parts[0]
	}

	return parts[0], parts[1]
}

func newAuthOptionsFromSecret(secretMap map[string]string) (gophercloud.AuthOptions, *http.Client, error) {
	var httpClient *http.Client
	emptyOptions := gophercloud.AuthOptions{}

	authURL, ok := secretMap["authURL"]
	if !ok {
		return emptyOptions, httpClient, errors.New("secret does not have the required 'authURL' key")
	}

	username, ok := secretMap["username"]
	if !ok {
		return emptyOptions, httpClient,  errors.New("secret does not have the required 'username' key")
	}

	password, ok := secretMap["password"]
	if !ok {
		return emptyOptions, httpClient, errors.New("secret does not have the required 'password' key")
	}

	project := secretMap["project"]
	domain := secretMap["domain"]

	caCert := secretMap["caCert"]
	cert := secretMap["cert"]
	key := secretMap["key"]

	if caCert != "" || (cert != "" && key != "") {
		certificates := []tls.Certificate{}
		caCertificates := x509.NewCertPool()

		// Add CA certificate.
		if caCert != "" {
			ok := caCertificates.AppendCertsFromPEM([]byte(caCert))
			if !ok {
				return emptyOptions, httpClient, errors.New("secret has an invalid 'caCert' key")
			}
		}

		// Add client certificate.
		if cert != "" && key != "" {
			certificate, err := tls.X509KeyPair([]byte(cert), []byte(key))
			if err != nil {
				return emptyOptions, httpClient, errors.New("secret has an invalid 'cert' or 'key' key")
			}
			certificates = append(certificates, certificate)
		}

		httpClient = &http.Client{
			Transport: utilnet.SetTransportDefaults(&http.Transport{
				TLSClientConfig: &tls.Config{
					Certificates: certificates,
					RootCAs:      caCertificates,
				},
			}),
		}
	}

	return gophercloud.AuthOptions{
		IdentityEndpoint: string(authURL),
		Username:         string(username),
		Password:         string(password),
		TenantName:       string(project),
		DomainName:       string(domain),
	}, httpClient, nil
}

func initializeConnection(client *gophercloud.ServiceClient, volumeID string) (drivers.ConnectionInfo, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return drivers.ConnectionInfo{}, fmt.Errorf("Unable to retrieve hostname: %v", err)
	}
	connOpts := volumeactions.InitializeConnectionOpts{
		Host:     hostname,
		Platform: build.Default.GOARCH,
		OSType:   build.Default.GOOS,
	}

	icResult := volumeactions.InitializeConnection(client, volumeID, &connOpts)
	if icResult.Err != nil {
		return drivers.ConnectionInfo{}, fmt.Errorf("InitializeConnection failed: %v", icResult.Err)
	}

	var s struct {
		ConnInfo drivers.ConnectionInfo `json:"connection_info"`
	}
	if err := icResult.ExtractInto(&s); err != nil {
		return drivers.ConnectionInfo{}, fmt.Errorf("InitializeConnection returned unexpected results: %v", err)
	}

	return s.ConnInfo, nil
}

func isVolumeAttachable(volume *volumes.Volume) error {
	if !volume.Multiattach && len(volume.Attachments) > 0 {
		return errors.New("Volume is already attached to another node(s), should be detached before proceeding")
	}
	if volume.Status != "available" || (!volume.Multiattach && volume.Status == "in-use") {
		return fmt.Errorf("Volume's status is %q, want 'available', or 'in-use' if multi-attachment is supported, to proceed", volume.Status)
	}
	return nil
}

func isVolumeAttachedToNode(volume *volumes.Volume, nodeName string) (bool, string) {
	for _, attachment := range volume.Attachments {
		if attachment.HostName == string(nodeName) {
			return true, attachment.AttachmentID
		}
	}
	return false, ""
}
