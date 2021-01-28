/*
Copyright 2017 The Kubernetes Authors.

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

package scaleio

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	utilexec "k8s.io/utils/exec"

	sio "github.com/thecodeteam/goscaleio"
	siotypes "github.com/thecodeteam/goscaleio/types/v1"
	"k8s.io/klog/v2"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
)

var (
	sioDiskIDPath = "/dev/disk/by-id"
)

type sioVolumeID string

type sioInterface interface {
	FindVolume(name string) (*siotypes.Volume, error)
	Volume(sioVolumeID) (*siotypes.Volume, error)
	CreateVolume(name string, sizeGB int64) (*siotypes.Volume, error)
	AttachVolume(sioVolumeID, bool) error
	DetachVolume(sioVolumeID) error
	DeleteVolume(sioVolumeID) error
	IID() (string, error)
	Devs() (map[string]string, error)
	WaitForAttachedDevice(token string) (string, error)
	WaitForDetachedDevice(token string) error
	GetVolumeRefs(sioVolumeID) (int, error)
}

type sioClient struct {
	client              *sio.Client
	gateway             string
	username            string
	password            string `datapolicy:"password"`
	insecure            bool
	certsEnabled        bool
	system              *siotypes.System
	sysName             string
	sysClient           *sio.System
	protectionDomain    *siotypes.ProtectionDomain
	pdName              string
	pdClient            *sio.ProtectionDomain
	storagePool         *siotypes.StoragePool
	spName              string
	spClient            *sio.StoragePool
	provisionMode       string
	sdcPath             string
	sdcGUID             string
	instanceID          string
	inited              bool
	diskRegex           *regexp.Regexp
	mtx                 sync.Mutex
	exec                utilexec.Interface
	filteredDialOptions *proxyutil.FilteredDialOptions
}

func newSioClient(gateway, username, password string, sslEnabled bool, exec utilexec.Interface, filteredDialOptions *proxyutil.FilteredDialOptions) (*sioClient, error) {
	client := new(sioClient)
	client.gateway = gateway
	client.username = username
	client.password = password
	client.exec = exec
	client.filteredDialOptions = filteredDialOptions
	if sslEnabled {
		client.insecure = false
		client.certsEnabled = true
	} else {
		client.insecure = true
		client.certsEnabled = false
	}
	r, err := regexp.Compile(`^emc-vol-\w*-\w*$`)
	if err != nil {
		klog.Error(log("failed to compile regex: %v", err))
		return nil, err
	}
	client.diskRegex = r

	// delay client setup/login until init()
	return client, nil
}

// init setups client and authenticate
func (c *sioClient) init() error {
	c.mtx.Lock()
	defer c.mtx.Unlock()
	if c.inited {
		return nil
	}
	klog.V(4).Infoln(log("initializing scaleio client"))
	client, err := sio.NewClientWithArgs(c.gateway, "", c.insecure, c.certsEnabled)
	if err != nil {
		klog.Error(log("failed to create client: %v", err))
		return err
	}
	transport, ok := client.Http.Transport.(*http.Transport)
	if !ok {
		return errors.New("could not set http.Transport options for scaleio client")
	}
	//lint:ignore SA1019 DialTLS must be used to support legacy clients.
	if transport.DialTLS != nil {
		return errors.New("DialTLS will be used instead of DialContext")
	}
	transport.DialContext = proxyutil.NewFilteredDialContext(transport.DialContext, nil, c.filteredDialOptions)
	c.client = client
	if _, err = c.client.Authenticate(
		&sio.ConfigConnect{
			Endpoint: c.gateway,
			Version:  "",
			Username: c.username,
			Password: c.password},
	); err != nil {
		// don't log error details from client calls in events
		klog.V(4).Infof(log("client authentication failed: %v", err))
		return errors.New("client authentication failed")
	}

	// retrieve system
	if c.system, err = c.findSystem(c.sysName); err != nil {
		klog.Error(log("unable to find system %s: %v", c.sysName, err))
		return err
	}

	// retrieve protection domain
	if c.protectionDomain, err = c.findProtectionDomain(c.pdName); err != nil {
		klog.Error(log("unable to find protection domain %s: %v", c.protectionDomain, err))
		return err
	}
	// retrieve storage pool
	if c.storagePool, err = c.findStoragePool(c.spName); err != nil {
		klog.Error(log("unable to find storage pool %s: %v", c.storagePool, err))
		return err
	}
	c.inited = true
	return nil
}

func (c *sioClient) Volumes() ([]*siotypes.Volume, error) {
	if err := c.init(); err != nil {
		return nil, err
	}
	vols, err := c.getVolumes()
	if err != nil {
		klog.Error(log("failed to retrieve volumes: %v", err))
		return nil, err
	}
	return vols, nil
}

func (c *sioClient) Volume(id sioVolumeID) (*siotypes.Volume, error) {
	if err := c.init(); err != nil {
		return nil, err
	}

	vols, err := c.getVolumesByID(id)
	if err != nil {
		klog.Error(log("failed to retrieve volume by id: %v", err))
		return nil, err
	}
	vol := vols[0]
	if vol == nil {
		klog.V(4).Info(log("volume not found, id %s", id))
		return nil, errors.New("volume not found")
	}
	return vol, nil
}

func (c *sioClient) FindVolume(name string) (*siotypes.Volume, error) {
	if err := c.init(); err != nil {
		return nil, err
	}

	klog.V(4).Info(log("searching for volume %s", name))
	volumes, err := c.getVolumesByName(name)
	if err != nil {
		klog.Error(log("failed to find volume by name %v", err))
		return nil, err
	}

	for _, volume := range volumes {
		if volume.Name == name {
			klog.V(4).Info(log("found volume %s", name))
			return volume, nil
		}
	}
	klog.V(4).Info(log("volume not found, name %s", name))
	return nil, errors.New("volume not found")
}

func (c *sioClient) CreateVolume(name string, sizeGB int64) (*siotypes.Volume, error) {
	if err := c.init(); err != nil {
		return nil, err
	}

	params := &siotypes.VolumeParam{
		Name:           name,
		VolumeSizeInKb: strconv.Itoa(int(sizeGB) * 1024 * 1024),
		VolumeType:     c.provisionMode,
	}
	createResponse, err := c.client.CreateVolume(params, c.storagePool.Name)
	if err != nil {
		// don't log error details from client calls in events
		klog.V(4).Infof(log("failed to create volume %s: %v", name, err))
		return nil, errors.New("failed to create volume: see kubernetes logs for details")
	}
	return c.Volume(sioVolumeID(createResponse.ID))
}

// AttachVolume maps the scaleio volume to an sdc node.  If the multipleMappings flag
// is true, ScaleIO will allow other SDC to map to that volume.
func (c *sioClient) AttachVolume(id sioVolumeID, multipleMappings bool) error {
	if err := c.init(); err != nil {
		klog.Error(log("failed to init'd client in attach volume: %v", err))
		return err
	}

	iid, err := c.IID()
	if err != nil {
		klog.Error(log("failed to get instanceIID for attach volume: %v", err))
		return err
	}

	params := &siotypes.MapVolumeSdcParam{
		SdcID:                 iid,
		AllowMultipleMappings: strconv.FormatBool(multipleMappings),
		AllSdcs:               "",
	}
	volClient := sio.NewVolume(c.client)
	volClient.Volume = &siotypes.Volume{ID: string(id)}

	if err := volClient.MapVolumeSdc(params); err != nil {
		// don't log error details from client calls in events
		klog.V(4).Infof(log("failed to attach volume id %s: %v", id, err))
		return errors.New("failed to attach volume: see kubernetes logs for details")
	}

	klog.V(4).Info(log("volume %s attached successfully", id))
	return nil
}

// DetachVolume detaches the volume with specified id.
func (c *sioClient) DetachVolume(id sioVolumeID) error {
	if err := c.init(); err != nil {
		return err
	}

	iid, err := c.IID()
	if err != nil {
		return err
	}
	params := &siotypes.UnmapVolumeSdcParam{
		SdcID:                "",
		IgnoreScsiInitiators: "true",
		AllSdcs:              iid,
	}
	volClient := sio.NewVolume(c.client)
	volClient.Volume = &siotypes.Volume{ID: string(id)}
	if err := volClient.UnmapVolumeSdc(params); err != nil {
		// don't log error details from client calls in events
		klog.V(4).Infof(log("failed to detach volume id %s: %v", id, err))
		return errors.New("failed to detach volume: see kubernetes logs for details")
	}
	return nil
}

// DeleteVolume deletes the volume with the specified id
func (c *sioClient) DeleteVolume(id sioVolumeID) error {
	if err := c.init(); err != nil {
		return err
	}

	vol, err := c.Volume(id)
	if err != nil {
		return err
	}
	volClient := sio.NewVolume(c.client)
	volClient.Volume = vol
	if err := volClient.RemoveVolume("ONLY_ME"); err != nil {
		// don't log error details from client calls in events
		klog.V(4).Infof(log("failed to remove volume id %s: %v", id, err))
		return errors.New("failed to remove volume: see kubernetes logs for details")
	}
	return nil
}

// IID returns the scaleio instance id for node
func (c *sioClient) IID() (string, error) {
	if err := c.init(); err != nil {
		return "", err
	}

	// if instanceID not set, retrieve it
	if c.instanceID == "" {
		guid, err := c.getGUID()
		if err != nil {
			return "", err
		}
		sdc, err := c.sysClient.FindSdc("SdcGUID", guid)
		if err != nil {
			// don't log error details from client calls in events
			klog.V(4).Infof(log("failed to retrieve sdc info %s", err))
			return "", errors.New("failed to retrieve sdc info: see kubernetes logs for details")
		}
		c.instanceID = sdc.Sdc.ID
		klog.V(4).Info(log("retrieved instanceID %s", c.instanceID))
	}
	return c.instanceID, nil
}

// getGUID returns instance GUID, if not set using resource labels
// it attempts to fallback to using drv_cfg binary
func (c *sioClient) getGUID() (string, error) {
	if c.sdcGUID == "" {
		klog.V(4).Info(log("sdc guid label not set, falling back to using drv_cfg"))
		cmd := c.getSdcCmd()
		output, err := c.exec.Command(cmd, "--query_guid").CombinedOutput()
		if err != nil {
			klog.Error(log("drv_cfg --query_guid failed: %v", err))
			return "", err
		}
		c.sdcGUID = strings.TrimSpace(string(output))
	}
	return c.sdcGUID, nil
}

// getSioDiskPaths traverse local disk devices to retrieve device path
// The path is extracted from /dev/disk/by-id; each sio device path has format:
// emc-vol-<mdmID-volID> e.g.:
// emc-vol-788d9efb0a8f20cb-a2b8419300000000
func (c *sioClient) getSioDiskPaths() ([]os.FileInfo, error) {
	files, err := ioutil.ReadDir(sioDiskIDPath)
	if err != nil {
		if os.IsNotExist(err) {
			// sioDiskIDPath may not exist yet which is fine
			return []os.FileInfo{}, nil
		}
		klog.Error(log("failed to ReadDir %s: %v", sioDiskIDPath, err))
		return nil, err

	}
	result := []os.FileInfo{}
	for _, file := range files {
		if c.diskRegex.MatchString(file.Name()) {
			result = append(result, file)
		}
	}

	return result, nil

}

// GetVolumeRefs counts the number of references an SIO volume has a disk device.
// This is useful in preventing premature detach.
func (c *sioClient) GetVolumeRefs(volID sioVolumeID) (refs int, err error) {
	files, err := c.getSioDiskPaths()
	if err != nil {
		return 0, err
	}
	for _, file := range files {
		if strings.Contains(file.Name(), string(volID)) {
			refs++
		}
	}
	return
}

// Devs returns a map of local devices as map[<volume.id>]<deviceName>
func (c *sioClient) Devs() (map[string]string, error) {
	volumeMap := make(map[string]string)

	files, err := c.getSioDiskPaths()
	if err != nil {
		return nil, err
	}

	for _, f := range files {
		// split emc-vol-<mdmID>-<volumeID> to pull out volumeID
		parts := strings.Split(f.Name(), "-")
		if len(parts) != 4 {
			return nil, errors.New("unexpected ScaleIO device name format")
		}
		volumeID := parts[3]
		devPath, err := filepath.EvalSymlinks(fmt.Sprintf("%s/%s", sioDiskIDPath, f.Name()))
		if err != nil {
			klog.Error(log("devicepath-to-volID mapping error: %v", err))
			return nil, err
		}
		// map volumeID to devicePath
		volumeMap[volumeID] = devPath
	}
	return volumeMap, nil
}

// WaitForAttachedDevice sets up a timer to wait for an attached device to appear in the instance's list.
func (c *sioClient) WaitForAttachedDevice(token string) (string, error) {
	if token == "" {
		return "", fmt.Errorf("invalid attach token")
	}

	// wait for device to  show up in local device list
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	timer := time.NewTimer(30 * time.Second)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			devMap, err := c.Devs()
			if err != nil {
				klog.Error(log("failed while waiting for volume to attach: %v", err))
				return "", err
			}
			go func() {
				klog.V(4).Info(log("waiting for volume %s to be mapped/attached", token))
			}()
			if path, ok := devMap[token]; ok {
				klog.V(4).Info(log("device %s mapped to vol %s", path, token))
				return path, nil
			}
		case <-timer.C:
			klog.Error(log("timed out while waiting for volume to be mapped to a device"))
			return "", fmt.Errorf("volume attach timeout")
		}
	}
}

// waitForDetachedDevice waits for device to be detached
func (c *sioClient) WaitForDetachedDevice(token string) error {
	if token == "" {
		return fmt.Errorf("invalid detach token")
	}

	// wait for attach.Token to show up in local device list
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	timer := time.NewTimer(30 * time.Second)
	defer timer.Stop()

	for {
		select {
		case <-ticker.C:
			devMap, err := c.Devs()
			if err != nil {
				klog.Error(log("failed while waiting for volume to unmap/detach: %v", err))
				return err
			}
			go func() {
				klog.V(4).Info(log("waiting for volume %s to be unmapped/detached", token))
			}()
			// cant find vol id, then ok.
			if _, ok := devMap[token]; !ok {
				return nil
			}
		case <-timer.C:
			klog.Error(log("timed out while waiting for volume %s to be unmapped/detached", token))
			return fmt.Errorf("volume detach timeout")
		}
	}
}

// ***********************************************************************
// Little Helpers!
// ***********************************************************************
func (c *sioClient) findSystem(sysname string) (sys *siotypes.System, err error) {
	if c.sysClient, err = c.client.FindSystem("", sysname, ""); err != nil {
		// don't log error details from clients in events
		klog.V(4).Infof(log("failed to find system %q: %v", sysname, err))
		return nil, errors.New("failed to find system: see kubernetes logs for details")
	}
	systems, err := c.client.GetInstance("")
	if err != nil {
		// don't log error details from clients in events
		klog.V(4).Infof(log("failed to retrieve instances: %v", err))
		return nil, errors.New("failed to retrieve instances: see kubernetes logs for details")
	}
	for _, sys = range systems {
		if sys.Name == sysname {
			return sys, nil
		}
	}
	klog.Error(log("system %s not found", sysname))
	return nil, errors.New("system not found")
}

func (c *sioClient) findProtectionDomain(pdname string) (*siotypes.ProtectionDomain, error) {
	c.pdClient = sio.NewProtectionDomain(c.client)
	if c.sysClient != nil {
		protectionDomain, err := c.sysClient.FindProtectionDomain("", pdname, "")
		if err != nil {
			// don't log error details from clients in events
			klog.V(4).Infof(log("failed to retrieve protection domains: %v", err))
			return nil, errors.New("failed to retrieve protection domains: see kubernetes logs for details")
		}
		c.pdClient.ProtectionDomain = protectionDomain
		return protectionDomain, nil
	}
	klog.Error(log("protection domain %s not set", pdname))
	return nil, errors.New("protection domain not set")
}

func (c *sioClient) findStoragePool(spname string) (*siotypes.StoragePool, error) {
	c.spClient = sio.NewStoragePool(c.client)
	if c.pdClient != nil {
		sp, err := c.pdClient.FindStoragePool("", spname, "")
		if err != nil {
			// don't log error details from clients in events
			klog.V(4).Infof(log("failed to retrieve storage pool: %v", err))
			return nil, errors.New("failed to retrieve storage pool: see kubernetes logs for details")
		}
		c.spClient.StoragePool = sp
		return sp, nil
	}
	klog.Error(log("storage pool %s not set", spname))
	return nil, errors.New("storage pool not set")
}

func (c *sioClient) getVolumes() ([]*siotypes.Volume, error) {
	volumes, err := c.client.GetVolume("", "", "", "", true)
	if err != nil {
		// don't log error details from clients in events
		klog.V(4).Infof(log("failed to get volumes: %v", err))
		return nil, errors.New("failed to get volumes: see kubernetes logs for details")
	}
	return volumes, nil
}
func (c *sioClient) getVolumesByID(id sioVolumeID) ([]*siotypes.Volume, error) {
	volumes, err := c.client.GetVolume("", string(id), "", "", true)
	if err != nil {
		// don't log error details from clients in events
		klog.V(4).Infof(log("failed to get volumes by id: %v", err))
		return nil, errors.New("failed to get volumes by id: see kubernetes logs for details")
	}
	return volumes, nil
}

func (c *sioClient) getVolumesByName(name string) ([]*siotypes.Volume, error) {
	volumes, err := c.client.GetVolume("", "", "", name, true)
	if err != nil {
		// don't log error details from clients in events
		klog.V(4).Infof(log("failed to get volumes by name: %v", err))
		return nil, errors.New("failed to get volumes by name: see kubernetes logs for details")
	}
	return volumes, nil
}

func (c *sioClient) getSdcPath() string {
	return sdcRootPath
}

func (c *sioClient) getSdcCmd() string {
	return filepath.Join(c.getSdcPath(), "drv_cfg")
}
