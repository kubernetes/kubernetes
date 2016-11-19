/*
Copyright 2015 The Kubernetes Authors.

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
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	sio "github.com/codedellemc/goscaleio"
	siotypes "github.com/codedellemc/goscaleio/types/v1"
)

type sioInterface interface {
	FindVolume(name string) (*siotypes.Volume, error)
	Volume(id string) (*siotypes.Volume, error)
	CreateVolume(name string, sizeGB int64) (*siotypes.Volume, error)
	AttachVolume(id string) error
	DetachVolume(id string) error
	DeleteVolume(id string) error
	IID() (string, error)
	Devs() (map[string]string, error)
	WaitForAttachedDevice(token string) (string, error)
	WaitForDetachedDevice(token string) error
}

type sioClient struct {
	client           *sio.Client
	gateway          string
	username         string
	password         string
	insecure         bool
	certsEnabled     bool
	system           *siotypes.System
	sysName          string
	sysClient        *sio.System
	protectionDomain *siotypes.ProtectionDomain
	pdName           string
	pdClient         *sio.ProtectionDomain
	storagePool      *siotypes.StoragePool
	spName           string
	spClient         *sio.StoragePool
	provisionMode    string
	sdcPath          string
	instanceID       string
	inited           bool
	mtx              sync.Mutex
}

func newSioClient(gateway, username, password string, sslEnabled bool) (*sioClient, error) {
	client := new(sioClient)
	client.gateway = gateway
	client.username = username
	client.password = password
	if sslEnabled {
		client.insecure = false
		client.certsEnabled = true
	} else {
		client.insecure = true
		client.certsEnabled = false
	}
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
	client, err := sio.NewClientWithArgs(c.gateway, "", c.insecure, c.certsEnabled)
	if err != nil {
		return err
	}
	c.client = client
	if _, err = c.client.Authenticate(
		&sio.ConfigConnect{
			Endpoint: c.gateway,
			Version:  "",
			Username: c.username,
			Password: c.password},
	); err != nil {
		return err
	}

	// retrieve system
	if c.system, err = c.findSystem(c.sysName); err != nil {
		return err
	}

	// retrieve protection domain
	if c.protectionDomain, err = c.findProtectionDomain(c.pdName); err != nil {
		return err
	}
	// retrieve storage pool
	if c.storagePool, err = c.findStoragePool(c.spName); err != nil {
		return err
	}
	c.inited = true
	return nil
}

func (c *sioClient) Volumes() ([]*siotypes.Volume, error) {
	if err := c.init(); err != nil {
		return nil, err
	}
	return c.getVolumes()
}

func (c *sioClient) Volume(id string) (*siotypes.Volume, error) {
	if err := c.init(); err != nil {
		return nil, err
	}

	vols, err := c.getVolumesByID(id)
	if err != nil {
		return nil, err
	}
	vol := vols[0]
	if vol == nil {
		return nil, errors.New("volume not found")
	}
	return vol, nil
}

func (c *sioClient) FindVolume(name string) (*siotypes.Volume, error) {
	if err := c.init(); err != nil {
		return nil, err
	}

	volumes, err := c.getVolumesByName(name)
	if err != nil {
		return nil, err
	}

	for _, volume := range volumes {
		if volume.Name == name {
			return volume, nil
		}
	}
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
		return nil, err
	}
	return c.Volume(createResponse.ID)
}

// AttachVolume attaches the volume to a device and returns the volumeID of  attached vol as token.
func (c *sioClient) AttachVolume(id string) error {
	if err := c.init(); err != nil {
		return err
	}

	iid, err := c.IID()
	if err != nil {
		return err
	}

	params := &siotypes.MapVolumeSdcParam{
		SdcID: iid,
		AllowMultipleMappings: "false",
		AllSdcs:               "",
	}
	volClient := sio.NewVolume(c.client)
	volClient.Volume = &siotypes.Volume{ID: id}

	return volClient.MapVolumeSdc(params)
}

// DetachVolume detaches the volume with specified id.
func (c *sioClient) DetachVolume(id string) error {
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
	volClient.Volume = &siotypes.Volume{ID: id}
	if err := volClient.UnmapVolumeSdc(params); err != nil {
		return err
	}
	return nil
}

// DeleteVolume deletes the volume with the specified id
func (c *sioClient) DeleteVolume(id string) error {
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
		return err
	}
	return nil
}

func (c *sioClient) IID() (string, error) {
	if c.instanceID == "" {
		cmd := c.getSdcCmd()
		output, err := exec.Command(cmd, "--query_guid").Output()
		if err != nil {
			return "", err
		}

		c.instanceID = strings.TrimSpace(string(output))
	}
	return c.instanceID, nil
}

// mapDevs returns a map of local devices as map[<volume.id>]<deviceName>
func (c *sioClient) Devs() (map[string]string, error) {
	volumeMap := make(map[string]string)

	// grab the sdc tool output
	out, err := exec.Command(c.getSdcCmd(), "--query_vols").Output()
	if err != nil {
		return nil, err
	}

	// parse output and store in map with format map[<mdmID-volID>]volID
	result := string(out)
	lines := strings.Split(result, "\n")
	var mdmMap map[string]string
	for _, line := range lines {
		split := strings.Split(line, " ")
		if split[0] == "VOL-ID" {
			// split[3] = mdmID, split[1] = volID
			key := fmt.Sprintf("%s-%s", split[3], split[1])
			mdmMap[key] = split[1]
		}
	}

	// traverse disk device list and map devicePath to volumeID
	diskIDPath := "/dev/disk/by-id"
	files, _ := ioutil.ReadDir(diskIDPath)
	r, _ := regexp.Compile(`^emc-vol-\w*-\w*$`)

	for _, f := range files {
		matched := r.MatchString(f.Name())
		if matched {
			// remove emec-vol- prefix to be left with concated mdmID-volID
			mdmVolumeID := strings.Replace(f.Name(), "emc-vol-", "", 1)
			devPath, err := filepath.EvalSymlinks(fmt.Sprintf("%s/%s", diskIDPath, f.Name()))
			if err != nil {
				return nil, err
			}
			if volumeID, ok := mdmMap[mdmVolumeID]; ok {
				volumeMap[volumeID] = devPath
			}
		}
	}
	return volumeMap, nil
}

// WaitForAttachedDevice sets up a timer to wait for an attached device to appear in the instance's list.
func (c *sioClient) WaitForAttachedDevice(token string) (string, error) {
	if token == "" {
		return "", fmt.Errorf("invalid attach token")
	}

	// wait for attach.Token to show up in local device list
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	timer := time.NewTimer(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			devMap, err := c.Devs()
			if err != nil {
				return "", err
			}
			for _, v := range devMap {
				if v == token {
					return devMap[token], nil
				}
			}
		case <-timer.C:
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
	timer := time.NewTimer(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			devMap, err := c.Devs()
			if err != nil {
				return err
			}
			// cant find vol id, then ok.
			if _, ok := devMap[token]; !ok {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("volume detach timeout")
		}
	}
}

// ***********************************************************************
// Little Helpers!
// ***********************************************************************
func (c *sioClient) findSystem(sysname string) (sys *siotypes.System, err error) {
	if c.sysClient, err = c.client.FindSystem("", sysname, ""); err != nil {
		return nil, err
	}
	systems, err := c.client.GetInstance("")
	if err != nil {
		return nil, err
	}
	for _, sys = range systems {
		if sys.Name == sysname {
			return sys, nil
		}
	}
	return nil, errors.New("instance not found")
}

func (c *sioClient) findProtectionDomain(pdname string) (*siotypes.ProtectionDomain, error) {
	c.pdClient = sio.NewProtectionDomain(c.client)
	if c.sysClient != nil {
		protectionDomain, err := c.sysClient.FindProtectionDomain("", pdname, "")
		if err != nil {
			return nil, err
		}
		c.pdClient.ProtectionDomain = protectionDomain
		return protectionDomain, nil
	}
	return nil, errors.New("system not set")
}

func (c *sioClient) findStoragePool(spname string) (*siotypes.StoragePool, error) {
	c.spClient = sio.NewStoragePool(c.client)
	if c.pdClient != nil {
		sp, err := c.pdClient.FindStoragePool("", spname, "")
		if err != nil {
			return nil, err
		}
		c.spClient.StoragePool = sp
		return sp, nil
	}
	return nil, errors.New("protection domain not set")
}

func (c *sioClient) getVolumes() ([]*siotypes.Volume, error) {
	return c.client.GetVolume("", "", "", "", true)
}
func (c *sioClient) getVolumesByID(id string) ([]*siotypes.Volume, error) {
	return c.client.GetVolume("", id, "", "", true)
}

func (c *sioClient) getVolumesByName(name string) ([]*siotypes.Volume, error) {
	return c.client.GetVolume("", "", "", name, true)
}

func (c *sioClient) getSdcPath() string {
	if c.sdcPath == "" {
		c.sdcPath = "/opt/emc/scaleio/sdc/bin"
	}
	return c.sdcPath
}

func (c *sioClient) getSdcCmd() string {
	return path.Join(c.getSdcPath(), "drv_cfg")
}
