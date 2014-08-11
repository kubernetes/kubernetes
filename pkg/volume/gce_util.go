/*
Copyright 2014 Google Inc. All rights reserved.

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

package volume

import (
	"errors"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"code.google.com/p/goauth2/compute/serviceaccount"
	compute "code.google.com/p/google-api-go-client/compute/v1"
)

type GCEDiskUtil struct {
	project   string
	zone      string
	instance  string
	service   *compute.Service
	connected bool
}

// Establishes a connection to the GCE API and initializes project, zone,
// and instance members.
func (GCEPD *GCEDiskUtil) Connect() error {
	var err error
	GCEPD.project, GCEPD.zone, err = getProjectIDAndZone()
	if err != nil {
		return err
	}
	GCEPD.instance, err = os.Hostname()
	if err != nil {
		return err
	}
	client, err := serviceaccount.NewClient(&serviceaccount.Options{})
	if err != nil {
		return err
	}
	GCEPD.service, err = compute.New(client)
	if err != nil {
		return err
	}
	GCEPD.connected = true
	return nil
}

// Acquires zone and project names from the metadata server.
func getProjectIDAndZone() (string, string, error) {
	req, _ := http.NewRequest("GET", "http://metadata.google.internal/computeMetadata/v1/instance/zone", nil)
	req.Header.Add("Metadata-Flavor", "Google")
	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return "", "", err
	}
	result, err := ioutil.ReadAll(resp.Body)
	fullProjectAndZone := string(result)
	splits := strings.Split(fullProjectAndZone, "/")
	projectID := splits[1]
	zone := splits[3]
	if err != nil {
		return "", "", err
	}
	return projectID, zone, nil
}

// Converts a Disk resource to an AttachedDisk resource.
func (util *GCEDiskUtil) convertDiskToAttachedDisk(disk *compute.Disk, readWrite string) *compute.AttachedDisk {
	return &compute.AttachedDisk{
		false,
		false,
		disk.Name,
		0,
		nil,
		disk.Kind,
		//TODO(jonesdl) The version of go-api-client in kubernetes is out of date.
		//The recent versions have a licenses member.
		//disk.Licenses,
		readWrite,
		"https://" + path.Join("www.googleapis.com/compute/v1/projects/", util.project, "zones", util.zone, "disks", disk.Name),
		"PERSISTENT",
	}
}

// API call to retrieve a disk
func (util *GCEDiskUtil) getDisk(GCEPD *PersistentDisk) (*compute.Disk, error) {
	return util.service.Disks.Get(util.project, util.zone, GCEPD.PDName).Do()
}

// API call to attach a disk
func (util *GCEDiskUtil) attachDisk(attachedDisk *compute.AttachedDisk) (*compute.Operation, error) {
	return util.service.Instances.AttachDisk(util.project, util.zone, util.instance, attachedDisk).Do()
}

// API call to detach a disk
func (util *GCEDiskUtil) detachDisk(devicePath string) (*compute.Operation, error) {
	return util.service.Instances.DetachDisk(util.project, util.zone, util.instance, devicePath).Do()
}

// Attaches a disk specified by a volume.PersistentDisk to the current kubelet.
// Mounts the disk to it's global path.
func (util *GCEDiskUtil) AttachDisk(GCEPD *PersistentDisk) error {
	if !util.connected {
		return errors.New("Not connected to API")
	}
	disk, err := util.getDisk(GCEPD)
	if err != nil {
		return err
	}
	readWrite := "READ_WRITE"
	flags := uintptr(0)
	if GCEPD.ReadOnly {
		readWrite = "READ_ONLY"
		flags = MOUNT_MS_RDONLY
	}
	attachedDisk := util.convertDiskToAttachedDisk(disk, readWrite)
	if _, err := util.attachDisk(attachedDisk); err != nil {
		return err
	}
	success := make(chan struct{})
	devicePath := path.Join("/dev/disk/by-id/", "google-"+disk.Name)
	if GCEPD.Partition != "" {
		devicePath = devicePath + "-part" + GCEPD.Partition
	}
	go func() {
		for _, err := os.Stat(devicePath); os.IsNotExist(err); _, err = os.Stat(devicePath) {
			time.Sleep(time.Second)
		}
		close(success)
	}()
	select {
	case <-success:
		break
	case <-time.After(10 * time.Second):
		return errors.New("Could not attach disk: Timeout after 10s")
	}
	globalPDPath := path.Join(GCEPD.RootDir, "global", "pd", GCEPD.PDName)
	// Only mount the PD globally once.
	if _, err = os.Stat(globalPDPath); os.IsNotExist(err) {
		err = os.MkdirAll(globalPDPath, 0750)
		if err != nil {
			return err
		}
		err = GCEPD.mounter.Mount(devicePath, globalPDPath, GCEPD.FSType, flags, "")
		if err != nil {
			os.RemoveAll(globalPDPath)
			return err
		}
	}
	return nil
}

// Unmounts the device and detaches the disk from the kubelet's host machine.
func (util *GCEDiskUtil) DetachDisk(GCEPD *PersistentDisk, devicePath string) error {
	if !util.connected {
		return errors.New("Not connected to API")
	}
	actualDevicePath, err := filepath.EvalSymlinks(devicePath)
	if err != nil {
		return err
	}
	//TODO(jonesdl) extract the partition number directory from the regex.
	isMatch, err := regexp.MatchString("[a-z][a-z]*[0-9]", path.Base(actualDevicePath))
	if err != nil {
		return err
	}
	if isMatch {
		partition := string(actualDevicePath[len(actualDevicePath)-1])
		devicePath = strings.TrimSuffix(devicePath, "-part"+partition)
	}
	deviceName := strings.TrimPrefix(path.Base(devicePath), "google-")
	globalPDPath := path.Join(GCEPD.RootDir, "global", "pd", deviceName)
	if err := GCEPD.mounter.Unmount(globalPDPath, 0); err != nil {
		return err
	}
	if err := os.RemoveAll(globalPDPath); err != nil {
		return err
	}
	if _, err := util.detachDisk(deviceName); err != nil {
		return err
	}
	return nil
}
