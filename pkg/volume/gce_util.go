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

func NewGCEDiskUtil() *GCEDiskUtil {
	return &GCEDiskUtil{}
}

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

func (util *GCEDiskUtil) convertDiskToAttachedDisk(disk *compute.Disk, readWrite string) *compute.AttachedDisk {
	return &compute.AttachedDisk{
		false,
		false,
		disk.Name,
		0,
		nil,
		disk.Kind,
		//disk.Licenses,
		readWrite,
		"https://" + path.Join("www.googleapis.com/compute/v1/projects/", util.project, "zones", util.zone, "disks", disk.Name),
		"PERSISTENT",
	}
}
func (util *GCEDiskUtil) getDisk(GCEPD *PersistentDisk) (*compute.Disk, error) {
	return util.service.Disks.Get(util.project, util.zone, GCEPD.PDName).Do()
}

func (util *GCEDiskUtil) attachDisk(attachedDisk *compute.AttachedDisk) (*compute.Operation, error) {
	return util.service.Instances.AttachDisk(util.project, util.zone, util.instance, attachedDisk).Do()
}

func (util *GCEDiskUtil) AttachDisk(GCEPD *PersistentDisk) (string, error) {
	if !util.connected {
		return "", errors.New("Not connected to API")
	}
	disk, err := util.getDisk(GCEPD)
	if err != nil {
		return "", err
	}
	readWrite := "READ_WRITE"
	if GCEPD.ReadOnly {
		readWrite = "READ_ONLY"
	}
	attachedDisk := util.convertDiskToAttachedDisk(disk, readWrite)
	if _, err := util.attachDisk(attachedDisk); err != nil {
		return "", err
	}
	success := make(chan struct{})
	devicePath := path.Join("/dev/disk/by-id/", "google-"+disk.Name)
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
		return "", errors.New("Could not attach disk: Timeout after 10s")
	}
	return devicePath, nil
}

func (util *GCEDiskUtil) DetachDisk(GCEPD *PersistentDisk) error {
	if !util.connected {
		return errors.New("Not connected to API")
	}
	return nil
}
