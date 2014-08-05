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

func GCEAttachDisk(GCEPD *GCEPersistentDisk) (string, error) {
	client, err := serviceaccount.NewClient(&serviceaccount.Options{})
	if err != nil {
		return "", err
	}
	project, zone, err := getProjectIDAndZone()
	if err != nil {
		return "", err
	}
	instance, err := os.Hostname()
	if err != nil {
		return "", err
	}
	computeService, err := compute.New(client)
	if err != nil {
		return "", err
	}
	disk, err := computeService.Disks.Get(project, zone, GCEPD.PDName).Do()
	if err != nil {
		return "", err
	}
	readWrite := "READ_WRITE"
	if GCEPD.ReadOnly {
		readWrite = "READ_ONLY"
	}
	attachedDisk := convertDiskToAttachedDisk(disk, project, zone, readWrite)
	_, err = computeService.Instances.AttachDisk(project, zone, instance, attachedDisk).Do()
	if err != nil {
		return "", err
	}
	success := make(chan struct{})
	devicePath := path.Join("/dev/disk/by-id/", "google-"+disk.Name)
	go func() {
		_, err := os.Stat(devicePath)
		for os.IsNotExist(err) {
			_, err = os.Stat(devicePath)
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

func convertDiskToAttachedDisk(disk *compute.Disk, project string, zone string, readWrite string) *compute.AttachedDisk {
	return &compute.AttachedDisk{
		false,
		false,
		disk.Name,
		0,
		nil,
		disk.Kind,
		//		disk.Licenses,
		readWrite,
		"https://" + path.Join("www.googleapis.com/compute/v1/projects/", project, "zones", zone, "disks", disk.Name),
		"PERSISTENT",
	}
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
