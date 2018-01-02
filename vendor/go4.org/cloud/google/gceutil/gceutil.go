/*
Copyright 2015 The Camlistore Authors

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

// Package gceutil provides utility functions to help with instances on
// Google Compute Engine.
package gceutil // import "go4.org/cloud/google/gceutil"

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"

	"google.golang.org/api/compute/v1"
)

// CoreOSImageURL returns the URL of the latest stable CoreOS image for running on Google Compute Engine.
func CoreOSImageURL(cl *http.Client) (string, error) {
	resp, err := cl.Get("https://www.googleapis.com/compute/v1/projects/coreos-cloud/global/images")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	type coreOSImage struct {
		SelfLink          string
		CreationTimestamp time.Time
		Name              string
	}

	type coreOSImageList struct {
		Items []coreOSImage
	}

	imageList := &coreOSImageList{}
	if err := json.NewDecoder(resp.Body).Decode(imageList); err != nil {
		return "", err
	}
	if imageList == nil || len(imageList.Items) == 0 {
		return "", errors.New("no images list in response")
	}

	imageURL := ""
	var max time.Time // latest stable image creation time
	for _, v := range imageList.Items {
		if !strings.HasPrefix(v.Name, "coreos-stable") {
			continue
		}
		if v.CreationTimestamp.After(max) {
			max = v.CreationTimestamp
			imageURL = v.SelfLink
		}
	}
	if imageURL == "" {
		return "", errors.New("no stable coreOS image found")
	}
	return imageURL, nil
}

// InstanceGroupAndManager contains both an InstanceGroup and
// its InstanceGroupManager, if any.
type InstanceGroupAndManager struct {
	Group *compute.InstanceGroup

	// Manager is the manager of the Group. It may be nil.
	Manager *compute.InstanceGroupManager
}

// InstanceGroups returns all the instance groups in a project's zone, along
// with their associated InstanceGroupManagers.
// The returned map is keyed by the instance group identifier URL.
func InstanceGroups(svc *compute.Service, proj, zone string) (map[string]InstanceGroupAndManager, error) {
	managerList, err := svc.InstanceGroupManagers.List(proj, zone).Do()
	if err != nil {
		return nil, err
	}
	if managerList.NextPageToken != "" {
		return nil, errors.New("too many managers; pagination not supported")
	}
	managedBy := make(map[string]*compute.InstanceGroupManager) // instance group URL -> its manager
	for _, it := range managerList.Items {
		managedBy[it.InstanceGroup] = it
	}
	groupList, err := svc.InstanceGroups.List(proj, zone).Do()
	if err != nil {
		return nil, err
	}
	if groupList.NextPageToken != "" {
		return nil, errors.New("too many instance groups; pagination not supported")
	}
	ret := make(map[string]InstanceGroupAndManager)
	for _, it := range groupList.Items {
		ret[it.SelfLink] = InstanceGroupAndManager{it, managedBy[it.SelfLink]}
	}
	return ret, nil
}
