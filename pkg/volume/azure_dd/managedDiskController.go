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

package azure_dd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"path"
	"strconv"
	"strings"

	"github.com/golang/glog"
	kwait "k8s.io/apimachinery/pkg/util/wait"
)

type managedDiskController struct {
	common *controllerCommon
}

func newManagedDiskController(common *controllerCommon) (ManagedDiskController, error) {
	return &managedDiskController{common: common}, nil
}

func (c *managedDiskController) AttachDisk(nodeName string, diskUri string, cacheMode string) (int, error) {
	// We don't need to validate if the disk is already attached
	// to a different VM. The VM update call below will fail if
	// it was attached somewhere else

	// this behaviour is expected in (i.e during a k8s drain node call)
	// k8s will evantually call detach->oldnode followed by attach->newnode
	var vmData interface{}
	vm, err := c.common.getArmVm(nodeName)

	if err != nil {
		return -1, err
	}

	if err := json.Unmarshal(vm, &vmData); err != nil {
		return -1, err
	}

	fragment := vmData.(map[string]interface{})

	// remove "resources" as ARM does not support PUT with "resources"
	delete(fragment, "resources")

	props := fragment["properties"].(map[string]interface{})
	hardwareProfile := props["hardwareProfile"].(map[string]interface{})
	vmSize := hardwareProfile["vmSize"].(string)
	storageProfile := props["storageProfile"].(map[string]interface{})

	managedVm := c.common.isManagedArmVm(storageProfile)
	if !managedVm {
		return -1, fmt.Errorf("azureDisk - error: attempt to attach managed disk %s to an unmanaged node  %s ", diskUri, nodeName)
	}

	dataDisks := storageProfile["dataDisks"].([]interface{})
	lun, err := findEmptyLun(vmSize, dataDisks)

	if err != nil {
		return -1, err
	}

	managedDiskInfo := &armVmManagedDiskInfo{Id: diskUri}
	newDisk := &armVmDataDisk{
		Caching:      cacheMode,
		CreateOption: "Attach",
		ManagedDisk:  managedDiskInfo,
		Lun:          lun,
	}
	dataDisks = append(dataDisks, newDisk)

	storageProfile["dataDisks"] = dataDisks // -> store back

	payload := new(bytes.Buffer)
	err = json.NewEncoder(payload).Encode(fragment)

	if err != nil {
		return -1, err
	}

	if err = c.common.updateArmVm(nodeName, payload); err != nil {
		return -1, err
	}

	// We don't need to poll ARM here, since WaitForAttach (running on node) will
	// be looping on the node to get devicepath /dev/sd* by lun#
	glog.V(2).Infof("azureDisk - Attached disk %s to node %s", diskUri, nodeName)

	return lun, err
}

func (c *managedDiskController) DetachDisk(nodeName string, hashedDiskId string) error {
	diskId := ""
	var vmData interface{}
	vm, err := c.common.getArmVm(nodeName)

	if err != nil {
		return err
	}

	if err := json.Unmarshal(vm, &vmData); err != nil {
		return err
	}

	fragment := vmData.(map[string]interface{})

	// remove "resources" as ARM does not support PUT with "resources"
	delete(fragment, "resources")
	props := fragment["properties"].(map[string]interface{})
	storageProfile := props["storageProfile"].(map[string]interface{})
	dataDisks, _ := storageProfile["dataDisks"].([]interface{})

	newDataDisks := make([]interface{}, 0)
	for _, v := range dataDisks {
		d := v.(map[string]interface{})
		md := d["managedDisk"].(map[string]interface{})

		currentDiskId := strings.ToLower(md["id"].(string))
		hashedCurrentDiskId := makeCRC32(currentDiskId)

		if hashedDiskId != hashedCurrentDiskId {
			newDataDisks = append(newDataDisks, d)
		} else {
			diskId = currentDiskId
		}
	}

	if diskId == "" {
		glog.Warningf("azureDisk - disk with hash %s was not found atached on node %s", hashedDiskId, nodeName)
		return nil
	}

	//get Disk Name
	diskName := path.Base(diskId)

	storageProfile["dataDisks"] = newDataDisks // -> store back
	payload := new(bytes.Buffer)
	err = json.NewEncoder(payload).Encode(fragment)

	if err != nil {
		return err
	}
	updateErr := c.common.updateArmVm(nodeName, payload)

	if updateErr != nil {
		glog.Infof("azureDisk - error while detaching a managed disk disk(%s) node(%s) error(%s)", diskId, nodeName, updateErr.Error())
		return updateErr
	}
	// poll
	// This is critical case, if this was a PVC, k8s will immediatly
	// attempt to delete the disk (according to policy)
	// a race condition will occure if we returned before
	// 1) disk is cleared from VM "dataDisks"
	// 2) disk status is not: unattached
	err = kwait.ExponentialBackoff(defaultBackOff, func() (bool, error) {
		// confirm that it is attached to the machine
		attached, _, err := c.common.isDiskAttached(hashedDiskId, nodeName, true)
		if err == nil && !attached {
			// confirm that the disk status has changed
			_, _, aState, err := c.getDisk(diskName)

			if err == nil && aState == "Unattached" {
				return true, nil
			} else {
				return false, err
			}
		} else {
			return false, err
		}
	})

	if err != nil {
		glog.V(2).Infof("azureDisk - detached disk %s from node %s but was unable to confirm complete complete-detach during poll", diskName, nodeName)
	} else {
		glog.V(2).Infof("azureDisk - detached disk %s from node %s", diskName, nodeName)
	}

	return nil
}

func (c *managedDiskController) CreateDataDisk(diskName string, storageAccountType string, sizeGB int, tags map[string]string) (string, error) {
	glog.V(4).Infof("azureDisk - dreating new managed Name:%s StorageAccountType:%s Size:%v", diskName, storageAccountType, sizeGB)

	if tags == nil {
		tags = make(map[string]string)
	}
	tags["created-by"] = "kubernetes/azure-dd"

	tagsPayload := new(bytes.Buffer)
	e := json.NewEncoder(tagsPayload).Encode(tags)

	if e != nil {
		glog.Infof("azureDisk - failed to encode tags for Azure Disk %s error: %s", diskName, e)
		return "", e
	}

	tagsString := tagsPayload.String()

	// Azure won't allow / (forward slash) in tags
	tagsString = strings.Replace(tagsString, "/", "-", -1)

	uri := fmt.Sprintf(diskEndPointTemplate, c.common.managementEndpoint, c.common.subscriptionId, c.common.resourceGroup, diskName, apiversion)

	requestData := `{ "tags" : ` + tagsString + `,   "location" : "` + c.common.location + `", "properties":  { "creationData":  {"createOption": "Empty" }, "accountType"  : "` + storageAccountType + `", "diskSizeGB": "` + strconv.Itoa(sizeGB) + `"  } }`

	client := &http.Client{}
	content := bytes.NewBufferString(requestData)
	r, err := http.NewRequest("PUT", uri, content)
	if err != nil {
		return "", err
	}

	token, err := c.common.getToken()
	if err != nil {
		return "", err
	}
	r.Header.Add("Content-Type", "application/json")
	r.Header.Add("Authorization", "Bearer "+token)

	resp, err := client.Do(r)
	if err != nil {
		return "", getRestError(fmt.Sprintf("Create Managed Disk: %s", diskName), err, 202, resp.StatusCode, resp.Body)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 202 {
		return "", getRestError(fmt.Sprintf("Create Managed Disk: %s", diskName), err, 202, resp.StatusCode, resp.Body)
	}

	diskId := fmt.Sprintf(diskIdTemplate, c.common.subscriptionId, c.common.resourceGroup, diskName)

	err = kwait.ExponentialBackoff(defaultBackOff, func() (bool, error) {
		exists, pState, _, err := c.getDisk(diskName)
		// We are waiting for Exists, provisioningState==Succeeded
		// We don't want to hand-off managed disks to k8s while they are
		//still being provisioned, this is to avoid some racy conditions
		if err != nil {
			return false, err
		} else {
			if exists && pState == "Succeeded" {
				return true, nil
			} else {
				return false, nil
			}
		}
	})

	if err != nil {
		glog.V(2).Infof("azureDisk - created new MD Name:%s StorageAccountType:%s Size:%v but was unable to confirm provisioningState in poll process", diskName, storageAccountType, sizeGB)
	} else {
		glog.V(2).Infof("azureDisk - created new MD Name:%s StorageAccountType:%s Size:%v", diskName, storageAccountType, sizeGB)
	}

	return diskId, nil
}

func (c *managedDiskController) DeleteDataDisk(diskUri string) error {
	diskName := path.Base(diskUri)
	uri := fmt.Sprintf(diskEndPointTemplate, c.common.managementEndpoint, c.common.subscriptionId, c.common.resourceGroup, diskName, apiversion)

	client := &http.Client{}
	r, err := http.NewRequest("DELETE", uri, nil)
	if err != nil {
		return nil
	}

	token, err := c.common.getToken()
	if err != nil {
		return err
	}

	r.Header.Add("Authorization", "Bearer "+token)

	resp, err := client.Do(r)
	if err != nil {
		return getRestError(fmt.Sprintf("Delete Managed Disk: %s", diskUri), err, 202, resp.StatusCode, resp.Body)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 202 {
		return getRestError(fmt.Sprintf("Delete Managed Disk: %s", diskUri), err, 202, resp.StatusCode, resp.Body)
	}

	// We don't need poll here, k8s will immediatly stop referencing the disk
	// the disk will be evantually deleted - cleanly - by ARM

	glog.V(2).Infof("azureDisk - deleted a managed disk: %s", diskUri)

	return nil
}

func (c *managedDiskController) getDisk(diskName string) (bool, string, string, error) {
	uri := fmt.Sprintf(diskEndPointTemplate, c.common.managementEndpoint, c.common.subscriptionId, c.common.resourceGroup, diskName, apiversion)

	client := &http.Client{}
	r, err := http.NewRequest("GET", uri, nil)
	if err != nil {
		return false, "", "", nil
	}

	token, err := c.common.getToken()
	if err != nil {
		return false, "", "", err
	}

	r.Header.Add("Authorization", "Bearer "+token)

	resp, err := client.Do(r)
	if err != nil {
		return false, "", "", getRestError("Get Managed Disk", err, 200, resp.StatusCode, resp.Body)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		newError := getRestError("Get Managed Disk", err, 200, resp.StatusCode, resp.Body)
		// log the new formatted error and return the original error
		glog.Infof(newError.Error())
		return false, "", "", err
	}

	bodyBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return false, "", "", err
	}

	var disk interface{}

	if err := json.Unmarshal(bodyBytes, &disk); err != nil {
		return false, "", "", err
	}

	// Extract Provisioning State & Disk State

	diskState := ""
	provisioningState := ""

	fragment := disk.(map[string]interface{})
	properties := fragment["properties"].(map[string]interface{})
	provisioningState = properties["provisioningState"].(string) // if there is a disk, provisioningState property will be there
	if ref, ok := properties["diskState"]; ok {
		diskState = ref.(string)
	}

	return true, provisioningState, diskState, nil
}
