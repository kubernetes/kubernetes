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
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	kwait "k8s.io/apimachinery/pkg/util/wait"

	"github.com/golang/glog"
)

type armVmVhdDiskInfo struct {
	Uri string `json:"uri"`
}
type armVmManagedDiskInfo struct {
	Id string `json:"id"`
}

type armVmDataDisk struct {
	Lun          int                   `json:"lun"`
	Name         string                `json:"name,omitempty"`
	CreateOption string                `json:"createOption"`
	ManagedDisk  *armVmManagedDiskInfo `json:"managedDisk,omitempty"`
	Vhd          *armVmVhdDiskInfo     `json:"vhd,omitempty"`
	Caching      string                `json:"caching"`
	DiskSizeGB   int                   `json:"diskSizeGB,omitempty"`
}

// interfaces used by attacher, deleter to communicate with ARM
type BlobDiskController interface {
	CreateDataDisk(dataDiskName string, storageAccountType string, sizeGB int, forceStandAlone bool) (string, error)
	DeleteDataDisk(diskUri string, wasForced bool) error

	AttachDisk(nodeName string, diskUri string, cacheMode string) (int, error)
	DetachDisk(nodeName string, hashedDiskUri string) error
}

type ManagedDiskController interface {
	CreateDataDisk(diskName string, storageAccountType string, sizeGB int, tags map[string]string) (string, error)
	DeleteDataDisk(diskUri string) error

	AttachDisk(nodeName string, diskUri string, cacheMode string) (int, error)
	DetachDisk(nodeName string, hashedDiskUri string) error
}

const (
	aadTokenEndPointPath string = "%s/oauth2/token/"
	apiversion           string = "2016-04-30-preview"
	//_apiversionVhdVm      string = "2016-03-30"
	diskEndPointTemplate string = "%ssubscriptions/%s/resourcegroups/%s/providers/microsoft.compute/disks/%s?api-version=%s"
	vMEndPointTemplate   string = "%ssubscriptions/%s/resourcegroups/%s/providers/microsoft.compute/virtualmachines/%s?api-version=%s"
	diskIdTemplate       string = "/subscriptions/%s/resourcegroups/%s/providers/microsoft.compute/disks/%s"
	defaultDataDiskCount int    = 16 // which will allow you to work with most medium size VMs (if not found in map)

	vhdBlobUriTemplate             = "https://%s.blob.core.windows.net/%s/%s"
	storageAccountUriTemplate      = "https://%s.blob.core.windows.net/%s?%s"
	storageAccountNameTemplate     = "pvc%s"
	storageAccountEndPointTemplate = "%ssubscriptions/%s/resourcegroups/%s/providers/microsoft.storage/storageaccounts/%s?api-version=2016-01-01"

	// for limits check https://docs.microsoft.com/en-us/azure/azure-subscription-service-limits#storage-limits
	maxStorageAccounts                     = 100 // max # is 200 (250 with special request). this allows 100 for everything else including stand alone disks
	maxDisksPerStorageAccounts             = 60
	storageAccountUtilizationBeforeGrowing = 0.5
	storageAccountsCountInit               = 2 // When the plug-in is init-ed, 2 storage accounts will be created to allow fast pvc create/attach/mount
)

var defaultBackOff = kwait.Backoff{
	Steps:    20,
	Duration: 2 * time.Second,
	Factor:   1.5,
	Jitter:   0.0,
}

var time1970 time.Time = time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC)

type controllerCommon struct {
	tenantId            string
	subscriptionId      string
	location            string
	resourceGroup       string
	clientId            string
	clientSecret        string
	managementEndpoint  string
	tokenEndPoint       string
	aadResourceEndPoint string
	aadToken            string
	expires_on          time.Time
}

func (c *controllerCommon) isManagedArmVm(storageProfile map[string]interface{}) bool {
	osDisk := storageProfile["osDisk"].(map[string]interface{})
	if _, ok := osDisk["managedDisk"]; ok {
		return true
	}
	return false
}

func (c *controllerCommon) getAttachedDisks(nodeName string) ([]string, error) {
	var disks []string
	var vmData interface{}
	vm, err := c.getArmVm(nodeName)

	if err != nil {
		return nil, err
	}

	if err := json.Unmarshal(vm, &vmData); err != nil {
		return disks, err
	}

	fragment := vmData.(map[string]interface{})

	props := fragment["properties"].(map[string]interface{})
	storageProfile := props["storageProfile"].(map[string]interface{})
	dataDisks, _ := storageProfile["dataDisks"].([]interface{})

	// we silently ignore, if VM does not have the disk attached
	for _, v := range dataDisks {
		d := v.(map[string]interface{})
		if _, ok := d["vhd"]; ok {
			// this is a blob disk
			vhdInfo := d["vhd"].(map[string]interface{})
			vhdUri := vhdInfo["uri"].(string)
			disks = append(disks, vhdUri)
		} else {
			// this is managed disk
			managedDiskInfo := d["managedDisk"].(map[string]interface{})
			managedDiskId := managedDiskInfo["id"].(string)
			disks = append(disks, managedDiskId)
		}
	}
	return disks, nil
}

// if disk attached returns bool + lun attached to
func (c *controllerCommon) isDiskAttached(hashedDiskUri, nodeName string, isManaged bool) (attached bool, lun int, err error) {
	attached = false
	lun = -1

	var vmData interface{}

	vm, err := c.getArmVm(nodeName)

	if err != nil {
		return attached, lun, err
	}

	if err := json.Unmarshal(vm, &vmData); err != nil {
		return attached, lun, err
	}

	fragment := vmData.(map[string]interface{})
	props := fragment["properties"].(map[string]interface{})
	storageProfile := props["storageProfile"].(map[string]interface{})
	dataDisks, _ := storageProfile["dataDisks"].([]interface{})

	for _, v := range dataDisks {
		d := v.(map[string]interface{})
		if isManaged {
			md := d["managedDisk"].(map[string]interface{})
			currentDiskId := strings.ToLower(md["id"].(string))
			hashedCurrentDiskId := makeCRC32(currentDiskId)
			if hashedCurrentDiskId == hashedDiskUri {
				attached = true
				lun = int(d["lun"].(float64))
				break
			}
		} else {
			blobDisk := d["vhd"].(map[string]interface{})
			blobDiskUri := blobDisk["uri"].(string)
			hashedBlobDiskUri := makeCRC32(blobDiskUri)
			if hashedBlobDiskUri == hashedDiskUri {
				attached = true
				lun = int(d["lun"].(float64))
				break
			}
		}
	}

	return attached, lun, nil
}

// Updates an arm VM based on the payload
func (c *controllerCommon) updateArmVm(armVmName string, buffer *bytes.Buffer) error {
	uri := fmt.Sprintf(vMEndPointTemplate, c.managementEndpoint, c.subscriptionId, c.resourceGroup, armVmName, apiversion)
	client := &http.Client{}
	r, err := http.NewRequest("PUT", uri, buffer)
	if err != nil {
		return err
	}

	token, err := c.getToken()
	if err != nil {
		return err
	}

	r.Header.Add("Content-Type", "application/json")
	r.Header.Add("Authorization", "Bearer "+token)
	resp, err := client.Do(r)

	if err != nil || resp.StatusCode != 200 {
		return getRestError(fmt.Sprintf("Update ARM VM: %s", armVmName), err, 200, resp.StatusCode, resp.Body)
	}
	return nil
}

// Gets ARM VM
func (c *controllerCommon) getArmVm(armVmName string) ([]byte, error) {
	uri := fmt.Sprintf(vMEndPointTemplate, c.managementEndpoint, c.subscriptionId, c.resourceGroup, armVmName, apiversion)
	client := &http.Client{}
	r, err := http.NewRequest("GET", uri, nil)
	if err != nil {
		return nil, err
	}

	token, err := c.getToken()
	if err != nil {
		return nil, err
	}

	r.Header.Add("Authorization", "Bearer "+token)
	resp, err := client.Do(r)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()
	if err != nil || resp.StatusCode != 200 {
		return nil, getRestError(fmt.Sprintf("Get ARM VM: %s", armVmName), err, 200, resp.StatusCode, resp.Body)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return body, nil
}

func (c *controllerCommon) getToken() (token string, err error) {
	if c.aadToken != "" && time.Now().UTC().Sub(c.expires_on).Seconds() <= 10 {
		// token cached and is valid.
		return c.aadToken, nil
	}

	apiUrl := c.tokenEndPoint
	resource := fmt.Sprintf(aadTokenEndPointPath, c.tenantId)
	// create form urlencoded post data
	formData := url.Values{}

	formData.Add("grant_type", "client_credentials")
	formData.Add("client_id", c.clientId)
	formData.Add("client_secret", c.clientSecret)
	formData.Add("resource", c.aadResourceEndPoint)

	urlStr := ""
	u := &url.URL{}
	if u, err = url.ParseRequestURI(apiUrl); err != nil {
		return "", err
	}

	u.Path = resource
	urlStr = fmt.Sprintf("%v", u)
	client := &http.Client{}

	r, err := http.NewRequest("POST", urlStr, bytes.NewBufferString(formData.Encode()))
	if err != nil {
		return "", err
	}

	// add headers
	r.Header.Add("Content-Type", "application/x-www-form-urlencoded")
	r.Header.Add("Content-Length", strconv.Itoa(len(formData.Encode())))
	resp, err := client.Do(r)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	payload, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	c.aadToken, c.expires_on, err = parseAADToken(payload)
	return c.aadToken, err
}

func parseAADToken(payload []byte) (string, time.Time, error) {
	var f interface{}
	var sToken string
	var expiresOn time.Time
	var expiresSec int
	var ok bool
	var expires string

	if err := json.Unmarshal(payload, &f); err != nil {
		return "", expiresOn, err
	}

	fragment := f.(map[string]interface{})
	if sToken, ok = fragment["access_token"].(string); ok != true {
		return "", expiresOn, fmt.Errorf("Disk controller (ARM Client) cannot parse AAD token - access_token field")
	}
	if expires, ok = fragment["expires_on"].(string); ok != true {
		return "", expiresOn, fmt.Errorf("Disk controller (ARM Client) cannot parse AAD token - expires_on field")
	}

	expiresSec, _ = strconv.Atoi(expires)
	// expires_on is seconds since 1970-01-01T0:0:0Z UTC
	expiresOn = time1970.UTC().Add(time.Duration(expiresSec) * time.Second)

	return sToken, expiresOn, nil
}

// Creats a slice  luns based on the VM size, used the static map declared on this package
func getLunMapForVm(vmSize string) []bool {
	count, ok := dataDisksPerVM[vmSize]
	if !ok {
		glog.Warningf("azureDisk - VM Size %s found no static lun count will use default which  %v", vmSize, defaultDataDiskCount)
		count = defaultDataDiskCount
	}

	m := make([]bool, count)
	return m
}

// finds an empty based on VM size and current attached disks
func findEmptyLun(vmSize string, dataDisks []interface{}) (int, error) {
	vmLuns := getLunMapForVm(vmSize)
	selectedLun := -1

	for _, v := range dataDisks {
		current := v.(map[string]interface{})
		lun := int(current["lun"].(float64))
		vmLuns[lun] = true
	}

	//find first empty lun
	for i, v := range vmLuns {
		if v == false {
			selectedLun = i
			break
		}
	}

	if selectedLun == -1 {
		return selectedLun, fmt.Errorf("azureDisk - failed to find empty lun on VM type:%s total-luns-checked:%v", vmSize, len(vmLuns))
	}

	return selectedLun, nil
}

func getRestError(operation string, restError error, expectedStatus int, actualStatus int, body io.ReadCloser) error {
	if restError != nil {
		return fmt.Errorf("azureDisk - %s - Rest Error: %s", operation, restError)
	}

	bodystr := ""
	if body != nil {
		bodyBytes, _ := ioutil.ReadAll(body)
		if bodyBytes != nil {
			bodystr = string(bodyBytes)
		}
	}
	return fmt.Errorf("azureDisk - %s - Rest Status Error:\n Expected: %v\n Got: %v\n ResponseBody:%s\n\n", operation, expectedStatus, actualStatus, bodystr)
}
