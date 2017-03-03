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
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"net/http"
	"os"
	"sync"

	"strconv"
	"strings"
	"sync/atomic"
	"time"

	azstorage "github.com/Azure/azure-sdk-for-go/storage"
	"github.com/golang/glog"
	"github.com/rubiojr/go-vhd/vhd"
	kwait "k8s.io/apimachinery/pkg/util/wait"
)

type storageAccountState struct {
	name                    string
	saType                  string
	key                     string
	diskCount               int32
	isValidating            int32
	defaultContainerCreated bool
}

type blobDiskController struct {
	common   *controllerCommon
	accounts map[string]*storageAccountState
}

var defaultContainerName = ""
var storageAccountNamePrefix string = ""
var storageAccountNameMatch string = ""
var initFlag int64 = 0

var accountsLock = &sync.Mutex{}

func newBlobDiskController(common *controllerCommon) (BlobDiskController, error) {
	c := blobDiskController{common: common}
	err := c.init()

	if err != nil {
		return nil, err
	}

	return &c, nil
}

// attaches a disk to node and return lun # as string
func (c *blobDiskController) AttachDisk(nodeName string, diskUri string, cacheMode string) (int, error) {
	// K8s in case of existing pods evication, will automatically attepmt to attach volumes
	// to a different node. Though it *knows* which disk attached to which node.
	// the following guards against this behaviour

	// avoid:
	// Azure in case of blob disks, does not maintain a list of vhd:attached-to:node
	// The call  attach-to will fail after it was OK on the ARM VM endpoint
	// possibly putting the entire VM in *failed* state
	noLease, e := c.diskHasNoLease(diskUri)
	if e != nil {
		return -1, e
	}

	if !noLease {
		return -1, fmt.Errorf("azureDisk - disk %s still have leases on it. Will not be able to attach to node %s", diskUri, nodeName)
	}

	var vmData interface{}
	_, diskName, err := diskNameandSANameFromUri(diskUri)
	if err != nil {
		return -1, err
	}

	vm, err := c.common.getArmVm(nodeName)
	if err != nil {
		return 0, err
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
	if managedVm {
		return -1, fmt.Errorf("azureDisk - error: attempt to attach blob disk %s to an managed node  %s ", diskName, nodeName)
	}

	dataDisks := storageProfile["dataDisks"].([]interface{})
	lun, err := findEmptyLun(vmSize, dataDisks)

	if err != nil {
		return -1, err
	}

	newDisk := &armVmDataDisk{
		Name:         diskName,
		Caching:      cacheMode,
		CreateOption: "Attach",
		//DiskSizeGB:   sizeGB,
		Vhd: &armVmVhdDiskInfo{Uri: diskUri},
		Lun: lun,
	}

	dataDisks = append(dataDisks, newDisk)
	storageProfile["dataDisks"] = dataDisks // -> store back

	payload := new(bytes.Buffer)
	err = json.NewEncoder(payload).Encode(fragment)

	if err != nil {
		return -1, err
	}

	err = c.common.updateArmVm(nodeName, payload)
	if err != nil {
		return -1, err
	}

	// We don't need to poll ARM here, since WaitForAttach (running on node) will
	// be looping on the node to get devicepath /dev/sd* by lun#
	glog.V(2).Infof("azureDisk - Attached disk %s to node %s", diskName, nodeName)
	return lun, nil
}

// detaches disk from a node
func (c *blobDiskController) DetachDisk(nodeName string, hashedDiskUri string) error {
	diskUri := ""
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

	// we silently ignore, if VM does not have the disk attached
	newDataDisks := make([]interface{}, 0)
	for _, v := range dataDisks {
		d := v.(map[string]interface{})
		vhdInfo := d["vhd"].(map[string]interface{})
		vhdUri := vhdInfo["uri"].(string)
		hashedVhdUri := makeCRC32(vhdUri)
		if hashedDiskUri != hashedVhdUri {
			dataDisks = append(dataDisks, v)
		} else {
			diskUri = vhdUri
		}

	}

	// no disk found
	if diskUri == "" {
		glog.Warningf("azureDisk - disk with hash %s was not found atached on node %s", hashedDiskUri, nodeName)
		return nil
	}

	storageProfile["dataDisks"] = newDataDisks // -> store back
	payload := new(bytes.Buffer)
	err = json.NewEncoder(payload).Encode(fragment)
	if err != nil {
		return err
	}
	updateErr := c.common.updateArmVm(nodeName, payload)
	if updateErr != nil {
		return updateErr
	}

	// Wait for ARM to remove the disk from datadisks collection on the VM
	err = kwait.ExponentialBackoff(defaultBackOff, func() (bool, error) {
		attached, _, err := c.common.isDiskAttached(hashedDiskUri, nodeName, false)
		if err == nil && !attached {
			return true, nil
		}
		return false, err
	})

	if err != nil {

		// confirm that the blob has no leases on it
		err = kwait.ExponentialBackoff(defaultBackOff, func() (bool, error) {
			var e error

			noLease, e := c.diskHasNoLease(diskUri)
			if e != nil {
				glog.Infof("azureDisk - failed to check if disk %s still has leases on it, we will assume clean-detach. Err:%s", diskUri, e.Error())
				return true, nil
			}

			if noLease {
				return true, nil
			}

			return false, nil
		})
	}

	if err != nil {
		glog.V(4).Infof("azureDisk - detached blob disk %s from node %s but was unable to confirm complete clean-detach during poll", diskUri, nodeName)
	} else {
		glog.V(4).Infof("azurDisk - detached blob disk %s from node %s", diskUri, nodeName)
	}

	return nil
}

func (c *blobDiskController) CreateDataDisk(dataDiskName string, storageAccountType string, sizeGB int, forceStandAlone bool) (string, error) {
	glog.V(4).Infof("azureDisk - creating blob data disk named:%s on StorageAccountType:%s StandAlone:%v", dataDiskName, storageAccountType, forceStandAlone)

	var storageAccountName = ""
	var err error
	sizeBytes := 1024 * 1024 * 1024 * int64(sizeGB)
	vhdName := dataDiskName + ".vhd"
	totalVhdSize := sizeBytes + vhd.VHD_HEADER_SIZE

	if forceStandAlone {
		// we have to wait until the storage account is is created
		storageAccountName = "p" + makeCRC32(c.common.subscriptionId+c.common.resourceGroup+dataDiskName)
		err = c.createStorageAccount(storageAccountName, storageAccountType, false)
		if err != nil {
			return "", err
		}
	} else {
		storageAccountName, err = c.findSANameForDisk(storageAccountType)
		if err != nil {
			return "", err
		}
	}

	blobSvc, err := c.getBlobSvcClient(storageAccountName)

	if err != nil {
		return "", err
	}

	tags := make(map[string]string)

	tags["created-by"] = "k8s-azure-DataDisk"

	glog.V(4).Infof("azureDisk - creating page blob for data disk %s\n", dataDiskName)

	if err := blobSvc.PutPageBlob(defaultContainerName, vhdName, totalVhdSize, tags); err != nil {
		glog.Infof("azureDisk - Failed to put page blob on account %s for data disk %s error was %s \n", storageAccountName, dataDiskName, err.Error())
		return "", err
	}

	vhdBytes, err := createVHDHeader(uint64(sizeBytes))

	if err != nil {
		glog.Infof("azureDisk - failed to load vhd asset for data disk %s size %v\n", dataDiskName, sizeGB)
		blobSvc.DeleteBlobIfExists(defaultContainerName, vhdName, nil)
		return "", err
	}

	headerBytes := vhdBytes[:vhd.VHD_HEADER_SIZE]

	if err = blobSvc.PutPage(defaultContainerName, vhdName, sizeBytes, totalVhdSize-1, azstorage.PageWriteTypeUpdate, headerBytes, nil); err != nil {
		_, _ = blobSvc.DeleteBlobIfExists(defaultContainerName, vhdName, nil)
		glog.Infof("azureDisk - failed to put header page for data disk %s on account %s error was %s\n", storageAccountName, dataDiskName, err.Error())
		return "", err
	}

	if !forceStandAlone {
		atomic.AddInt32(&c.accounts[storageAccountName].diskCount, 1)
	}

	return fmt.Sprintf(vhdBlobUriTemplate, storageAccountName, defaultContainerName, vhdName), nil
}

func (c *blobDiskController) DeleteDataDisk(diskUri string, wasForced bool) error {
	storageAccountName, vhdName, err := diskNameandSANameFromUri(diskUri)
	if err != nil {
		return err
	}
	// if forced (as in one disk = one storage account)
	// delete the account completely
	if wasForced {
		return c.deleteStorageAccount(storageAccountName)
	}

	blobSvc, err := c.getBlobSvcClient(storageAccountName)
	if err != nil {
		return err
	}

	glog.V(2).Infof("azureDisk - About to delete vhd file %s on storage account %s container %s", vhdName, storageAccountName, defaultContainerName)

	_, err = blobSvc.DeleteBlobIfExists(defaultContainerName, vhdName, nil)

	if c.accounts[storageAccountName].diskCount == -1 {
		if diskCount, err := c.getDiskCount(storageAccountName); err != nil {
			c.accounts[storageAccountName].diskCount = int32(diskCount)
		} else {
			glog.Warningf("azureDisk - failed to get disk count for %s however the delete disk operation was ok", storageAccountName)
			return nil // we have failed to aquire a new count. not an error condition
		}
	}
	atomic.AddInt32(&c.accounts[storageAccountName].diskCount, -1)
	return err
}

func (c *blobDiskController) diskHasNoLease(diskUri string) (bool, error) {
	if !strings.Contains(diskUri, defaultContainerName) {
		// if the disk was attached via PV (with possibility of existing out side
		// this RG), we will have to drop this check, as we are not sure if we can
		// get keys for this account
		glog.Infof("azureDisk - assumed that disk %s is not provisioned via PV and will not check if it has leases on it", diskUri)
		return true, nil
	}

	diskStorageAccount, vhdName, err := diskNameandSANameFromUri(diskUri)
	if err != nil {
		glog.Infof("azureDisk - could not check if disk %s has a lease on it (diskNameandSANameFromUri):%s", diskUri, err.Error())
		return false, err
	}

	blobSvc, e := c.getBlobSvcClient(diskStorageAccount)
	if e != nil {
		glog.Infof("azureDisk - could not check if disk %s has a lease on it (getBlobSvcClient):%s", diskUri, err.Error())
		return false, e
	}

	metaMap := make(map[string]string)
	metaMap["azureddheck"] = "ok"
	e = blobSvc.SetBlobMetadata(defaultContainerName, vhdName, metaMap, nil)
	if e != nil {
		// disk has lease on it or does not exist, in both cases it something we can not go forward with
		return false, nil
	}
	return true, nil
}

// Init tries best effort to ensure that 2 accounts standard/premium were craeted
// to be used by shared blob disks. This to increase the speed pvc provisioning (in most of cases)
func (c *blobDiskController) init() error {
	if !c.shouldInit() {
		return nil
	}

	c.setUniqueStrings()

	// get accounts
	accounts, err := c.getAllStorageAccounts()
	if err != nil {
		return err
	}
	c.accounts = accounts

	if len(c.accounts) == 0 {
		counter := 1
		for counter <= storageAccountsCountInit {

			accountType := "premium_lrs"
			if n := math.Mod(float64(counter), 2); n == 0 {
				accountType = "standard_lrs"
			}

			// We don't really care if these calls failed
			// at this stage, we are trying to ensure 2 accounts (Standard/Premium)
			// are there ready for PVC creation

			// if we failed here, the accounts will be created in the process
			// of creating PVC

			// nor do we care if they were partially created, as the entire
			// account creation process is idempotent
			go func(thisNext int) {
				newAccountName := getAccountNameForNum(thisNext)

				glog.Infof("azureDisk - blobDiskController init process  will create new storageAccount:%s type:%s", newAccountName, accountType)
				err := c.createStorageAccount(newAccountName, accountType, true)
				// TODO return created and error from
				if err != nil {
					glog.Infof("azureDisk - blobDiskController init: create account %s with error:%s", newAccountName, err.Error())

				} else {
					glog.Infof("azureDisk - blobDiskController init: created account %s", newAccountName)
				}
			}(counter)
			counter = counter + 1
		}
	}

	return nil
}

//Sets unique strings to be used as accountnames && || blob containers names
func (c *blobDiskController) setUniqueStrings() {
	uniqueString := c.common.resourceGroup + c.common.location + c.common.subscriptionId
	hash := makeCRC32(uniqueString)
	//used to generate a unqie container name used by this cluster PVC
	defaultContainerName = hash

	storageAccountNamePrefix = fmt.Sprintf(storageAccountNameTemplate, hash)
	// Used to filter relevant accounts (accounts used by shared PVC)
	storageAccountNameMatch = storageAccountNamePrefix
	// Used as a template to create new names for relevant accounts
	storageAccountNamePrefix = storageAccountNamePrefix + "%s"
}
func (c *blobDiskController) getStorageAccountKey(SAName string) (string, error) {
	if account, exists := c.accounts[SAName]; exists && account.key != "" {
		return c.accounts[SAName].key, nil
	}

	uri := fmt.Sprintf(storageAccountEndPointTemplate,
		c.common.managementEndpoint,
		c.common.subscriptionId,
		c.common.resourceGroup,
		SAName+"/listkeys")

	client := &http.Client{}
	r, err := http.NewRequest("POST", uri, nil)
	if err != nil {
		return "", err
	}

	token, err := c.common.getToken()

	if err != nil {
		return "", err
	}

	r.Header.Add("Authorization", "Bearer "+token)

	resp, err := client.Do(r)
	if err != nil || resp.StatusCode != 200 {
		newError := getRestError("GetStorageAccountKeys", err, 200, resp.StatusCode, resp.Body)
		return "", newError
	}

	defer resp.Body.Close()

	bodyBytes, err := ioutil.ReadAll(resp.Body)

	if err != nil {
		return "", err
	}

	var payload interface{}

	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return "", err
	}

	fragment := payload.(map[string]interface{})
	keys := fragment["keys"].([]interface{})
	for _, v := range keys {
		key := v.(map[string]interface{})

		if key["keyName"].(string) == "key1" {
			accountKey := key["value"].(string)
			if _, ok := c.accounts[SAName]; !ok {
				glog.Warningf("azureDisk - account %s was not cached while getting keys", SAName)
				return accountKey, nil
			}

			c.accounts[SAName].key = key["value"].(string)
			return c.accounts[SAName].key, nil
		}
	}

	return "", fmt.Errorf("couldn't find key named key1 in storage account:%s keys", SAName)
}

func (c *blobDiskController) getBlobSvcClient(SAName string) (azstorage.BlobStorageClient, error) {
	key := ""
	var client azstorage.Client
	var blobSvc azstorage.BlobStorageClient
	var err error
	if key, err = c.getStorageAccountKey(SAName); err != nil {
		return blobSvc, err
	}

	if client, err = azstorage.NewBasicClient(SAName, key); err != nil {
		return blobSvc, err
	}

	blobSvc = client.GetBlobService()
	return blobSvc, nil
}

func (c *blobDiskController) ensureDefaultContainer(storageAccountName string) error {
	var bExist bool
	var provisionState string
	var err error
	var blobSvc azstorage.BlobStorageClient

	// short circut the check via local cache
	// we are forgiving the fact that account may not be in cache yet
	if v, ok := c.accounts[storageAccountName]; ok && v.defaultContainerCreated {
		return nil
	}

	// not cached, check existance and readiness
	bExist, provisionState, _ = c.getStorageAccount(storageAccountName)

	// account does not exist
	if !bExist {
		return fmt.Errorf("azureDisk - account %s does not exist while trying to create/ensure default container", storageAccountName)
	}

	// account exists but not ready yet
	if provisionState != "Succeeded" {
		// we don't want many attempts to validate the account readiness
		// here hence we are locking
		counter := 1
		for swapped := atomic.CompareAndSwapInt32(&c.accounts[storageAccountName].isValidating, 0, 1); swapped != true; {
			time.Sleep(3 * time.Second)
			counter = counter + 1
			// check if we passed the max sleep
			if counter >= 20 {
				return fmt.Errorf("azureDisk - timeout waiting to aquire lock to validate account:%s readiness", storageAccountName)
			}
		}

		// swapped
		defer func() {
			c.accounts[storageAccountName].isValidating = 0
		}()

		// short circut the check again.
		if v, ok := c.accounts[storageAccountName]; ok && v.defaultContainerCreated {
			return nil
		}

		err = kwait.ExponentialBackoff(defaultBackOff, func() (bool, error) {
			_, provisionState, err := c.getStorageAccount(storageAccountName)

			if err != nil {
				glog.V(4).Infof("azureDisk - GetStorageAccount:%s err %s", storageAccountName, err.Error())
				return false, err
			}

			if provisionState == "Succeeded" {
				return true, nil
			} else {

				glog.V(4).Infof("azureDisk - GetStorageAccount:%s not ready yet", storageAccountName)
				// leave it for next loop/sync loop
				return false, fmt.Errorf("azureDisk - Account %s has not been flagged Succeeded by ARM", storageAccountName)
			}
		})
		// we have failed to ensure that account is ready for us to create
		// the default vhd container
		if err != nil {
			return err
		}
	}

	if blobSvc, err = c.getBlobSvcClient(storageAccountName); err != nil {
		return err
	}

	if bCreated, err := blobSvc.CreateContainerIfNotExists(defaultContainerName, azstorage.ContainerAccessType("")); err != nil {
		return err
	} else {
		if bCreated {
			glog.V(2).Infof("azureDisk - storage account:%s had no default container(%s) and it was created \n", storageAccountName, defaultContainerName)
		}
	}

	// flag so we no longer have to check on ARM
	c.accounts[storageAccountName].defaultContainerCreated = true
	return nil
}

// Gets Disk counts per storage account
func (c *blobDiskController) getDiskCount(SAName string) (int, error) {
	// if we have it in cache
	if c.accounts[SAName].diskCount != -1 {
		return int(c.accounts[SAName].diskCount), nil
	}

	var err error
	var blobSvc azstorage.BlobStorageClient

	if err = c.ensureDefaultContainer(SAName); err != nil {
		return 0, err
	}

	if blobSvc, err = c.getBlobSvcClient(SAName); err != nil {
		return 0, err
	}
	params := azstorage.ListBlobsParameters{}
	if response, err := blobSvc.ListBlobs(defaultContainerName, params); err != nil {
		return 0, err
	} else {

		glog.V(4).Infof("azure-Disk -  refreshed data count for account %s and found %v", SAName, len(response.Blobs))
		c.accounts[SAName].diskCount = int32(len(response.Blobs))
	}

	return int(c.accounts[SAName].diskCount), nil
}

// shouldInit ensures that we only init the plugin once
// and we only do that in the controller

func (c *blobDiskController) shouldInit() bool {
	if os.Args[0] == "kube-controller-manager" || (os.Args[0] == "/hyperkube" && os.Args[1] == "controller-manager") {
		swapped := atomic.CompareAndSwapInt64(&initFlag, 0, 1)
		if swapped {
			return true
		}
	}
	return false
}

func (c *blobDiskController) getAllStorageAccounts() (map[string]*storageAccountState, error) {
	uri := fmt.Sprintf(storageAccountEndPointTemplate, c.common.managementEndpoint, c.common.subscriptionId, c.common.resourceGroup, "")
	client := &http.Client{}
	r, err := http.NewRequest("GET", uri, nil)

	if err != nil {
		return nil, err
	}

	token, err := c.common.getToken()
	if err != nil {
		return nil, err
	}

	r.Header.Add("Authorization", "Bearer "+token)

	resp, err := client.Do(r)

	if err != nil || resp.StatusCode != 200 {
		newError := getRestError("GetAllStorageAccounts", err, 200, resp.StatusCode, resp.Body)
		glog.Infof(newError.Error())
		return nil, newError
	}

	defer resp.Body.Close()

	bodyBytes, err := ioutil.ReadAll(resp.Body)

	if err != nil {
		return nil, err
	}

	var payload interface{}

	accounts := make(map[string]*storageAccountState)
	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return nil, err
	}

	fragment := payload.(map[string]interface{})
	value := fragment["value"].([]interface{})

	for _, entry := range value {
		sa := entry.(map[string]interface{})
		name := sa["name"].(string)

		if strings.Index(name, storageAccountNameMatch) != 0 {
			continue
		}

		glog.Infof("azureDisk - identified account %s as part of shared PVC accounts", name)
		sku := sa["sku"].(map[string]interface{})
		skuName := sku["name"].(string)

		sastate := &storageAccountState{
			name:      name,
			saType:    strings.ToLower(skuName),
			diskCount: -1,
		}

		accounts[name] = sastate
	}

	return accounts, nil
}

func (c *blobDiskController) createStorageAccount(storageAccountName string, storageAccountType string, checkMaxAccounts bool) error {
	bExist, _, _ := c.getStorageAccount(storageAccountName)
	if bExist {
		newAccountState := &storageAccountState{
			diskCount: -1,
			saType:    storageAccountType,
			name:      storageAccountName,
		}

		c.addAccountState(storageAccountName, newAccountState)
	}
	// Account Does not exist
	if !bExist {
		if len(c.accounts) == maxStorageAccounts && checkMaxAccounts {
			return fmt.Errorf("azureDisk - can not create new storage account, current storage accounts count:%v Max is:%v", len(c.accounts), maxStorageAccounts)
		}

		glog.V(2).Infof("azureDisk - Creating storage account %s type %s \n", storageAccountName, storageAccountType)

		sPayload := `{  "location" : "%s", 
                  "tags" : {"created-by" : "azure-dd"} , 
                  "sku": {"name" : "%s"}, 
                  "kind" : "Storage"  
              }`

		sPayload = fmt.Sprintf(sPayload, c.common.location, storageAccountType)
		uri := fmt.Sprintf(storageAccountEndPointTemplate,
			c.common.managementEndpoint,
			c.common.subscriptionId,
			c.common.resourceGroup,
			storageAccountName)

		client := &http.Client{}
		content := bytes.NewBufferString(sPayload)
		r, err := http.NewRequest("PUT", uri, content)
		if err != nil {
			return err
		}

		token, err := c.common.getToken()

		if err != nil {
			return err
		}

		r.Header.Add("Content-Type", "application/json")
		r.Header.Add("Authorization", "Bearer "+token)

		resp, err := client.Do(r)

		defer resp.Body.Close()
		if err != nil || resp.StatusCode != 202 {
			return getRestError(fmt.Sprintf("Create Storage Account: %s", storageAccountName), err, 200, resp.StatusCode, resp.Body)
		}

		newAccountState := &storageAccountState{
			diskCount: -1,
			saType:    storageAccountType,
			name:      storageAccountName,
		}

		c.addAccountState(storageAccountName, newAccountState)
	}

	if !bExist {
		// SA Accounts takes time to be provisioned
		// so if this account was just created allow it sometime
		// before polling
		glog.V(4).Infof("azureDisk - storage account %s was just created, allowing time before polling status")
		time.Sleep(25 * time.Second) // as observed 25 is the average time for SA to be provisioned
	}

	// finally, make sure that we default container is created
	// before handing it back over
	return c.ensureDefaultContainer(storageAccountName)
}

// finds a new suitable storageAccount for this disk
func (c *blobDiskController) findSANameForDisk(storageAccountType string) (string, error) {
	maxDiskCount := maxDisksPerStorageAccounts
	SAName := ""
	totalDiskCounts := 0
	countAccounts := 0 // account of this type.
	for _, v := range c.accounts {
		// filter out any stand-alone disks/accounts
		if strings.Index(v.name, storageAccountNameMatch) != 0 {
			continue
		}

		// note: we compute avge stratified by type.
		// this to enable user to grow per SA type to avoid low
		//avg utilization on one account type skewing all data.

		if v.saType == storageAccountType {
			// compute average
			dCount, err := c.getDiskCount(v.name)
			if err != nil {
				return "", err
			}
			totalDiskCounts = totalDiskCounts + dCount
			countAccounts = countAccounts + 1
			// empty account
			if dCount == 0 {
				glog.V(4).Infof("azureDisk - account %s identified for a new disk  is because it has 0 allocated disks", v.name)
				return v.name, nil // shortcircut, avg is good and no need to adjust
			}
			// if this account is less allocated
			if dCount < maxDiskCount {
				maxDiskCount = dCount
				SAName = v.name
			}
		}
	}

	// if we failed to find storageaccount
	if SAName == "" {

		glog.Infof("azureDisk - failed to identify a suitable account for new disk and will attempt to create new account")
		SAName = getAccountNameForNum(c.getNextAccountNum())
		err := c.createStorageAccount(SAName, storageAccountType, true)
		if err != nil {
			return "", err
		}
		return SAName, nil
	}

	disksAfter := totalDiskCounts + 1 // with the new one!

	avgUtilization := float64(disksAfter) / float64(countAccounts*maxDisksPerStorageAccounts)
	aboveAvg := (avgUtilization > storageAccountUtilizationBeforeGrowing)

	// avg are not create and we should craete more accounts if we can
	if aboveAvg && countAccounts < maxStorageAccounts {
		glog.Infof("azureDisk - shared storageAccounts utilzation(%v) >  grow-at-avg-utilization (%v). New storage account will be created", avgUtilization, storageAccountUtilizationBeforeGrowing)
		SAName = getAccountNameForNum(c.getNextAccountNum())
		err := c.createStorageAccount(SAName, storageAccountType, true)
		if err != nil {
			return "", err
		}
		return SAName, nil
	}

	// avergates are not ok and we are at capacity(max storage accounts allowed)
	if aboveAvg && countAccounts == maxStorageAccounts {

		glog.Infof("azureDisk - shared storageAccounts utilzation(%v) > grow-at-avg-utilization (%v). But k8s maxed on SAs for PVC(%v). k8s will now exceed  grow-at-avg-utilization without adding accounts",
			avgUtilization, storageAccountUtilizationBeforeGrowing, maxStorageAccounts)
	}

	// we found a  storage accounts && [ avg are ok || we reached max sa count ]
	return SAName, nil
}
func (c *blobDiskController) getNextAccountNum() int {
	max := 0

	for k := range c.accounts {
		// filter out accounts that are for standalone
		if strings.Index(k, storageAccountNameMatch) != 0 {
			continue
		}
		num := getAccountNumFromName(k)
		if num > max {
			max = num
		}
	}

	return max + 1
}

func (c *blobDiskController) deleteStorageAccount(storageAccountName string) error {
	uri := fmt.Sprintf(storageAccountEndPointTemplate,
		c.common.managementEndpoint,
		c.common.subscriptionId,
		c.common.resourceGroup,
		storageAccountName)

	client := &http.Client{}
	r, err := http.NewRequest("DELETE", uri, nil)
	if err != nil {
		return err
	}

	token, err := c.common.getToken()

	if err != nil {
		return err
	}

	r.Header.Add("Authorization", "Bearer "+token)

	resp, err := client.Do(r)
	if err != nil || resp.StatusCode != 200 {
		newError := getRestError("DeleteStorageAccount", err, 200, resp.StatusCode, resp.Body)
		return newError
	}

	c.removeAccountState(storageAccountName)

	glog.Infof("azureDisk - Storage Account %s was deleted", storageAccountName)
	return nil
}

//Gets storage account exist, provisionStatus, Error if any
func (c *blobDiskController) getStorageAccount(storageAccountName string) (bool, string, error) {
	// should be get or create storage accounts and should return keys (from cache)
	uri := fmt.Sprintf(storageAccountEndPointTemplate,
		c.common.managementEndpoint,
		c.common.subscriptionId,
		c.common.resourceGroup,
		storageAccountName)

	client := &http.Client{}
	r, err := http.NewRequest("GET", uri, nil)
	if err != nil {
		return false, "", err
	}

	token, err := c.common.getToken()

	if err != nil {
		return false, "", err
	}

	r.Header.Add("Content-Type", "application/json")
	r.Header.Add("Authorization", "Bearer "+token)

	resp, err := client.Do(r)

	defer resp.Body.Close()

	if err != nil || resp.StatusCode != 200 {
		if resp.StatusCode == 404 {
			// we are ok the account does not exist
			return false, "", nil
		}

		glog.Infof("++ GetStorageAccount Status %v ", resp.StatusCode)
		glog.Infof("GetStorageAccount:%s failed with error %s", storageAccountName, err.Error())
		return false, "", getRestError(fmt.Sprintf("GetStorageAccount: %s", storageAccountName), err, 200, resp.StatusCode, resp.Body)
	}

	// extract data

	var payload interface{}
	bodyBytes, err := ioutil.ReadAll(resp.Body)

	if err := json.Unmarshal(bodyBytes, &payload); err != nil {
		return false, "", err
	}

	fragment := payload.(map[string]interface{})
	props := fragment["properties"].(map[string]interface{})
	provisionState := props["provisioningState"].(string)

	return true, provisionState, nil
}

func (c *blobDiskController) addAccountState(key string, state *storageAccountState) {
	accountsLock.Lock()
	defer func() {
		accountsLock.Unlock()
	}()

	if _, ok := c.accounts[key]; !ok {
		c.accounts[key] = state
	}
}

func (c *blobDiskController) removeAccountState(key string) {
	accountsLock.Lock()
	defer func() {
		accountsLock.Unlock()
	}()
	delete(c.accounts, key)
}

// pads account num with zeros as needed
func getAccountNameForNum(num int) string {
	sNum := strconv.Itoa(num)
	missingZeros := 3 - len(sNum)
	strZero := ""
	for missingZeros > 0 {
		strZero = strZero + "0"
		missingZeros = missingZeros - 1
	}

	sNum = strZero + sNum
	return fmt.Sprintf(storageAccountNamePrefix, sNum)
}

func getAccountNumFromName(accountName string) int {
	nameLen := len(accountName)
	num, _ := strconv.Atoi(accountName[nameLen-3:])

	return num
}

func createVHDHeader(size uint64) ([]byte, error) {
	h := vhd.CreateFixedHeader(size, &vhd.VHDOptions{})
	b := new(bytes.Buffer)
	err := binary.Write(b, binary.BigEndian, h)
	if err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}
