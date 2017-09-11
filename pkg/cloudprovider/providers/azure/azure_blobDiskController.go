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

package azure

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"net/url"
	"os"
	"regexp"
	"sync"

	"strconv"
	"strings"
	"sync/atomic"
	"time"

	storage "github.com/Azure/azure-sdk-for-go/arm/storage"
	azstorage "github.com/Azure/azure-sdk-for-go/storage"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/glog"
	"github.com/rubiojr/go-vhd/vhd"
	kwait "k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	vhdContainerName         = "vhds"
	useHTTPSForBlobBasedDisk = true
	blobServiceName          = "blob"
)

type storageAccountState struct {
	name                    string
	saType                  storage.SkuName
	key                     string
	diskCount               int32
	isValidating            int32
	defaultContainerCreated bool
}

//BlobDiskController : blob disk controller struct
type BlobDiskController struct {
	common   *controllerCommon
	accounts map[string]*storageAccountState
}

var defaultContainerName = ""
var storageAccountNamePrefix = ""
var storageAccountNameMatch = ""
var initFlag int64

var accountsLock = &sync.Mutex{}

func newBlobDiskController(common *controllerCommon) (*BlobDiskController, error) {
	c := BlobDiskController{common: common}
	err := c.init()

	if err != nil {
		return nil, err
	}

	return &c, nil
}

// CreateVolume creates a VHD blob in a given storage account, will create the given storage account if it does not exist in current resource group
func (c *BlobDiskController) CreateVolume(name, storageAccount string, storageAccountType storage.SkuName, location string, requestGB int) (string, string, int, error) {
	key, err := c.common.cloud.getStorageAccesskey(storageAccount)
	if err != nil {
		glog.V(2).Infof("azureDisk - no key found for storage account %s in resource group %s, begin to create a new storage account", storageAccount, c.common.resourceGroup)

		cp := storage.AccountCreateParameters{
			Sku:      &storage.Sku{Name: storageAccountType},
			Tags:     &map[string]*string{"created-by": to.StringPtr("azure-dd")},
			Location: &location}
		cancel := make(chan struct{})

		_, errchan := c.common.cloud.StorageAccountClient.Create(c.common.resourceGroup, storageAccount, cp, cancel)
		err = <-errchan
		if err != nil {
			return "", "", 0, fmt.Errorf(fmt.Sprintf("Create Storage Account %s, error: %s", storageAccount, err))
		}

		key, err = c.common.cloud.getStorageAccesskey(storageAccount)
		if err != nil {
			return "", "", 0, fmt.Errorf("no key found for storage account %s even after creating a new storage account", storageAccount)
		}

		glog.Errorf("no key found for storage account %s in resource group %s", storageAccount, c.common.resourceGroup)
		return "", "", 0, err
	}

	client, err := azstorage.NewBasicClient(storageAccount, key)
	if err != nil {
		return "", "", 0, err
	}
	blobClient := client.GetBlobService()

	container := blobClient.GetContainerReference(vhdContainerName)
	_, err = container.CreateIfNotExists(&azstorage.CreateContainerOptions{Access: azstorage.ContainerAccessTypePrivate})
	if err != nil {
		return "", "", 0, err
	}

	diskName, diskURI, err := c.createVHDBlobDisk(blobClient, storageAccount, name, vhdContainerName, int64(requestGB))
	if err != nil {
		return "", "", 0, err
	}

	glog.V(4).Infof("azureDisk - created vhd blob uri: %s", diskURI)
	return diskName, diskURI, requestGB, err
}

// DeleteVolume deletes a VHD blob
func (c *BlobDiskController) DeleteVolume(diskURI string) error {
	glog.V(4).Infof("azureDisk - begin to delete volume %s", diskURI)
	accountName, blob, err := c.common.cloud.getBlobNameAndAccountFromURI(diskURI)
	if err != nil {
		return fmt.Errorf("failed to parse vhd URI %v", err)
	}
	key, err := c.common.cloud.getStorageAccesskey(accountName)
	if err != nil {
		return fmt.Errorf("no key for storage account %s, err %v", accountName, err)
	}
	err = c.common.cloud.deleteVhdBlob(accountName, key, blob)
	if err != nil {
		glog.Warningf("azureDisk - failed to delete blob %s err: %v", diskURI, err)
		detail := err.Error()
		if strings.Contains(detail, errLeaseIDMissing) {
			// disk is still being used
			// see https://msdn.microsoft.com/en-us/library/microsoft.windowsazure.storage.blob.protocol.bloberrorcodestrings.leaseidmissing.aspx
			return volume.NewDeletedVolumeInUseError(fmt.Sprintf("disk %q is still in use while being deleted", diskURI))
		}
		return fmt.Errorf("failed to delete vhd %v, account %s, blob %s, err: %v", diskURI, accountName, blob, err)
	}
	glog.V(4).Infof("azureDisk - blob %s deleted", diskURI)
	return nil

}

// get diskURI https://foo.blob.core.windows.net/vhds/bar.vhd and return foo (account) and bar.vhd (blob name)
func (c *BlobDiskController) getBlobNameAndAccountFromURI(diskURI string) (string, string, error) {
	scheme := "http"
	if useHTTPSForBlobBasedDisk {
		scheme = "https"
	}
	host := fmt.Sprintf("%s://(.*).%s.%s", scheme, blobServiceName, c.common.storageEndpointSuffix)
	reStr := fmt.Sprintf("%s/%s/(.*)", host, vhdContainerName)
	re := regexp.MustCompile(reStr)
	res := re.FindSubmatch([]byte(diskURI))
	if len(res) < 3 {
		return "", "", fmt.Errorf("invalid vhd URI for regex %s: %s", reStr, diskURI)
	}
	return string(res[1]), string(res[2]), nil
}

func (c *BlobDiskController) createVHDBlobDisk(blobClient azstorage.BlobStorageClient, accountName, vhdName, containerName string, sizeGB int64) (string, string, error) {
	container := blobClient.GetContainerReference(containerName)
	_, err := container.CreateIfNotExists(&azstorage.CreateContainerOptions{Access: azstorage.ContainerAccessTypePrivate})
	if err != nil {
		return "", "", err
	}

	size := 1024 * 1024 * 1024 * sizeGB
	vhdSize := size + vhd.VHD_HEADER_SIZE /* header size */
	// Blob name in URL must end with '.vhd' extension.
	vhdName = vhdName + ".vhd"

	tags := make(map[string]string)
	tags["createdby"] = "k8sAzureDataDisk"
	glog.V(4).Infof("azureDisk - creating page blob %name in container %s account %s", vhdName, containerName, accountName)

	blob := container.GetBlobReference(vhdName)
	blob.Properties.ContentLength = vhdSize
	blob.Metadata = tags
	err = blob.PutPageBlob(nil)
	if err != nil {
		return "", "", fmt.Errorf("failed to put page blob %s in container %s: %v", vhdName, containerName, err)
	}

	// add VHD signature to the blob
	h, err := createVHDHeader(uint64(size))
	if err != nil {
		blob.DeleteIfExists(nil)
		return "", "", fmt.Errorf("failed to create vhd header, err: %v", err)
	}

	blobRange := azstorage.BlobRange{
		Start: uint64(size),
		End:   uint64(vhdSize - 1),
	}
	if err = blob.WriteRange(blobRange, bytes.NewBuffer(h[:vhd.VHD_HEADER_SIZE]), nil); err != nil {
		glog.Infof("azureDisk - failed to put header page for data disk %s in container %s account %s, error was %s\n",
			vhdName, containerName, accountName, err.Error())
		return "", "", err
	}

	scheme := "http"
	if useHTTPSForBlobBasedDisk {
		scheme = "https"
	}

	host := fmt.Sprintf("%s://%s.%s.%s", scheme, accountName, blobServiceName, c.common.storageEndpointSuffix)
	uri := fmt.Sprintf("%s/%s/%s", host, containerName, vhdName)
	return vhdName, uri, nil
}

// delete a vhd blob
func (c *BlobDiskController) deleteVhdBlob(accountName, accountKey, blobName string) error {
	client, err := azstorage.NewBasicClient(accountName, accountKey)
	if err != nil {
		return err
	}
	blobSvc := client.GetBlobService()

	container := blobSvc.GetContainerReference(vhdContainerName)
	blob := container.GetBlobReference(blobName)
	return blob.Delete(nil)
}

//CreateBlobDisk : create a blob disk in a node
func (c *BlobDiskController) CreateBlobDisk(dataDiskName string, storageAccountType storage.SkuName, sizeGB int, forceStandAlone bool) (string, error) {
	glog.V(4).Infof("azureDisk - creating blob data disk named:%s on StorageAccountType:%s StandAlone:%v", dataDiskName, storageAccountType, forceStandAlone)

	var storageAccountName = ""
	var err error

	if forceStandAlone {
		// we have to wait until the storage account is is created
		storageAccountName = "p" + MakeCRC32(c.common.subscriptionID+c.common.resourceGroup+dataDiskName)
		err = c.createStorageAccount(storageAccountName, storageAccountType, c.common.location, false)
		if err != nil {
			return "", err
		}
	} else {
		storageAccountName, err = c.findSANameForDisk(storageAccountType)
		if err != nil {
			return "", err
		}
	}

	blobClient, err := c.getBlobSvcClient(storageAccountName)
	if err != nil {
		return "", err
	}

	_, diskURI, err := c.createVHDBlobDisk(blobClient, storageAccountName, dataDiskName, defaultContainerName, int64(sizeGB))
	if err != nil {
		return "", err
	}

	if !forceStandAlone {
		atomic.AddInt32(&c.accounts[storageAccountName].diskCount, 1)
	}

	return diskURI, nil
}

//DeleteBlobDisk : delete a blob disk from a node
func (c *BlobDiskController) DeleteBlobDisk(diskURI string, wasForced bool) error {
	storageAccountName, vhdName, err := diskNameandSANameFromURI(diskURI)
	if err != nil {
		return err
	}

	_, ok := c.accounts[storageAccountName]
	if !ok {
		// the storage account is specified by user
		glog.V(4).Infof("azureDisk - deleting volume %s", diskURI)
		return c.DeleteVolume(diskURI)
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

	glog.V(4).Infof("azureDisk - About to delete vhd file %s on storage account %s container %s", vhdName, storageAccountName, defaultContainerName)

	container := blobSvc.GetContainerReference(defaultContainerName)
	blob := container.GetBlobReference(vhdName)
	_, err = blob.DeleteIfExists(nil)

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

// Init tries best effort to ensure that 2 accounts standard/premium were created
// to be used by shared blob disks. This to increase the speed pvc provisioning (in most of cases)
func (c *BlobDiskController) init() error {
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

			accountType := storage.PremiumLRS
			if n := math.Mod(float64(counter), 2); n == 0 {
				accountType = storage.StandardLRS
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

				glog.Infof("azureDisk - BlobDiskController init process  will create new storageAccount:%s type:%s", newAccountName, accountType)
				err := c.createStorageAccount(newAccountName, accountType, c.common.location, true)
				// TODO return created and error from
				if err != nil {
					glog.Infof("azureDisk - BlobDiskController init: create account %s with error:%s", newAccountName, err.Error())

				} else {
					glog.Infof("azureDisk - BlobDiskController init: created account %s", newAccountName)
				}
			}(counter)
			counter = counter + 1
		}
	}

	return nil
}

//Sets unique strings to be used as accountnames && || blob containers names
func (c *BlobDiskController) setUniqueStrings() {
	uniqueString := c.common.resourceGroup + c.common.location + c.common.subscriptionID
	hash := MakeCRC32(uniqueString)
	//used to generate a unqie container name used by this cluster PVC
	defaultContainerName = hash

	storageAccountNamePrefix = fmt.Sprintf(storageAccountNameTemplate, hash)
	// Used to filter relevant accounts (accounts used by shared PVC)
	storageAccountNameMatch = storageAccountNamePrefix
	// Used as a template to create new names for relevant accounts
	storageAccountNamePrefix = storageAccountNamePrefix + "%s"
}
func (c *BlobDiskController) getStorageAccountKey(SAName string) (string, error) {
	if account, exists := c.accounts[SAName]; exists && account.key != "" {
		return c.accounts[SAName].key, nil
	}
	listKeysResult, err := c.common.cloud.StorageAccountClient.ListKeys(c.common.resourceGroup, SAName)
	if err != nil {
		return "", err
	}
	if listKeysResult.Keys == nil {
		return "", fmt.Errorf("azureDisk - empty listKeysResult in storage account:%s keys", SAName)
	}
	for _, v := range *listKeysResult.Keys {
		if v.Value != nil && *v.Value == "key1" {
			if _, ok := c.accounts[SAName]; !ok {
				glog.Warningf("azureDisk - account %s was not cached while getting keys", SAName)
				return *v.Value, nil
			}
		}

		c.accounts[SAName].key = *v.Value
		return c.accounts[SAName].key, nil
	}

	return "", fmt.Errorf("couldn't find key named key1 in storage account:%s keys", SAName)
}

func (c *BlobDiskController) getBlobSvcClient(SAName string) (azstorage.BlobStorageClient, error) {
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

func (c *BlobDiskController) ensureDefaultContainer(storageAccountName string) error {
	var err error
	var blobSvc azstorage.BlobStorageClient

	// short circut the check via local cache
	// we are forgiving the fact that account may not be in cache yet
	if v, ok := c.accounts[storageAccountName]; ok && v.defaultContainerCreated {
		return nil
	}

	// not cached, check existance and readiness
	bExist, provisionState, _ := c.getStorageAccountState(storageAccountName)

	// account does not exist
	if !bExist {
		return fmt.Errorf("azureDisk - account %s does not exist while trying to create/ensure default container", storageAccountName)
	}

	// account exists but not ready yet
	if provisionState != storage.Succeeded {
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
			_, provisionState, err := c.getStorageAccountState(storageAccountName)

			if err != nil {
				glog.V(4).Infof("azureDisk - GetStorageAccount:%s err %s", storageAccountName, err.Error())
				return false, err
			}

			if provisionState == storage.Succeeded {
				return true, nil
			}

			glog.V(4).Infof("azureDisk - GetStorageAccount:%s not ready yet", storageAccountName)
			// leave it for next loop/sync loop
			return false, fmt.Errorf("azureDisk - Account %s has not been flagged Succeeded by ARM", storageAccountName)
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

	container := blobSvc.GetContainerReference(defaultContainerName)
	bCreated, err := container.CreateIfNotExists(&azstorage.CreateContainerOptions{Access: azstorage.ContainerAccessTypePrivate})
	if err != nil {
		return err
	}
	if bCreated {
		glog.V(2).Infof("azureDisk - storage account:%s had no default container(%s) and it was created \n", storageAccountName, defaultContainerName)
	}

	// flag so we no longer have to check on ARM
	c.accounts[storageAccountName].defaultContainerCreated = true
	return nil
}

// Gets Disk counts per storage account
func (c *BlobDiskController) getDiskCount(SAName string) (int, error) {
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

	container := blobSvc.GetContainerReference(defaultContainerName)
	response, err := container.ListBlobs(params)
	if err != nil {
		return 0, err
	}
	glog.V(4).Infof("azure-Disk -  refreshed data count for account %s and found %v", SAName, len(response.Blobs))
	c.accounts[SAName].diskCount = int32(len(response.Blobs))

	return int(c.accounts[SAName].diskCount), nil
}

// shouldInit ensures that we only init the plugin once
// and we only do that in the controller

func (c *BlobDiskController) shouldInit() bool {
	if os.Args[0] == "kube-controller-manager" || (os.Args[0] == "/hyperkube" && os.Args[1] == "controller-manager") {
		swapped := atomic.CompareAndSwapInt64(&initFlag, 0, 1)
		if swapped {
			return true
		}
	}
	return false
}

func (c *BlobDiskController) getAllStorageAccounts() (map[string]*storageAccountState, error) {
	accountListResult, err := c.common.cloud.StorageAccountClient.List()
	if err != nil {
		return nil, err
	}
	if accountListResult.Value == nil {
		return nil, fmt.Errorf("azureDisk - empty accountListResult")
	}

	accounts := make(map[string]*storageAccountState)
	for _, v := range *accountListResult.Value {
		if strings.Index(*v.Name, storageAccountNameMatch) != 0 {
			continue
		}
		if v.Name == nil || v.Sku == nil {
			glog.Infof("azureDisk - accountListResult Name or Sku is nil")
			continue
		}
		glog.Infof("azureDisk - identified account %s as part of shared PVC accounts", *v.Name)

		sastate := &storageAccountState{
			name:      *v.Name,
			saType:    (*v.Sku).Name,
			diskCount: -1,
		}

		accounts[*v.Name] = sastate
	}

	return accounts, nil
}

func (c *BlobDiskController) createStorageAccount(storageAccountName string, storageAccountType storage.SkuName, location string, checkMaxAccounts bool) error {
	bExist, _, _ := c.getStorageAccountState(storageAccountName)
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

		glog.V(2).Infof("azureDisk - Creating storage account %s type %s \n", storageAccountName, string(storageAccountType))

		cp := storage.AccountCreateParameters{
			Sku:      &storage.Sku{Name: storageAccountType},
			Tags:     &map[string]*string{"created-by": to.StringPtr("azure-dd")},
			Location: &location}
		cancel := make(chan struct{})

		_, errChan := c.common.cloud.StorageAccountClient.Create(c.common.resourceGroup, storageAccountName, cp, cancel)
		err := <-errChan
		if err != nil {
			return fmt.Errorf(fmt.Sprintf("Create Storage Account: %s, error: %s", storageAccountName, err))
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
		glog.V(2).Infof("azureDisk - storage account %s was just created, allowing time before polling status")
		time.Sleep(25 * time.Second) // as observed 25 is the average time for SA to be provisioned
	}

	// finally, make sure that we default container is created
	// before handing it back over
	return c.ensureDefaultContainer(storageAccountName)
}

// finds a new suitable storageAccount for this disk
func (c *BlobDiskController) findSANameForDisk(storageAccountType storage.SkuName) (string, error) {
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
				glog.V(2).Infof("azureDisk - account %s identified for a new disk  is because it has 0 allocated disks", v.name)
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
		glog.V(2).Infof("azureDisk - failed to identify a suitable account for new disk and will attempt to create new account")
		SAName = getAccountNameForNum(c.getNextAccountNum())
		err := c.createStorageAccount(SAName, storageAccountType, c.common.location, true)
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
		glog.V(2).Infof("azureDisk - shared storageAccounts utilzation(%v) >  grow-at-avg-utilization (%v). New storage account will be created", avgUtilization, storageAccountUtilizationBeforeGrowing)
		SAName = getAccountNameForNum(c.getNextAccountNum())
		err := c.createStorageAccount(SAName, storageAccountType, c.common.location, true)
		if err != nil {
			return "", err
		}
		return SAName, nil
	}

	// avergates are not ok and we are at capacity(max storage accounts allowed)
	if aboveAvg && countAccounts == maxStorageAccounts {
		glog.Infof("azureDisk - shared storageAccounts utilzation(%v) > grow-at-avg-utilization (%v). But k8s maxed on SAs for PVC(%v). k8s will now exceed grow-at-avg-utilization without adding accounts",
			avgUtilization, storageAccountUtilizationBeforeGrowing, maxStorageAccounts)
	}

	// we found a  storage accounts && [ avg are ok || we reached max sa count ]
	return SAName, nil
}
func (c *BlobDiskController) getNextAccountNum() int {
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

func (c *BlobDiskController) deleteStorageAccount(storageAccountName string) error {
	resp, err := c.common.cloud.StorageAccountClient.Delete(c.common.resourceGroup, storageAccountName)
	if err != nil {
		return fmt.Errorf("azureDisk - Delete of storage account '%s' failed with status %s...%v", storageAccountName, resp.Status, err)
	}

	c.removeAccountState(storageAccountName)

	glog.Infof("azureDisk - Storage Account %s was deleted", storageAccountName)
	return nil
}

//Gets storage account exist, provisionStatus, Error if any
func (c *BlobDiskController) getStorageAccountState(storageAccountName string) (bool, storage.ProvisioningState, error) {
	account, err := c.common.cloud.StorageAccountClient.GetProperties(c.common.resourceGroup, storageAccountName)
	if err != nil {
		return false, "", err
	}
	return true, account.AccountProperties.ProvisioningState, nil
}

func (c *BlobDiskController) addAccountState(key string, state *storageAccountState) {
	accountsLock.Lock()
	defer accountsLock.Unlock()

	if _, ok := c.accounts[key]; !ok {
		c.accounts[key] = state
	}
}

func (c *BlobDiskController) removeAccountState(key string) {
	accountsLock.Lock()
	defer accountsLock.Unlock()
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

func diskNameandSANameFromURI(diskURI string) (string, string, error) {
	uri, err := url.Parse(diskURI)
	if err != nil {
		return "", "", err
	}

	hostName := uri.Host
	storageAccountName := strings.Split(hostName, ".")[0]

	segments := strings.Split(uri.Path, "/")
	diskNameVhd := segments[len(segments)-1]

	return storageAccountName, diskNameVhd, nil
}
