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

package nutanix_volume

import (
	"bytes"
	"crypto/sha256"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	ospath "path"
	"time"

	"github.com/golang/glog"
	gouuid "github.com/pborman/uuid"
)

const (
	restPath = "api/nutanix/v2.0"
	// Rest call response maximum wait time in seconds.
	responseWait = 120
)

type NutanixClient struct {
	prismEndPoint string
	secretValue   string
	apiPath       string
	client        *http.Client
}

func NewNutanixClient(prismEndPoint string, secretValue string) (*NutanixClient, error) {
	return &NutanixClient{
		prismEndPoint,
		secretValue,
		restPath,
		&http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			},
		},
	}, nil
}

func (c *NutanixClient) DoRequest(method, path string, query map[string][]string, data interface{}) ([]byte, error) {
	reqStream := bytes.NewBuffer(nil)
	if data != nil {
		glog.V(4).Infof("nutanix_volume: rest request method:%s path: %s, data: %+v", method, path, data)
		reqBodyBytes, err := json.Marshal(data)
		if err != nil {
			glog.Errorf("nutanix_volume: error: %v", err)
			return nil, err
		}
		reqStream.Write(reqBodyBytes)
	}

	u := url.URL{
		Scheme: "https",
		Host:   c.prismEndPoint,
		Path:   ospath.Join(c.apiPath, path),
	}

	if query != nil {
		values := u.Query()
		for key, val := range query {
			for _, v := range val {
				values.Add(key, v)
			}
		}
	}

	req, err := http.NewRequest(method, u.String(), reqStream)
	if err != nil {
		return nil, err
	}

	req.Header.Add("Accept", "application/json")
	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Authorization", "Basic "+c.secretValue)
	req.Close = true

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		return nil, fmt.Errorf("Error with request, got status code %d", resp.StatusCode)
	}

	defer resp.Body.Close()
	respStream := bytes.NewBuffer(nil)

	_, err = respStream.ReadFrom(resp.Body)
	if err != nil {
		return nil, err
	}

	return respStream.Bytes(), nil
}

type Metadata struct {
	GrandTotalEntities int    `json:"grand_total_entities,omitempty"`
	NextCursor         string `json:"next_cursor,omitempty"`
	SearchString       string `json:"search_string,omitempty"`
	StartIndex         int    `json:"start_index,omitempty"`
	PreviousCursor     string `json:"previous_cursor,omitempty"`
	FilterCriteria     string `json:"filter_criteria,omitempty"`
	EndIndex           int    `json:"end_index,omitempty"`
	Count              int    `json:"count,omitempty"`
	Page               int    `json:"page,omitempty"`
	SortCriteria       string `json:"sort_criteria,omitempty"`
	TotalEntities      int    `json:"total_entities,omitempty"`
}

type StorageContainerList struct {
	Metadata *Metadata              `json:"metadata,omitempty"`
	Entities []*StorageContainerDTO `json:"entities,omitempty"`
}

type StorageContainerDTO struct {
	ID                      string `json:"id,omitempty"`
	ReplicationFactor       int    `json:"replication_factor,omitempty"`
	ErasureCode             string `json:"erasure_code,omitempty"`
	ErasureCodeDelaySecs    int    `json:"erasure_code_delay_secs,omitempty"`
	CompressionEnabled      bool   `json:"compression_enabled,omitempty"`
	OplogReplicationFactor  int    `json:"oplog_replication_factor,omitempty"`
	CompressionDelayInSecs  int    `json:"compression_delay_in_secs,omitempty"`
	OnDiskDedup             string `json:"on_disk_dedup,omitempty"`
	AdvertisedCapacityBytes int64  `json:"advertised_capacity,omitempty"`
	FingerPrintOnWrite      string `json:"finger_print_on_write,omitempty"`
	StoragePoolID           string `json:"storage_pool_id,omitempty"`
	ClusterUUID             string `json:"cluster_uuid,omitempty"`
	StorageContainerUUID    string `json:"storage_container_uuid,omitempty"`
	MaxCapacityBytes        int64  `json:"max_capacity,omitempty"`
	Name                    string `json:"name,omitempty"`
}

func (c *NutanixClient) getStorageContainerUUID(name string) (string, error) {
	containerList, err := c.GetStorageContainerList()
	if err != nil {
		return "", fmt.Errorf("Failed to get storage container list from nutanix cluster, error: %s", err)
	}

	for _, container := range containerList.Entities {
		if container.Name == name {
			return container.StorageContainerUUID, nil
		}
	}
	return "", fmt.Errorf("Could not find storage container by name %s", name)
}

func (c *NutanixClient) GetStorageContainerList() (*StorageContainerList, error) {
	respBytes, err := c.DoRequest("GET", "/storage_containers/", nil, nil)
	if err != nil {
		return nil, err
	}
	containerList := &StorageContainerList{}
	err = json.Unmarshal(respBytes, containerList)
	if err != nil {
		return nil, err
	}
	return containerList, nil
}

type ReturnValueDTO struct {
	TaskUUID string `json:"taskUUID"`
}

type VGCreateDTO struct {
	Name             string       `json:"name,omitempty"`
	UUID             string       `json:"uuid,omitempty"`
	IsShared         bool         `json:"is_shared,omitempty"`
	IscsiTarget      string       `json:"iscsi_target,omitempty"`
	FlashModeEnabled bool         `json:"flash_mode_enabled,omitempty"`
	DiskList         []*VGDiskDTO `json:"disk_list,omitempty"`
}

type VGDiskDTO struct {
	CreateConfig VGDiskCreateDTO `json:"create_config,omitempty"`
}

type VGDiskCreateDTO struct {
	Size                 int64  `json:"size,omitempty"`
	StorageContainerUUID string `json:"storage_container_uuid,omitempty"`
}

type VGList struct {
	Metadata *Metadata    `json:"metadata,omitempty"`
	Entities []*VGInfoDTO `json:"entities,omitempty"`
}

type VGInfoDTO struct {
	LogicalTimestamp int              `json:"logical_timestamp,omitempty"`
	IscsiTarget      string           `json:"iscsi_target,omitempty"`
	UUID             string           `json:"uuid,omitempty"`
	Name             string           `json:"name,omitempty"`
	Description      string           `json:"description,omitempty"`
	IsShared         bool             `json:"is_shared,omitempty"`
	FlashModeEnabled bool             `json:"flash_mode_enabled,omitempty"`
	DiskList         []*VGDiskInfoDTO `json:"disk_list,omitempty"`
	AttachmentList   []*VGClientDTO   `json:"attachment_list,omitempty"`
}

type VGDiskInfoDTO struct {
	Index                int    `json:"index,omitempty"`
	SizeMb               int64  `json:"vmdisk_size_mb,omitempty"`
	FlashModeEnabled     bool   `json:"flash_mode_enabled,omitempty"`
	StorageContainerUUID string `json:"storage_container_uuid,omitempty"`
}

type VGClientDTO struct {
	ClientUUID string `json:"client_uuid,omitempty"`
	IQN        string `json:"iscsi_initiator_name,omitempty"`
}

type VGAttachDetachDTO struct {
	IscsiClientIP    string         `json:"iscsi_client_ip,omitempty"`
	Operation        string         `json:"operation,omitempty"`
	LogicalTimestamp int            `json:"logical_time_stamp,omitempty"`
	UUID             string         `json:"uuid,omitempty"`
	IscsiClient      IscsiClientDTO `json:"iscsi_client,omitempty"`
}

type IscsiClientDTO struct {
	ClientAddress string `json:"client_address,omitempty"`
}

func (c *NutanixClient) CreateVG(vgConfig *VGCreateDTO) (*VGInfoDTO, error) {
	var vgInfo *VGInfoDTO
	var err error

	if vgConfig.IscsiTarget == "" {
		vgConfig.IscsiTarget = fmt.Sprintf("%x:nutanix-k8-volume-plugin", sha256.Sum256([]byte(vgConfig.Name)))
	}

	// Check if VG is already created.
	vgInfo, err = c.getVG(vgConfig)
	if err != nil {
		return nil, err
	}
	if vgInfo != nil {
		return vgInfo, nil
	}

	// Create a new VG.
	vgConfig.UUID = gouuid.New()
	glog.V(4).Infof("nutanix_volume: Generated uuid %s", vgConfig.UUID)
	respBytes, err := c.DoRequest("POST", "/volume_groups/", nil, vgConfig)
	if err != nil {
		return nil, err
	}
	taskDO := &ReturnValueDTO{}
	err = json.Unmarshal(respBytes, taskDO)
	if err != nil {
		return nil, err
	}

	for i := 0; i < responseWait; i++ {
		vgInfo, err = c.GetVGInfo(vgConfig.UUID)
		if err == nil && len(vgInfo.DiskList) < len(vgConfig.DiskList) {
			<-time.After(1 * time.Second)
			continue
		}

		// We may get an error like status code 404 if cluster is busy. So
		// continue polling in error case also.
		if err != nil {
			glog.Errorf("nutanix_volume: ignoring error %v while checking for VG creation", err)
			<-time.After(1 * time.Second)
			continue
		}
		break
	}
	return vgInfo, err
}

func (c *NutanixClient) GetVGList() (*VGList, error) {
	respBytes, err := c.DoRequest("GET", "/volume_groups/", nil, nil)
	if err != nil {
		return nil, err
	}
	vgList := &VGList{}
	err = json.Unmarshal(respBytes, vgList)
	if err != nil {
		return nil, err
	}
	return vgList, nil
}

func (c *NutanixClient) GetVGInfo(vgUUID string) (*VGInfoDTO, error) {
	respBytes, err := c.DoRequest("GET", fmt.Sprintf("%s%s", "/volume_groups/", vgUUID), nil, nil)
	if err != nil {
		return nil, err
	}
	vgInfo := &VGInfoDTO{}
	err = json.Unmarshal(respBytes, vgInfo)
	if err != nil {
		return nil, err
	}
	return vgInfo, nil
}

func (c *NutanixClient) DeleteVG(vgUUID string) error {
	respBytes, err := c.DoRequest("DELETE", fmt.Sprintf("%s%s", "/volume_groups/", vgUUID), nil, nil)
	if err != nil {
		return err
	}
	taskDO := &ReturnValueDTO{}
	err = json.Unmarshal(respBytes, taskDO)
	if err != nil {
		return err
	}

	// Verify that VG has been deleted.
	for i := 0; i < responseWait; i++ {
		_, err = c.GetVGInfo(vgUUID)
		if err != nil {
			return nil
		}
	}
	return fmt.Errorf("Failed to delete volume %s", vgUUID)
}

func (c *NutanixClient) AttachVG(vgUUID string, data *VGAttachDetachDTO) error {
	iqnAttached, err := c.IsIqnAttached(data.IscsiClient.ClientAddress, vgUUID)
	if err != nil {
		return err
	}
	if iqnAttached {
		glog.V(4).Infof("nutanix_volume: iqn already attached %s", vgUUID)
		return nil
	}

	respBytes, err := c.DoRequest("POST", fmt.Sprintf("/volume_groups/%s/open", vgUUID), nil, data)
	if err != nil {
		return err
	}
	taskDO := &ReturnValueDTO{}
	err = json.Unmarshal(respBytes, taskDO)
	if err != nil {
		return err
	}

	for i := 0; i < responseWait; i++ {
		iqnAttached, err := c.IsIqnAttached(data.IscsiClient.ClientAddress, vgUUID)
		if err != nil {
			return err
		}
		if iqnAttached == false {
			<-time.After(1 * time.Second)
		} else {
			return nil
		}
	}
	return fmt.Errorf("nutanix_volume: failed to attach IQN to volume %s", vgUUID)
}

func (c *NutanixClient) DetachVG(vgUUID string, data *VGAttachDetachDTO) error {
	iqnAttached, err := c.IsIqnAttached(data.IscsiClient.ClientAddress, vgUUID)
	if err != nil {
		return err
	}
	if !iqnAttached {
		glog.V(4).Infof("nutanix_volume: iqn already detached %s", vgUUID)
		return nil
	}

	respBytes, err := c.DoRequest("POST", fmt.Sprintf("/volume_groups/%s/close", vgUUID), nil, data)
	if err != nil {
		return err
	}
	taskDO := &ReturnValueDTO{}
	err = json.Unmarshal(respBytes, taskDO)
	if err != nil {
		return err
	}

	for i := 0; i < responseWait; i++ {
		iqnAttached, err := c.IsIqnAttached(data.IscsiClient.ClientAddress, vgUUID)
		if err != nil {
			return err
		}
		if iqnAttached == true {
			<-time.After(1 * time.Second)
		} else {
			return nil
		}
	}
	return fmt.Errorf("Failed to detach IQN from volume %s", vgUUID)
}

func (c *NutanixClient) DetachAllVG(vgUUID string) error {
	var err error

	vgInfo, _ := c.GetVGInfo(vgUUID)
	if vgInfo == nil {
		glog.Errorf("nutanix_volume: Did not find any matching VG with uuid: %s", vgUUID)
		return nil
	}

	for _, attachment := range vgInfo.AttachmentList {
		iqnCfg := IscsiClientDTO{ClientAddress: attachment.IQN}
		data := &VGAttachDetachDTO{
			IscsiClient: iqnCfg,
		}
		err = c.DetachVG(vgUUID, data)
		if err != nil {
			return fmt.Errorf("nutanix_volume: error: %v", err)
		}
	}
	return err
}

func (c *NutanixClient) getVG(vgConfig *VGCreateDTO) (*VGInfoDTO, error) {
	vgList, err := c.GetVGList()
	if err != nil {
		return nil, err
	}

	for _, vgInfo := range vgList.Entities {
		if vgInfo.IscsiTarget == vgConfig.IscsiTarget {
			return vgInfo, nil
		}
	}
	return nil, nil
}

func (c *NutanixClient) getVolumeFromName(name string) (*VGInfoDTO, error) {
	vgList, err := c.GetVGList()
	if err != nil {
		return nil, err
	}

	for _, vgInfo := range vgList.Entities {
		if vgInfo.Name == name {
			return vgInfo, nil
		}
	}
	return nil, nil
}

func (c *NutanixClient) IsIqnAttached(iqn string, vgUUID string) (bool, error) {
	vgInfo, err := c.GetVGInfo(vgUUID)
	if err != nil {
		return false, err
	}
	if vgInfo == nil {
		return false, fmt.Errorf("nutanix_volume: Invalid volume uuid %s", vgUUID)
	}

	for _, attachment := range vgInfo.AttachmentList {
		if attachment.IQN == iqn {
			glog.V(4).Infof("nutanix_volume: Found matching attached IQN %s", iqn)
			return true, nil
		}
	}
	return false, nil
}
