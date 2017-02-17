package ecs

import (
	"time"

	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/util"
)

// Types of disks
type DiskType string

const (
	DiskTypeAll       = DiskType("all") //Default
	DiskTypeAllSystem = DiskType("system")
	DiskTypeAllData   = DiskType("data")
)

// Categories of disks
type DiskCategory string

const (
	DiskCategoryAll             = DiskCategory("all") //Default
	DiskCategoryCloud           = DiskCategory("cloud")
	DiskCategoryEphemeral       = DiskCategory("ephemeral")
	DiskCategoryEphemeralSSD    = DiskCategory("ephemeral_ssd")
	DiskCategoryCloudEfficiency = DiskCategory("cloud_efficiency")
	DiskCategoryCloudSSD        = DiskCategory("cloud_ssd")
)

// Status of disks
type DiskStatus string

const (
	DiskStatusInUse     = DiskStatus("In_use")
	DiskStatusAvailable = DiskStatus("Available")
	DiskStatusAttaching = DiskStatus("Attaching")
	DiskStatusDetaching = DiskStatus("Detaching")
	DiskStatusCreating  = DiskStatus("Creating")
	DiskStatusReIniting = DiskStatus("ReIniting")
	DiskStatusAll       = DiskStatus("All") //Default
)

// Charge type of disks
type DiskChargeType string

const (
	PrePaid  = DiskChargeType("PrePaid")
	PostPaid = DiskChargeType("PostPaid")
)

// A DescribeDisksArgs defines the arguments to describe disks
type DescribeDisksArgs struct {
	RegionId           common.Region
	ZoneId             string
	DiskIds            []string
	InstanceId         string
	DiskType           DiskType     //enum for all(default) | system | data
	Category           DiskCategory //enum for all(default) | cloud | ephemeral
	Status             DiskStatus   //enum for In_use | Available | Attaching | Detaching | Creating | ReIniting | All(default)
	SnapshotId         string
	Name               string
	Portable           *bool //optional
	DeleteWithInstance *bool //optional
	DeleteAutoSnapshot *bool //optional
	EnableAutoSnapshot *bool //optional
	DiskChargeType     DiskChargeType
	Tag                map[string]string
	common.Pagination
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&diskitemtype
type DiskItemType struct {
	DiskId             string
	RegionId           common.Region
	ZoneId             string
	DiskName           string
	Description        string
	Type               DiskType
	Category           DiskCategory
	Size               int
	ImageId            string
	SourceSnapshotId   string
	ProductCode        string
	Portable           bool
	Status             DiskStatus
	OperationLocks     OperationLocksType
	InstanceId         string
	Device             string
	DeleteWithInstance bool
	DeleteAutoSnapshot bool
	EnableAutoSnapshot bool
	CreationTime       util.ISO6801Time
	AttachedTime       util.ISO6801Time
	DetachedTime       util.ISO6801Time
	DiskChargeType     DiskChargeType
}

type DescribeDisksResponse struct {
	common.Response
	common.PaginationResult
	RegionId common.Region
	Disks    struct {
		Disk []DiskItemType
	}
}

// DescribeDisks describes Disks
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/disk&describedisks
func (client *Client) DescribeDisks(args *DescribeDisksArgs) (disks []DiskItemType, pagination *common.PaginationResult, err error) {
	response := DescribeDisksResponse{}

	err = client.Invoke("DescribeDisks", args, &response)

	if err != nil {
		return nil, nil, err
	}

	return response.Disks.Disk, &response.PaginationResult, err
}

type CreateDiskArgs struct {
	RegionId     common.Region
	ZoneId       string
	DiskName     string
	Description  string
	DiskCategory DiskCategory
	Size         int
	SnapshotId   string
	ClientToken  string
}

type CreateDisksResponse struct {
	common.Response
	DiskId string
}

// CreateDisk creates a new disk
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/disk&createdisk
func (client *Client) CreateDisk(args *CreateDiskArgs) (diskId string, err error) {
	response := CreateDisksResponse{}
	err = client.Invoke("CreateDisk", args, &response)
	if err != nil {
		return "", err
	}
	return response.DiskId, err
}

type DeleteDiskArgs struct {
	DiskId string
}

type DeleteDiskResponse struct {
	common.Response
}

// DeleteDisk deletes disk
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/disk&deletedisk
func (client *Client) DeleteDisk(diskId string) error {
	args := DeleteDiskArgs{
		DiskId: diskId,
	}
	response := DeleteDiskResponse{}
	err := client.Invoke("DeleteDisk", &args, &response)
	return err
}

type ReInitDiskArgs struct {
	DiskId string
}

type ReInitDiskResponse struct {
	common.Response
}

// ReInitDisk reinitizes disk
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/disk&reinitdisk
func (client *Client) ReInitDisk(diskId string) error {
	args := ReInitDiskArgs{
		DiskId: diskId,
	}
	response := ReInitDiskResponse{}
	err := client.Invoke("ReInitDisk", &args, &response)
	return err
}

type AttachDiskArgs struct {
	InstanceId         string
	DiskId             string
	Device             string
	DeleteWithInstance bool
}

type AttachDiskResponse struct {
	common.Response
}

// AttachDisk attaches disk to instance
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/disk&attachdisk
func (client *Client) AttachDisk(args *AttachDiskArgs) error {
	response := AttachDiskResponse{}
	err := client.Invoke("AttachDisk", args, &response)
	return err
}

type DetachDiskArgs struct {
	InstanceId string
	DiskId     string
}

type DetachDiskResponse struct {
	common.Response
}

// DetachDisk detaches disk from instance
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/disk&detachdisk
func (client *Client) DetachDisk(instanceId string, diskId string) error {
	args := DetachDiskArgs{
		InstanceId: instanceId,
		DiskId:     diskId,
	}
	response := DetachDiskResponse{}
	err := client.Invoke("DetachDisk", &args, &response)
	return err
}

type ResetDiskArgs struct {
	DiskId     string
	SnapshotId string
}

type ResetDiskResponse struct {
	common.Response
}

// ResetDisk resets disk to original status
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/disk&resetdisk
func (client *Client) ResetDisk(diskId string, snapshotId string) error {
	args := ResetDiskArgs{
		SnapshotId: snapshotId,
		DiskId:     diskId,
	}
	response := ResetDiskResponse{}
	err := client.Invoke("ResetDisk", &args, &response)
	return err
}

type ModifyDiskAttributeArgs struct {
	DiskId             string
	DiskName           string
	Description        string
	DeleteWithInstance *bool
	DeleteAutoSnapshot *bool
	EnableAutoSnapshot *bool
}

type ModifyDiskAttributeResponse struct {
	common.Response
}

// ModifyDiskAttribute modifies disk attribute
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/disk&modifydiskattribute
func (client *Client) ModifyDiskAttribute(args *ModifyDiskAttributeArgs) error {
	response := ModifyDiskAttributeResponse{}
	err := client.Invoke("ModifyDiskAttribute", args, &response)
	return err
}

type ReplaceSystemDiskArgs struct {
	InstanceId  string
	ImageId     string
	SystemDisk  SystemDiskType
	ClientToken string
}

type ReplaceSystemDiskResponse struct {
	common.Response
	DiskId string
}

// ReplaceSystemDisk replace system disk
//
// You can read doc at https://help.aliyun.com/document_detail/ecs/open-api/disk/replacesystemdisk.html
func (client *Client) ReplaceSystemDisk(args *ReplaceSystemDiskArgs) (diskId string, err error) {
	response := ReplaceSystemDiskResponse{}
	err = client.Invoke("ReplaceSystemDisk", args, &response)
	if err != nil {
		return "", err
	}
	return response.DiskId, nil
}

// WaitForDisk waits for disk to given status
func (client *Client) WaitForDisk(regionId common.Region, diskId string, status DiskStatus, timeout int) error {
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	args := DescribeDisksArgs{
		RegionId: regionId,
		DiskIds:  []string{diskId},
	}

	for {
		disks, _, err := client.DescribeDisks(&args)
		if err != nil {
			return err
		}
		if disks == nil || len(disks) == 0 {
			return common.GetClientErrorFromString("Not found")
		}
		if disks[0].Status == status {
			break
		}
		timeout = timeout - DefaultWaitForInterval
		if timeout <= 0 {
			return common.GetClientErrorFromString("Timeout")
		}
		time.Sleep(DefaultWaitForInterval * time.Second)
	}
	return nil
}
