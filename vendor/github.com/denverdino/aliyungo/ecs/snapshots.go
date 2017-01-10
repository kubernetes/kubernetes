package ecs

import (
	"time"

	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/util"
)

type DescribeSnapshotsArgs struct {
	RegionId    common.Region
	InstanceId  string
	DiskId      string
	SnapshotIds []string //["s-xxxxxxxxx", "s-yyyyyyyyy", ..."s-zzzzzzzzz"]
	common.Pagination
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&snapshottype
type SnapshotType struct {
	SnapshotId     string
	SnapshotName   string
	Description    string
	Progress       string
	SourceDiskId   string
	SourceDiskSize int
	SourceDiskType string //enum for System | Data
	ProductCode    string
	CreationTime   util.ISO6801Time
}

type DescribeSnapshotsResponse struct {
	common.Response
	common.PaginationResult
	Snapshots struct {
		Snapshot []SnapshotType
	}
}

// DescribeSnapshots describe snapshots
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/snapshot&describesnapshots
func (client *Client) DescribeSnapshots(args *DescribeSnapshotsArgs) (snapshots []SnapshotType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeSnapshotsResponse{}

	err = client.Invoke("DescribeSnapshots", args, &response)

	if err != nil {
		return nil, nil, err
	}
	return response.Snapshots.Snapshot, &response.PaginationResult, nil

}

type DeleteSnapshotArgs struct {
	SnapshotId string
}

type DeleteSnapshotResponse struct {
	common.Response
}

// DeleteSnapshot deletes snapshot
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/snapshot&deletesnapshot
func (client *Client) DeleteSnapshot(snapshotId string) error {
	args := DeleteSnapshotArgs{SnapshotId: snapshotId}
	response := DeleteSnapshotResponse{}

	return client.Invoke("DeleteSnapshot", &args, &response)
}

type CreateSnapshotArgs struct {
	DiskId       string
	SnapshotName string
	Description  string
	ClientToken  string
}

type CreateSnapshotResponse struct {
	common.Response
	SnapshotId string
}

// CreateSnapshot creates a new snapshot
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/snapshot&createsnapshot
func (client *Client) CreateSnapshot(args *CreateSnapshotArgs) (snapshotId string, err error) {

	response := CreateSnapshotResponse{}

	err = client.Invoke("CreateSnapshot", args, &response)
	if err == nil {
		snapshotId = response.SnapshotId
	}
	return snapshotId, err
}

// Default timeout value for WaitForSnapShotReady method
const SnapshotDefaultTimeout = 120

// WaitForSnapShotReady waits for snapshot ready
func (client *Client) WaitForSnapShotReady(regionId common.Region, snapshotId string, timeout int) error {
	if timeout <= 0 {
		timeout = SnapshotDefaultTimeout
	}
	for {
		args := DescribeSnapshotsArgs{
			RegionId:    regionId,
			SnapshotIds: []string{snapshotId},
		}

		snapshots, _, err := client.DescribeSnapshots(&args)
		if err != nil {
			return err
		}
		if snapshots == nil || len(snapshots) == 0 {
			return common.GetClientErrorFromString("Not found")
		}
		if snapshots[0].Progress == "100%" {
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
