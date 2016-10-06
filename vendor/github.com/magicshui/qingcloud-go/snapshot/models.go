package snapshot

import (
	"time"
)

type Snapshot struct {
	Status   string `json:"status"`
	Resource struct {
		ResourceName string `json:"resource_name"`
		ResourceType string `json:"resource_type"`
		ResourceID   string `json:"resource_id"`
	} `json:"resource"`
	SnapshotTime        time.Time `json:"snapshot_time"`
	IsHead              int       `json:"is_head"`
	RootID              string    `json:"root_id"`
	TotalSize           int       `json:"total_size"`
	TotalCount          int       `json:"total_count"`
	SubCode             int       `json:"sub_code"`
	SnapshotType        int       `json:"snapshot_type"`
	ParentID            string    `json:"parent_id"`
	SnapshotName        string    `json:"snapshot_name"`
	CreateTime          time.Time `json:"create_time"`
	HeadChain           int       `json:"head_chain"`
	SnapshotID          string    `json:"snapshot_id"`
	StatusTime          time.Time `json:"status_time"`
	Size                int       `json:"size"`
	LastestSnapshotTime time.Time `json:"lastest_snapshot_time"`
	Description         string    `json:"description"`
}
