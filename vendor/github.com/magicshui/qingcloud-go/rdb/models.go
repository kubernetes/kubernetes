package rdb

import (
	"time"
)

type Rdb struct {
	RdbID          string    `json:"rdb_id"`
	AutoBackupTime int       `json:"auto_backup_time"`
	ConsoleID      string    `json:"console_id"`
	CreateTime     time.Time `json:"create_time"`
	AlarmStatus    string    `json:"alarm_status"`
	Owner          string    `json:"owner"`
	RdbName        string    `json:"rdb_name"`
	SubCode        int       `json:"sub_code"`
	MasterIP       string    `json:"master_ip"`
	StatusTime     time.Time `json:"status_time"`
	Vxnet          struct {
		VxnetName string `json:"vxnet_name"`
		VxnetID   string `json:"vxnet_id"`
	} `json:"vxnet"`
	Status              string    `json:"status"`
	Description         string    `json:"description"`
	TransitionStatus    string    `json:"transition_status"`
	StorageSize         int       `json:"storage_size"`
	Controller          string    `json:"controller"`
	RdbType             int       `json:"rdb_type"`
	AutoMinorVerUpgrade int       `json:"auto_minor_ver_upgrade"`
	LastestSnapshotTime time.Time `json:"lastest_snapshot_time"`
	EngineVersion       string    `json:"engine_version"`
	RootUserID          string    `json:"root_user_id"`
	RdbEngine           string    `json:"rdb_engine"`
}

type File struct {
	SlowLog []struct {
		LastModify string `json:"last_modify"`
		File       string `json:"file"`
		Size       int    `json:"size"`
	} `json:"slow_log"`
	BinaryLog []struct {
		LastModify string `json:"last_modify"`
		File       string `json:"file"`
		Size       int    `json:"size"`
	} `json:"binary_log"`
}
