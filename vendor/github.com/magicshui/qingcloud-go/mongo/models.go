package mongo

import (
	"time"
)

type MongoNode struct {
	Status           string    `json:"status"`
	MongoID          string    `json:"mongo_id"`
	VxnetID          string    `json:"vxnet_id"`
	IP               string    `json:"ip"`
	TransitionStatus string    `json:"transition_status"`
	Controller       string    `json:"controller"`
	Primary          bool      `json:"primary"`
	ConsoleID        string    `json:"console_id"`
	InstanceID       string    `json:"instance_id"`
	MongoRole        string    `json:"mongo_role"`
	RootUserID       string    `json:"root_user_id"`
	CreateTime       time.Time `json:"create_time"`
	PgID             string    `json:"pg_id"`
	VolumeID         string    `json:"volume_id"`
	Owner            string    `json:"owner"`
	StatusTime       time.Time `json:"status_time"`
	MongoNodeID      string    `json:"mongo_node_id"`
	VMInstanceID     string    `json:"vm_instance_id"`
}

type Parameter struct {
	IsStatic        int    `json:"is_static"`
	ParameterValue  string `json:"parameter_value"`
	ParameterType   string `json:"parameter_type"`
	IsReadonly      int    `json:"is_readonly"`
	OptName         string `json:"opt_name"`
	ParameterName   string `json:"parameter_name"`
	ValueRange      string `json:"value_range"`
	ResourceVersion string `json:"resource_version"`
	ResourceType    string `json:"resource_type"`
}

type Mongo struct {
	Status              string    `json:"status"`
	MongoName           string    `json:"mongo_name"`
	Description         string    `json:"description"`
	AutoBackupTime      int       `json:"auto_backup_time"`
	RootUserID          string    `json:"root_user_id"`
	LatestSnapshotTime  string    `json:"latest_snapshot_time"`
	SubCode             int       `json:"sub_code"`
	TransitionStatus    string    `json:"transition_status"`
	StorageSize         int       `json:"storage_size"`
	ConsoleID           string    `json:"console_id"`
	MongoID             string    `json:"mongo_id"`
	MongoVersion        string    `json:"mongo_version"`
	Controller          string    `json:"controller"`
	CreateTime          time.Time `json:"create_time"`
	Owner               string    `json:"owner"`
	StatusTime          time.Time `json:"status_time"`
	MongoType           int       `json:"mongo_type"`
	AutoMinorVerUpgrade int       `json:"auto_minor_ver_upgrade"`
	Vxnet               struct {
		VxnetName string `json:"vxnet_name"`
		VxnetID   string `json:"vxnet_id"`
	} `json:"vxnet"`
}
