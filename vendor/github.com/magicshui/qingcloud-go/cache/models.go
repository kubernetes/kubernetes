package cache

import (
	"time"
)

type Cache struct {
	Status                string    `json:"status"`
	CachePort             int       `json:"cache_port"`
	IsApplied             int       `json:"is_applied"`
	Description           string    `json:"description"`
	AutoBackupTime        int       `json:"auto_backup_time"`
	CacheType             string    `json:"cache_type"`
	LastestSnapshotTime   time.Time `json:"lastest_snapshot_time"`
	CacheParameterGroupID string    `json:"cache_parameter_group_id"`
	SubCode               int       `json:"sub_code"`
	TransitionStatus      string    `json:"transition_status"`
	MaxMemory             int       `json:"max_memory"`
	SecurityGroupID       string    `json:"security_group_id"`
	CacheSize             int       `json:"cache_size"`
	CreateTime            time.Time `json:"create_time"`
	CacheID               string    `json:"cache_id"`
	StatusTime            time.Time `json:"status_time"`
	Nodes                 []struct {
		Status           string    `json:"status"`
		CacheType        string    `json:"cache_type"`
		TransitionStatus string    `json:"transition_status"`
		CacheID          string    `json:"cache_id"`
		CacheNodeID      string    `json:"cache_node_id"`
		CacheRole        string    `json:"cache_role"`
		CreateTime       time.Time `json:"create_time"`
		CacheNodeName    string    `json:"cache_node_name"`
		StatusTime       time.Time `json:"status_time"`
		AlarmStatus      string    `json:"alarm_status"`
		PrivateIP        string    `json:"private_ip"`
	} `json:"nodes"`
	CacheName string `json:"cache_name"`
	NodeCount int    `json:"node_count"`
	Vxnet     struct {
		VxnetName string `json:"vxnet_name"`
		VxnetID   string `json:"vxnet_id"`
	} `json:"vxnet"`
}

type CacheNode struct {
	Status           string    `json:"status"`
	CacheType        string    `json:"cache_type"`
	TransitionStatus string    `json:"transition_status"`
	CacheID          string    `json:"cache_id"`
	CacheNodeID      string    `json:"cache_node_id"`
	CacheRole        string    `json:"cache_role"`
	CreateTime       time.Time `json:"create_time"`
	CacheNodeName    string    `json:"cache_node_name"`
	StatusTime       time.Time `json:"status_time"`
	AlarmStatus      string    `json:"alarm_status"`
	PrivateIP        string    `json:"private_ip"`
}

type CacheParameterGroup struct {
	IsApplied               int       `json:"is_applied"`
	Description             string    `json:"description"`
	CacheType               string    `json:"cache_type"`
	CacheParameterGroupName string    `json:"cache_parameter_group_name"`
	IsDefault               int       `json:"is_default"`
	CreateTime              time.Time `json:"create_time"`
	CacheParameterGroupID   string    `json:"cache_parameter_group_id"`
}
