package instance

import (
	"time"
)

// Instance 主机
type Instance struct {
	Vxnets []struct {
		VxnetName string `json:"vxnet_name"`
		VxnetType int    `json:"vxnet_type"`
		VxnetID   string `json:"vxnet_id"`
		NicID     string `json:"nic_id"`
		PrivateIP string `json:"private_ip"`
	} `json:"vxnets"`
	MemoryCurrent int `json:"memory_current"`
	Eip           struct {
		EipAddr   string `json:"eip_addr"`
		Bandwidth int    `json:"bandwidth"`
		EipID     string `json:"eip_id"`
	} `json:"eip"`
	Extra struct {
		NoRestrict int         `json:"no_restrict"`
		NicMqueue  int         `json:"nic_mqueue"`
		NoLimit    int         `json:"no_limit"`
		CPUModel   string      `json:"cpu_model"`
		Slots      interface{} `json:"slots"`
		BootDev    string      `json:"boot_dev"`
		BlockBus   string      `json:"block_bus"`
	} `json:"extra"`
	Image struct {
		UIType        string `json:"ui_type"`
		ProcessorType string `json:"processor_type"`
		Platform      string `json:"platform"`
		ImageSize     int    `json:"image_size"`
		ImageName     string `json:"image_name"`
		ImageID       string `json:"image_id"`
		OsFamily      string `json:"os_family"`
		Provider      string `json:"provider"`
	} `json:"image"`
	CreateTime          time.Time     `json:"create_time"`
	AlarmStatus         string        `json:"alarm_status"`
	Owner               string        `json:"owner"`
	KeypairIds          []string      `json:"keypair_ids"`
	VcpusCurrent        int           `json:"vcpus_current"`
	InstanceID          string        `json:"instance_id"`
	SubCode             int           `json:"sub_code"`
	InstanceClass       int           `json:"instance_class"`
	StatusTime          time.Time     `json:"status_time"`
	Status              string        `json:"status"`
	Description         string        `json:"description"`
	CPUTopology         string        `json:"cpu_topology"`
	Tags                []interface{} `json:"tags"`
	TransitionStatus    string        `json:"transition_status"`
	VolumeIds           []interface{} `json:"volume_ids"`
	LastestSnapshotTime string        `json:"lastest_snapshot_time"`
	InstanceName        string        `json:"instance_name"`
	InstanceType        string        `json:"instance_type"`
	DNSAliases          []interface{} `json:"dns_aliases"`
	Volumes             []struct {
		Device   string `json:"device"`
		VolumeID string `json:"volume_id"`
	} `json:"volumes"`
	SecurityGroup struct {
		IsDefault       int    `json:"is_default"`
		SecurityGroupID string `json:"security_group_id"`
	} `json:"security_group"`
}
