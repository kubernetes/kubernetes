package vxnet

import (
	"time"
)

type Vxnet struct {
	VxnetType   int       `json:"vxnet_type"`
	VxnetID     string    `json:"vxnet_id"`
	InstanceIds []string  `json:"instance_ids"`
	VxnetName   string    `json:"vxnet_name"`
	CreateTime  time.Time `json:"create_time"`
	Router      struct {
		RouterID   string `json:"router_id"`
		RouterName string `json:"router_name"`
		ManagerIP  string `json:"manager_ip"`
		IPNetwork  string `json:"ip_network"`
		DynIPEnd   string `json:"dyn_ip_end"`
		DynIPStart string `json:"dyn_ip_start"`
		Mode       int    `json:"mode"`
	} `json:"router"`
	Description interface{} `json:"description"`
}

type Instance struct {
	VcpusCurrent     int         `json:"vcpus_current"`
	InstanceID       string      `json:"instance_id"`
	ImageID          string      `json:"image_id"`
	VxnetID          string      `json:"vxnet_id"`
	Sequence         int         `json:"sequence"`
	SubCode          int         `json:"sub_code"`
	TransitionStatus string      `json:"transition_status"`
	InstanceName     string      `json:"instance_name"`
	InstanceType     string      `json:"instance_type"`
	CreateTime       time.Time   `json:"create_time"`
	Status           string      `json:"status"`
	PrivateIP        string      `json:"private_ip"`
	Description      interface{} `json:"description"`
	StatusTime       time.Time   `json:"status_time"`
	NicID            string      `json:"nic_id"`
	DhcpOptions      struct {
		Val2           string `json:"val2"`
		RouterStaticID string `json:"router_static_id"`
	} `json:"dhcp_options"`
	MemoryCurrent int `json:"memory_current"`
}
