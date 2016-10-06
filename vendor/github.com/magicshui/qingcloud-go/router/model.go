package router

import (
	"time"
)

type Router struct {
	RouterID    string `json:"router_id"`
	Status      string `json:"status"`
	IsApplied   int    `json:"is_applied"`
	Description string `json:"description"`
	Eip         struct {
		EipName string `json:"eip_name"`
		EipID   string `json:"eip_id"`
		EipAddr string `json:"eip_addr"`
	} `json:"eip"`
	SubCode          int       `json:"sub_code"`
	TransitionStatus string    `json:"transition_status"`
	SecurityGroupID  string    `json:"security_group_id"`
	CreateTime       time.Time `json:"create_time"`
	PrivateIP        string    `json:"private_ip"`
	RouterType       int       `json:"router_type"`
	Vxnets           []struct {
		NicID   string `json:"nic_id"`
		VxnetID string `json:"vxnet_id"`
	} `json:"vxnets"`
	RouterName string `json:"router_name"`
}

type RouterStatic struct {
	RouterID         string    `json:"router_id"`
	VxnetID          string    `json:"vxnet_id"`
	RouterStaticID   string    `json:"router_static_id"`
	StaticType       int       `json:"static_type"`
	RouterStaticName string    `json:"router_static_name"`
	Disabled         int       `json:"disabled"`
	Owner            string    `json:"owner"`
	CreateTime       time.Time `json:"create_time"`
	Val3             string    `json:"val3"`
	Val2             string    `json:"val2"`
	Val1             string    `json:"val1"`
	Val6             string    `json:"val6"`
	Val5             string    `json:"val5"`
	Val4             string    `json:"val4"`
}

type RouterVxnet struct {
	RouterID   string    `json:"router_id"`
	ManagerIP  string    `json:"manager_ip"`
	IPNetwork  string    `json:"ip_network"`
	DynIPEnd   string    `json:"dyn_ip_end"`
	VxnetID    string    `json:"vxnet_id"`
	DynIPStart string    `json:"dyn_ip_start"`
	VxnetName  string    `json:"vxnet_name"`
	CreateTime time.Time `json:"create_time"`
	Features   int       `json:"features"`
}

type RouterStaticEntry struct {
	RouterID              string `json:"router_id"`
	RouterStaticEntryID   string `json:"router_static_entry_id"`
	RouterStaticID        string `json:"router_static_id"`
	RouterStaticEntryName string `json:"router_static_entry_name"`
	Val2                  string `json:"val2"`
	Val1                  string `json:"val1"`
}
