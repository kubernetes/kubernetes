package eip

import (
	"time"
)

type Eip struct {
	Status           string      `json:"status"`
	EipID            string      `json:"eip_id"`
	Description      interface{} `json:"description"`
	NeedIcp          int         `json:"need_icp"`
	SubCode          int         `json:"sub_code"`
	TransitionStatus string      `json:"transition_status"`
	IcpCodes         string      `json:"icp_codes"`
	EipGroup         struct {
		EipGroupID   string `json:"eip_group_id"`
		EipGroupName string `json:"eip_group_name"`
	} `json:"eip_group"`
	Bandwidth   int       `json:"bandwidth"`
	BillingMode string    `json:"billing_mode"`
	CreateTime  time.Time `json:"create_time"`
	StatusTime  time.Time `json:"status_time"`
	EipName     string    `json:"eip_name"`
	Resource    struct {
		ResourceName string `json:"resource_name"`
		ResourceType string `json:"resource_type"`
		ResourceID   string `json:"resource_id"`
	} `json:"resource"`
	EipAddr string `json:"eip_addr"`
}
