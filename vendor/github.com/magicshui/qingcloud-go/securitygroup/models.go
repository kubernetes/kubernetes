package securitygroup

import (
	"time"
)

type SecurityGroup struct {
	IsApplied         int       `json:"is_applied"`
	Description       string    `json:"description"`
	SecurityGroupID   string    `json:"security_group_id"`
	IsDefault         int       `json:"is_default"`
	CreateTime        time.Time `json:"create_time"`
	SecurityGroupName string    `json:"security_group_name"`
	Resources         []struct {
		ResourceName string `json:"resource_name"`
		ResourceType string `json:"resource_type"`
		ResourceID   string `json:"resource_id"`
	} `json:"resources"`
}

type SecurityGroupRule struct {
	Protocol            string `json:"protocol"`
	SecurityGroupID     string `json:"security_group_id"`
	Priority            int    `json:"priority"`
	Action              string `json:"action"`
	SecurityGroupRuleID string `json:"security_group_rule_id"`
	Val2                string `json:"val2"`
	Val1                string `json:"val1"`
	Val3                string `json:"val3"`
	Direction           int    `json:"direction"`
}

type SecurityGroupSnapshot struct {
	Description string `json:"description"`
	Rules       []struct {
		Disabled              int    `json:"disabled"`
		Direction             int    `json:"direction"`
		Protocol              string `json:"protocol"`
		Priority              int    `json:"priority"`
		Val3                  string `json:"val3"`
		Action                string `json:"action"`
		Val2                  string `json:"val2"`
		Val1                  string `json:"val1"`
		SecurityGroupRuleName string `json:"security_group_rule_name"`
	} `json:"rules"`
	RootUserID              string    `json:"root_user_id"`
	CreateTime              time.Time `json:"create_time"`
	SecurityGroupSnapshotID string    `json:"security_group_snapshot_id"`
	Owner                   string    `json:"owner"`
	GroupID                 string    `json:"group_id"`
	Name                    string    `json:"name"`
}
