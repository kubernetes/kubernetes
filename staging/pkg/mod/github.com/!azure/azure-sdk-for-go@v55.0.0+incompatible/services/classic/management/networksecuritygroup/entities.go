// +build go1.7

// Package networksecuritygroup implements operations for managing network security groups
// using the Service Management REST API
//
// https://msdn.microsoft.com/en-us/library/azure/dn913824.aspx
package networksecuritygroup

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"encoding/xml"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

// SecurityGroupClient is used to perform operations on network security groups
type SecurityGroupClient struct {
	client management.Client
}

// SecurityGroupRequest represents a network security group
//
// https://msdn.microsoft.com/en-us/library/azure/dn913821.aspx
type SecurityGroupRequest struct {
	XMLName  xml.Name `xml:"http://schemas.microsoft.com/windowsazure NetworkSecurityGroup"`
	Name     string
	Label    string `xml:",omitempty"`
	Location string `xml:",omitempty"`
}

// SecurityGroupResponse represents a network security group
//
// https://msdn.microsoft.com/en-us/library/azure/dn913821.aspx
type SecurityGroupResponse struct {
	XMLName  xml.Name `xml:"http://schemas.microsoft.com/windowsazure NetworkSecurityGroup"`
	Name     string
	Label    string             `xml:",omitempty"`
	Location string             `xml:",omitempty"`
	State    SecurityGroupState `xml:",omitempty"`
	Rules    []RuleResponse     `xml:">Rule,omitempty"`
}

// SecurityGroupList represents a list of security groups
type SecurityGroupList []SecurityGroupResponse

// SecurityGroupState represents a security group state
type SecurityGroupState string

// These constants represent the possible security group states
const (
	SecurityGroupStateCreated     SecurityGroupState = "Created"
	SecurityGroupStateCreating    SecurityGroupState = "Creating"
	SecurityGroupStateUpdating    SecurityGroupState = "Updating"
	SecurityGroupStateDeleting    SecurityGroupState = "Deleting"
	SecurityGroupStateUnavailable SecurityGroupState = "Unavailable"
)

// RuleRequest represents a single rule of a network security group
//
// https://msdn.microsoft.com/en-us/library/azure/dn913821.aspx#bk_rules
type RuleRequest struct {
	XMLName                  xml.Name `xml:"http://schemas.microsoft.com/windowsazure Rule"`
	Name                     string
	Type                     RuleType
	Priority                 int
	Action                   RuleAction
	SourceAddressPrefix      string
	SourcePortRange          string
	DestinationAddressPrefix string
	DestinationPortRange     string
	Protocol                 RuleProtocol
}

// RuleResponse represents a single rule of a network security group
//
// https://msdn.microsoft.com/en-us/library/azure/dn913821.aspx#bk_rules
type RuleResponse struct {
	XMLName                  xml.Name `xml:"http://schemas.microsoft.com/windowsazure Rule"`
	Name                     string
	Type                     RuleType
	Priority                 int
	Action                   RuleAction
	SourceAddressPrefix      string
	SourcePortRange          string
	DestinationAddressPrefix string
	DestinationPortRange     string
	Protocol                 RuleProtocol
	State                    string `xml:",omitempty"`
	IsDefault                bool   `xml:",omitempty"`
}

// RuleType represents a rule type
type RuleType string

// These constants represent the possible rule types
const (
	RuleTypeInbound  RuleType = "Inbound"
	RuleTypeOutbound RuleType = "Outbound"
)

// RuleAction represents a rule action
type RuleAction string

// These constants represent the possible rule actions
const (
	RuleActionAllow RuleAction = "Allow"
	RuleActionDeny  RuleAction = "Deny"
)

// RuleProtocol represents a rule protocol
type RuleProtocol string

// These constants represent the possible rule types
const (
	RuleProtocolTCP RuleProtocol = "TCP"
	RuleProtocolUDP RuleProtocol = "UDP"
	RuleProtocolAll RuleProtocol = "*"
)
