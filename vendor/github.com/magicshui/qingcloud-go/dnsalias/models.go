package dnsalias

import (
	"time"
)

type DnsAlias struct {
	CreateTime   time.Time `json:"create_time"`
	DNSAliasID   string    `json:"dns_alias_id"`
	DNSAliasName string    `json:"dns_alias_name"`
	ResourceID   string    `json:"resource_id"`
	Status       string    `json:"status"`
}
