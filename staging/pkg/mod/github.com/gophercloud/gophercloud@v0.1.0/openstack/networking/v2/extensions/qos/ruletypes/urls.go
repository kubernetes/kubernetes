package ruletypes

import "github.com/gophercloud/gophercloud"

func listRuleTypesURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL("qos", "rule-types")
}
