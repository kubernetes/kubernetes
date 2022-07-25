package l7policies

import "github.com/gophercloud/gophercloud"

const (
	rootPath     = "lbaas"
	resourcePath = "l7policies"
	rulePath     = "rules"
)

func rootURL(c *gophercloud.ServiceClient) string {
	return c.ServiceURL(rootPath, resourcePath)
}

func resourceURL(c *gophercloud.ServiceClient, id string) string {
	return c.ServiceURL(rootPath, resourcePath, id)
}

func ruleRootURL(c *gophercloud.ServiceClient, policyID string) string {
	return c.ServiceURL(rootPath, resourcePath, policyID, rulePath)
}

func ruleResourceURL(c *gophercloud.ServiceClient, policyID string, ruleID string) string {
	return c.ServiceURL(rootPath, resourcePath, policyID, rulePath, ruleID)
}
