package members

import "github.com/gophercloud/gophercloud"

func imageMembersURL(c *gophercloud.ServiceClient, imageID string) string {
	return c.ServiceURL("images", imageID, "members")
}

func listMembersURL(c *gophercloud.ServiceClient, imageID string) string {
	return imageMembersURL(c, imageID)
}

func createMemberURL(c *gophercloud.ServiceClient, imageID string) string {
	return imageMembersURL(c, imageID)
}

func imageMemberURL(c *gophercloud.ServiceClient, imageID string, memberID string) string {
	return c.ServiceURL("images", imageID, "members", memberID)
}

func getMemberURL(c *gophercloud.ServiceClient, imageID string, memberID string) string {
	return imageMemberURL(c, imageID, memberID)
}

func updateMemberURL(c *gophercloud.ServiceClient, imageID string, memberID string) string {
	return imageMemberURL(c, imageID, memberID)
}

func deleteMemberURL(c *gophercloud.ServiceClient, imageID string, memberID string) string {
	return imageMemberURL(c, imageID, memberID)
}
