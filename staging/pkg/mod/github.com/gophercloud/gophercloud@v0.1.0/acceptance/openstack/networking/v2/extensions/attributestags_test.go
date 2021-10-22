// +build acceptance networking tags

package extensions

import (
	"fmt"
	"sort"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	networking "github.com/gophercloud/gophercloud/acceptance/openstack/networking/v2"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/attributestags"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/networks"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func createNetworkWithTags(t *testing.T, client *gophercloud.ServiceClient, tags []string) (network *networks.Network) {
	// Create Network
	network, err := networking.CreateNetwork(t, client)
	th.AssertNoErr(t, err)

	tagReplaceAllOpts := attributestags.ReplaceAllOpts{
		// docs say list of tags, but it's a set e.g no duplicates
		Tags: tags,
	}
	rtags, err := attributestags.ReplaceAll(client, "networks", network.ID, tagReplaceAllOpts).Extract()
	th.AssertNoErr(t, err)
	sort.Strings(rtags) // Ensure ordering, older OpenStack versions aren't sorted...
	th.AssertDeepEquals(t, rtags, tags)

	// Verify the tags are also set in the object Get response
	gnetwork, err := networks.Get(client, network.ID).Extract()
	th.AssertNoErr(t, err)
	rtags = gnetwork.Tags
	sort.Strings(rtags)
	th.AssertDeepEquals(t, rtags, tags)
	return network
}

func TestTags(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create Network
	network := createNetworkWithTags(t, client, []string{"a", "b", "c"})
	defer networking.DeleteNetwork(t, client, network.ID)

	// Add a tag
	err = attributestags.Add(client, "networks", network.ID, "d").ExtractErr()
	th.AssertNoErr(t, err)

	// Delete a tag
	err = attributestags.Delete(client, "networks", network.ID, "a").ExtractErr()
	th.AssertNoErr(t, err)

	// Verify expected tags are set in the List response
	tags, err := attributestags.List(client, "networks", network.ID).Extract()
	th.AssertNoErr(t, err)
	sort.Strings(tags)
	th.AssertDeepEquals(t, []string{"b", "c", "d"}, tags)

	// Confirm tags exist/don't exist
	exists, err := attributestags.Confirm(client, "networks", network.ID, "d").Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, true, exists)
	noexists, err := attributestags.Confirm(client, "networks", network.ID, "a").Extract()
	th.AssertEquals(t, false, noexists)

	// Delete all tags
	err = attributestags.DeleteAll(client, "networks", network.ID).ExtractErr()
	th.AssertNoErr(t, err)
	tags, err = attributestags.List(client, "networks", network.ID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, 0, len(tags))
}

func listNetworkWithTagOpts(t *testing.T, client *gophercloud.ServiceClient, listOpts networks.ListOpts) (ids []string) {
	allPages, err := networks.List(client, listOpts).AllPages()
	th.AssertNoErr(t, err)
	allNetworks, err := networks.ExtractNetworks(allPages)
	th.AssertNoErr(t, err)
	for _, network := range allNetworks {
		ids = append(ids, network.ID)
	}
	return ids
}

func TestQueryByTags(t *testing.T) {
	client, err := clients.NewNetworkV2Client()
	th.AssertNoErr(t, err)

	// Create a random tag to ensure we only get networks created
	// by this test
	testtag := tools.RandomString("zzz-tag-", 8)

	// Create Networks
	network1 := createNetworkWithTags(
		t, client, []string{"a", "b", "c", testtag})
	defer networking.DeleteNetwork(t, client, network1.ID)

	network2 := createNetworkWithTags(
		t, client, []string{"b", "c", "d", testtag})
	defer networking.DeleteNetwork(t, client, network2.ID)

	// Tags - Networks that match all tags will be returned
	listOpts := networks.ListOpts{
		Tags: fmt.Sprintf("a,b,c,%s", testtag)}
	ids := listNetworkWithTagOpts(t, client, listOpts)
	th.AssertDeepEquals(t, []string{network1.ID}, ids)

	// TagsAny - Networks that match any tag will be returned
	listOpts = networks.ListOpts{
		SortKey: "id", SortDir: "asc",
		TagsAny: fmt.Sprintf("a,b,c,%s", testtag)}
	ids = listNetworkWithTagOpts(t, client, listOpts)
	expected_ids := []string{network1.ID, network2.ID}
	sort.Strings(expected_ids)
	th.AssertDeepEquals(t, expected_ids, ids)

	// NotTags - Networks that match all tags will be excluded
	listOpts = networks.ListOpts{Tags: testtag, NotTags: "a,b,c"}
	ids = listNetworkWithTagOpts(t, client, listOpts)
	th.AssertDeepEquals(t, []string{network2.ID}, ids)

	// NotTagsAny - Networks that match any tag will be excluded.
	listOpts = networks.ListOpts{Tags: testtag, NotTagsAny: "d"}
	ids = listNetworkWithTagOpts(t, client, listOpts)
	th.AssertDeepEquals(t, []string{network1.ID}, ids)
}
