// +build acceptance db rackspace

package v1

import (
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/rackspace/db/v1/instances"
	th "github.com/rackspace/gophercloud/testhelper"
)

func (c *context) createReplica() {
	opts := instances.CreateOpts{
		FlavorRef: "2",
		Size:      1,
		Name:      tools.RandomString("gopher_db", 5),
		ReplicaOf: c.instanceID,
	}

	repl, err := instances.Create(c.client, opts).Extract()
	th.AssertNoErr(c.test, err)

	c.Logf("Creating replica of %s. Waiting...", c.instanceID)
	c.WaitUntilActive(repl.ID)
	c.Logf("Created replica %#v", repl)

	c.replicaID = repl.ID
}

func (c *context) detachReplica() {
	err := instances.DetachReplica(c.client, c.replicaID).ExtractErr()
	c.Logf("Detached replica %s", c.replicaID)
	c.AssertNoErr(err)
}
