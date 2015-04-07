// +build acceptance rackspace objectstorage v1

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud/rackspace/objectstorage/v1/bulk"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestBulk(t *testing.T) {
	c, err := createClient(t, false)
	th.AssertNoErr(t, err)

	var options bulk.DeleteOpts
	options = append(options, "container/object1")
	res := bulk.Delete(c, options)
	th.AssertNoErr(t, res.Err)
	body, err := res.ExtractBody()
	th.AssertNoErr(t, err)
	t.Logf("Response body from Bulk Delete Request: %+v\n", body)
}
