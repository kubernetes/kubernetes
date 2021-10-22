// +build acceptance quotasets

package v3

import (
	"os"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/extensions/quotasets"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestQuotasetGet(t *testing.T) {
	clients.RequireAdmin(t)

	client, projectID := getClientAndProject(t)

	quotaSet, err := quotasets.Get(client, projectID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, quotaSet)
}

func TestQuotasetGetDefaults(t *testing.T) {
	clients.RequireAdmin(t)

	client, projectID := getClientAndProject(t)

	quotaSet, err := quotasets.GetDefaults(client, projectID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, quotaSet)
}

func TestQuotasetGetUsage(t *testing.T) {
	clients.RequireAdmin(t)

	client, projectID := getClientAndProject(t)

	quotaSetUsage, err := quotasets.GetUsage(client, projectID).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, quotaSetUsage)
}

var UpdateQuotaOpts = quotasets.UpdateOpts{
	Volumes:            gophercloud.IntToPointer(100),
	Snapshots:          gophercloud.IntToPointer(200),
	Gigabytes:          gophercloud.IntToPointer(300),
	PerVolumeGigabytes: gophercloud.IntToPointer(50),
	Backups:            gophercloud.IntToPointer(2),
	BackupGigabytes:    gophercloud.IntToPointer(300),
}

var UpdatedQuotas = quotasets.QuotaSet{
	Volumes:            100,
	Snapshots:          200,
	Gigabytes:          300,
	PerVolumeGigabytes: 50,
	Backups:            2,
	BackupGigabytes:    300,
}

func TestQuotasetUpdate(t *testing.T) {
	clients.RequireAdmin(t)

	client, projectID := getClientAndProject(t)

	// save original quotas
	orig, err := quotasets.Get(client, projectID).Extract()
	th.AssertNoErr(t, err)

	defer func() {
		restore := quotasets.UpdateOpts{}
		FillUpdateOptsFromQuotaSet(*orig, &restore)

		_, err = quotasets.Update(client, projectID, restore).Extract()
		th.AssertNoErr(t, err)
	}()

	// test Update
	resultQuotas, err := quotasets.Update(client, projectID, UpdateQuotaOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, UpdatedQuotas, *resultQuotas)

	// We dont know the default quotas, so just check if the quotas are not the
	// same as before
	newQuotas, err := quotasets.Get(client, projectID).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, resultQuotas.Volumes, newQuotas.Volumes)
}

func TestQuotasetDelete(t *testing.T) {
	clients.RequireAdmin(t)

	client, projectID := getClientAndProject(t)

	// save original quotas
	orig, err := quotasets.Get(client, projectID).Extract()
	th.AssertNoErr(t, err)

	defer func() {
		restore := quotasets.UpdateOpts{}
		FillUpdateOptsFromQuotaSet(*orig, &restore)

		_, err = quotasets.Update(client, projectID, restore).Extract()
		th.AssertNoErr(t, err)
	}()

	// Obtain environment default quotaset values to validate deletion.
	defaultQuotaSet, err := quotasets.GetDefaults(client, projectID).Extract()
	th.AssertNoErr(t, err)

	// Test Delete
	err = quotasets.Delete(client, projectID).ExtractErr()
	th.AssertNoErr(t, err)

	newQuotas, err := quotasets.Get(client, projectID).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, newQuotas.Volumes, defaultQuotaSet.Volumes)
}

// getClientAndProject reduces boilerplate by returning a new blockstorage v3
// ServiceClient and a project ID obtained from the OS_PROJECT_NAME envvar.
func getClientAndProject(t *testing.T) (*gophercloud.ServiceClient, string) {
	client, err := clients.NewBlockStorageV3Client()
	th.AssertNoErr(t, err)

	projectID := os.Getenv("OS_PROJECT_NAME")
	th.AssertNoErr(t, err)
	return client, projectID
}

func FillUpdateOptsFromQuotaSet(src quotasets.QuotaSet, dest *quotasets.UpdateOpts) {
	dest.Volumes = &src.Volumes
	dest.Snapshots = &src.Snapshots
	dest.Gigabytes = &src.Gigabytes
	dest.PerVolumeGigabytes = &src.PerVolumeGigabytes
	dest.Backups = &src.Backups
	dest.BackupGigabytes = &src.BackupGigabytes
}
