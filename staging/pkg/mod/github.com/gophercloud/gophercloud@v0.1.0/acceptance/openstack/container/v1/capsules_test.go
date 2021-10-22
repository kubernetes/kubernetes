// +build acceptance containers capsules

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/container/v1/capsules"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCapsuleBase(t *testing.T) {
	client, err := clients.NewContainerV1Client()
	th.AssertNoErr(t, err)

	template := new(capsules.Template)
	template.Bin = []byte(capsuleTemplate)

	createOpts := capsules.CreateOpts{
		TemplateOpts: template,
	}

	v, err := capsules.Create(client, createOpts).Extract()
	th.AssertNoErr(t, err)
	capsule := v.(*capsules.Capsule)

	err = WaitForCapsuleStatus(client, capsule.UUID, "Running")
	th.AssertNoErr(t, err)

	pager := capsules.List(client, nil)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		v, err := capsules.ExtractCapsules(page)
		th.AssertNoErr(t, err)
		allCapsules := v.([]capsules.Capsule)

		for _, m := range allCapsules {
			capsuleUUID := m.UUID
			if capsuleUUID != capsule.UUID {
				continue
			}
			capsule, err := capsules.Get(client, capsuleUUID).ExtractBase()

			th.AssertNoErr(t, err)
			th.AssertEquals(t, capsule.MetaName, "template")

			err = capsules.Delete(client, capsuleUUID).ExtractErr()
			th.AssertNoErr(t, err)

		}
		return true, nil
	})
	th.AssertNoErr(t, err)
}

func TestCapsuleV132(t *testing.T) {
	client, err := clients.NewContainerV1Client()
	th.AssertNoErr(t, err)

	client.Microversion = "1.32"

	template := new(capsules.Template)
	template.Bin = []byte(capsuleTemplate)

	createOpts := capsules.CreateOpts{
		TemplateOpts: template,
	}

	capsule, err := capsules.Create(client, createOpts).ExtractV132()
	th.AssertNoErr(t, err)

	err = WaitForCapsuleStatus(client, capsule.UUID, "Running")
	th.AssertNoErr(t, err)

	pager := capsules.List(client, nil)
	err = pager.EachPage(func(page pagination.Page) (bool, error) {
		allCapsules, err := capsules.ExtractCapsulesV132(page)
		th.AssertNoErr(t, err)

		for _, m := range allCapsules {
			capsuleUUID := m.UUID
			if capsuleUUID != capsule.UUID {
				continue
			}
			capsule, err := capsules.Get(client, capsuleUUID).ExtractV132()

			th.AssertNoErr(t, err)
			th.AssertEquals(t, capsule.MetaName, "template")

			err = capsules.Delete(client, capsuleUUID).ExtractErr()
			th.AssertNoErr(t, err)

		}
		return true, nil
	})
	th.AssertNoErr(t, err)
}
