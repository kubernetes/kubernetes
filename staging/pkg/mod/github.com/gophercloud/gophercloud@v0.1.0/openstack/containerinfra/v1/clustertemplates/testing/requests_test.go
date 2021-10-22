package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/containerinfra/v1/clustertemplates"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreateClusterTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateClusterTemplateSuccessfully(t)

	boolFalse := false
	boolTrue := true
	dockerVolumeSize := 3
	opts := clustertemplates.CreateOpts{
		Name:                "kubernetes-dev",
		Labels:              map[string]string{},
		FixedSubnet:         "",
		MasterFlavorID:      "",
		NoProxy:             "10.0.0.0/8,172.0.0.0/8,192.0.0.0/8,localhost",
		HTTPSProxy:          "http://10.164.177.169:8080",
		TLSDisabled:         &boolFalse,
		KeyPairID:           "kp",
		Public:              &boolFalse,
		HTTPProxy:           "http://10.164.177.169:8080",
		DockerVolumeSize:    &dockerVolumeSize,
		ServerType:          "vm",
		ExternalNetworkID:   "public",
		ImageID:             "Fedora-Atomic-27-20180212.2.x86_64",
		VolumeDriver:        "cinder",
		RegistryEnabled:     &boolFalse,
		DockerStorageDriver: "devicemapper",
		NetworkDriver:       "flannel",
		FixedNetwork:        "",
		COE:                 "kubernetes",
		FlavorID:            "m1.small",
		MasterLBEnabled:     &boolTrue,
		DNSNameServer:       "8.8.8.8",
	}

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	res := clustertemplates.Create(sc, opts)
	th.AssertNoErr(t, res.Err)

	requestID := res.Header.Get("X-OpenStack-Request-Id")
	th.AssertEquals(t, "req-781e9bdc-4163-46eb-91c9-786c53188bbb", requestID)

	actual, err := res.Extract()
	th.AssertNoErr(t, err)

	actual.CreatedAt = actual.CreatedAt.UTC()
	th.AssertDeepEquals(t, ExpectedClusterTemplate, *actual)
}

func TestDeleteClusterTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleDeleteClusterSuccessfully(t)

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	res := clustertemplates.Delete(sc, "6dc6d336e3fc4c0a951b5698cd1236ee")
	th.AssertNoErr(t, res.Err)
	requestID := res.Header["X-Openstack-Request-Id"][0]
	th.AssertEquals(t, "req-781e9bdc-4163-46eb-91c9-786c53188bbb", requestID)
}

func TestListClusterTemplates(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListClusterTemplateSuccessfully(t)

	count := 0

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	clustertemplates.List(sc, clustertemplates.ListOpts{Limit: 2}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := clustertemplates.ExtractClusterTemplates(page)
		th.AssertNoErr(t, err)
		for idx, _ := range actual {
			actual[idx].CreatedAt = actual[idx].CreatedAt.UTC()
		}
		th.AssertDeepEquals(t, ExpectedClusterTemplates, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGetClusterTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetClusterTemplateSuccessfully(t)

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	actual, err := clustertemplates.Get(sc, "7d85f602-a948-4a30-afd4-e84f47471c15").Extract()
	th.AssertNoErr(t, err)
	actual.CreatedAt = actual.CreatedAt.UTC()
	th.AssertDeepEquals(t, ExpectedClusterTemplate, *actual)
}

func TestGetClusterTemplateEmptyTime(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetClusterTemplateEmptyTimeSuccessfully(t)

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	actual, err := clustertemplates.Get(sc, "7d85f602-a948-4a30-afd4-e84f47471c15").Extract()
	th.AssertNoErr(t, err)
	actual.CreatedAt = actual.CreatedAt.UTC()
	th.AssertDeepEquals(t, ExpectedClusterTemplate_EmptyTime, *actual)
}

func TestUpdateClusterTemplate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateClusterTemplateSuccessfully(t)

	updateOpts := []clustertemplates.UpdateOptsBuilder{
		clustertemplates.UpdateOpts{
			Path:  "/master_lb_enabled",
			Value: "True",
			Op:    clustertemplates.ReplaceOp,
		},
		clustertemplates.UpdateOpts{
			Path:  "/registry_enabled",
			Value: "True",
			Op:    clustertemplates.ReplaceOp,
		},
	}

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	res := clustertemplates.Update(sc, "7d85f602-a948-4a30-afd4-e84f47471c15", updateOpts)
	th.AssertNoErr(t, res.Err)

	actual, err := res.Extract()
	th.AssertNoErr(t, err)
	actual.CreatedAt = actual.CreatedAt.UTC()
	th.AssertDeepEquals(t, ExpectedUpdateClusterTemplate, *actual)
}

func TestUpdateClusterTemplateEmptyTime(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateClusterTemplateEmptyTimeSuccessfully(t)

	updateOpts := []clustertemplates.UpdateOptsBuilder{
		clustertemplates.UpdateOpts{
			Op:    clustertemplates.ReplaceOp,
			Path:  "/master_lb_enabled",
			Value: "True",
		},
		clustertemplates.UpdateOpts{
			Op:    clustertemplates.ReplaceOp,
			Path:  "/registry_enabled",
			Value: "True",
		},
	}

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	actual, err := clustertemplates.Update(sc, "7d85f602-a948-4a30-afd4-e84f47471c15", updateOpts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedUpdateClusterTemplate_EmptyTime, *actual)
}

func TestUpdateClusterTemplateInvalidUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateClusterTemplateInvalidUpdate(t)

	updateOpts := []clustertemplates.UpdateOptsBuilder{
		clustertemplates.UpdateOpts{
			Op:   clustertemplates.ReplaceOp,
			Path: "/master_lb_enabled",
		},
		clustertemplates.UpdateOpts{
			Op:   clustertemplates.RemoveOp,
			Path: "/master_lb_enabled",
		},
		clustertemplates.UpdateOpts{
			Op:   clustertemplates.AddOp,
			Path: "/master_lb_enabled",
		},
	}

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	_, err := clustertemplates.Update(sc, "7d85f602-a948-4a30-afd4-e84f47471c15", updateOpts).Extract()
	th.AssertEquals(t, true, err != nil)
}
