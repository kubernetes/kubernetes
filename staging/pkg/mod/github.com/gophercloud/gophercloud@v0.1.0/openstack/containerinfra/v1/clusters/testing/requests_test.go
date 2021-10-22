package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/containerinfra/v1/clusters"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreateCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateClusterSuccessfully(t)

	masterCount := 1
	nodeCount := 1
	createTimeout := 30
	opts := clusters.CreateOpts{
		ClusterTemplateID: "0562d357-8641-4759-8fed-8173f02c9633",
		CreateTimeout:     &createTimeout,
		DiscoveryURL:      "",
		FlavorID:          "m1.small",
		Keypair:           "my_keypair",
		Labels:            map[string]string{},
		MasterCount:       &masterCount,
		MasterFlavorID:    "m1.small",
		Name:              "k8s",
		NodeCount:         &nodeCount,
	}

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	res := clusters.Create(sc, opts)
	th.AssertNoErr(t, res.Err)

	requestID := res.Header.Get("X-OpenStack-Request-Id")
	th.AssertEquals(t, requestUUID, requestID)

	actual, err := res.Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, clusterUUID, actual)
}

func TestGetCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetClusterSuccessfully(t)

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	actual, err := clusters.Get(sc, "746e779a-751a-456b-a3e9-c883d734946f").Extract()
	th.AssertNoErr(t, err)
	actual.CreatedAt = actual.CreatedAt.UTC()
	actual.UpdatedAt = actual.UpdatedAt.UTC()
	th.AssertDeepEquals(t, ExpectedCluster, *actual)
}

func TestListClusters(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListClusterSuccessfully(t)

	count := 0
	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	clusters.List(sc, clusters.ListOpts{Limit: 2}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := clusters.ExtractClusters(page)
		th.AssertNoErr(t, err)
		for idx := range actual {
			actual[idx].CreatedAt = actual[idx].CreatedAt.UTC()
			actual[idx].UpdatedAt = actual[idx].UpdatedAt.UTC()
		}
		th.AssertDeepEquals(t, ExpectedClusters, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestListDetailClusters(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleListDetailClusterSuccessfully(t)

	count := 0
	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	clusters.ListDetail(sc, clusters.ListOpts{Limit: 2}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := clusters.ExtractClusters(page)
		th.AssertNoErr(t, err)
		for idx := range actual {
			actual[idx].CreatedAt = actual[idx].CreatedAt.UTC()
			actual[idx].UpdatedAt = actual[idx].UpdatedAt.UTC()
		}
		th.AssertDeepEquals(t, ExpectedClusters, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestUpdateCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateClusterSuccessfully(t)

	updateOpts := []clusters.UpdateOptsBuilder{
		clusters.UpdateOpts{
			Op:    clusters.ReplaceOp,
			Path:  "/master_lb_enabled",
			Value: "True",
		},
		clusters.UpdateOpts{
			Op:    clusters.ReplaceOp,
			Path:  "/registry_enabled",
			Value: "True",
		},
	}

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	res := clusters.Update(sc, clusterUUID, updateOpts)
	th.AssertNoErr(t, res.Err)

	requestID := res.Header.Get("X-OpenStack-Request-Id")
	th.AssertEquals(t, requestUUID, requestID)

	actual, err := res.Extract()
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, clusterUUID, actual)
}

func TestDeleteCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleDeleteClusterSuccessfully(t)

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	r := clusters.Delete(sc, clusterUUID)
	err := r.ExtractErr()
	th.AssertNoErr(t, err)

	uuid := ""
	idKey := "X-Openstack-Request-Id"
	if len(r.Header[idKey]) > 0 {
		uuid = r.Header[idKey][0]
		if uuid == "" {
			t.Errorf("No value for header [%s]", idKey)
		}
	} else {
		t.Errorf("Missing header [%s]", idKey)
	}

	th.AssertEquals(t, requestUUID, uuid)
}

func TestResizeCluster(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleResizeClusterSuccessfully(t)

	nodeCount := 2

	var opts clusters.ResizeOptsBuilder
	opts = clusters.ResizeOpts{
		NodeCount: &nodeCount,
	}

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"
	res := clusters.Resize(sc, clusterUUID, opts)
	th.AssertNoErr(t, res.Err)

	requestID := res.Header.Get("X-OpenStack-Request-Id")
	th.AssertEquals(t, requestUUID, requestID)

	actual, err := res.Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, nodeCount, actual.NodeCount)
}
