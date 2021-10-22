package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/containerinfra/v1/certificates"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestGetCertificates(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleGetCertificateSuccessfully(t)

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"

	actual, err := certificates.Get(sc, "d564b18a-2890-4152-be3d-e05d784ff72").Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCertificate, *actual)
}

func TestCreateCertificates(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleCreateCertificateSuccessfully(t)

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"

	opts := certificates.CreateOpts{
		BayUUID: "d564b18a-2890-4152-be3d-e05d784ff727",
		CSR:     "FAKE_CERTIFICATE_CSR",
	}

	actual, err := certificates.Create(sc, opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, ExpectedCreateCertificateResponse, *actual)
}

func TestUpdateCertificates(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	HandleUpdateCertificateSuccessfully(t)

	sc := fake.ServiceClient()
	sc.Endpoint = sc.Endpoint + "v1/"

	err := certificates.Update(sc, "d564b18a-2890-4152-be3d-e05d784ff72").ExtractErr()
	th.AssertNoErr(t, err)
}
