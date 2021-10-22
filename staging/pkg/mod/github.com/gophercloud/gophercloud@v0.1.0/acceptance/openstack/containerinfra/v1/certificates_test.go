// +build acceptance containerinfra

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/containerinfra/v1/certificates"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestCertificatesCRUD(t *testing.T) {
	client, err := clients.NewContainerInfraV1Client()
	th.AssertNoErr(t, err)

	clusterUUID := "8934d2d1-6bce-4ffa-a017-fb437777269d"

	opts := certificates.CreateOpts{
		BayUUID: clusterUUID,
		CSR: "-----BEGIN CERTIFICATE REQUEST-----\n" +
			"MIIByjCCATMCAQAwgYkxCzAJBgNVBAYTAlVTMRMwEQYDVQQIEwpDYWxpZm9ybmlh" +
			"MRYwFAYDVQQHEw1Nb3VudGFpbiBWaWV3MRMwEQYDVQQKEwpHb29nbGUgSW5jMR8w" +
			"HQYDVQQLExZJbmZvcm1hdGlvbiBUZWNobm9sb2d5MRcwFQYDVQQDEw53d3cuZ29v" +
			"Z2xlLmNvbTCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEApZtYJCHJ4VpVXHfV" +
			"IlstQTlO4qC03hjX+ZkPyvdYd1Q4+qbAeTwXmCUKYHThVRd5aXSqlPzyIBwieMZr" +
			"WFlRQddZ1IzXAlVRDWwAo60KecqeAXnnUK+5fXoTI/UgWshre8tJ+x/TMHaQKR/J" +
			"cIWPhqaQhsJuzZbvAdGA80BLxdMCAwEAAaAAMA0GCSqGSIb3DQEBBQUAA4GBAIhl" +
			"4PvFq+e7ipARgI5ZM+GZx6mpCz44DTo0JkwfRDf+BtrsaC0q68eTf2XhYOsq4fkH" +
			"Q0uA0aVog3f5iJxCa3Hp5gxbJQ6zV6kJ0TEsuaaOhEko9sdpCoPOnRBm2i/XRD2D" +
			"6iNh8f8z0ShGsFqjDgFHyF3o+lUyj+UC6H1QW7bn\n" +
			"-----END CERTIFICATE REQUEST-----",
	}

	createResponse, err := certificates.Create(client, opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertEquals(t, opts.CSR, createResponse.CSR)

	certificate, err := certificates.Get(client, clusterUUID).Extract()
	th.AssertNoErr(t, err)
	t.Log(certificate.PEM)

	err = certificates.Update(client, clusterUUID).ExtractErr()
	th.AssertNoErr(t, err)
}
