// Package certificates contains functionality for working with Magnum Certificate
// resources.
/*
Package certificates provides information and interaction with the certificates through
the OpenStack Container Infra service.

Example to get certificates

	certificate, err := certificates.Get(serviceClient, "d564b18a-2890-4152-be3d-e05d784ff72").Extract()
	if err != nil {
		panic(err)
	}

Example to create certificates

	opts := certificates.CreateOpts{
		BayUUID:	"d564b18a-2890-4152-be3d-e05d784ff727",
		CSR:		"-----BEGIN CERTIFICATE REQUEST-----\nMIIEfzCCAmcCAQAwFDESMBAGA1UEAxMJWW91ciBOYW1lMIICIjANBgkqhkiG9w0B\n-----END CERTIFICATE REQUEST-----\n",
	}

	response, err := certificates.Create(sc, opts).Extract()
	if err != nil {
		panic(err)
	}

Example to update certificates

	err := certificates.Update(client, clusterUUID).ExtractErr()
	if err != nil {
		panic(err)
	}
*/
package certificates
