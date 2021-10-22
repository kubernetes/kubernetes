package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/containerinfra/v1/certificates"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const CertificateResponse = `
{
	"cluster_uuid": "d564b18a-2890-4152-be3d-e05d784ff727",
	"bay_uuid": "d564b18a-2890-4152-be3d-e05d784ff727",
	"pem": "FAKE_CERTIFICATE",
	"links": [
	  {
  		"href": "http://10.63.176.154:9511/v1/certificates/d564b18a-2890-4152-be3d-e05d784ff727",
		  "rel": "self"
	  },
	  {
  		"href": "http://10.63.176.154:9511/certificates/d564b18a-2890-4152-be3d-e05d784ff727",
		  "rel": "bookmark"
	  }
	]
}`

const CreateCertificateResponse = `
{
	"cluster_uuid": "d564b18a-2890-4152-be3d-e05d784ff727",
	"bay_uuid": "d564b18a-2890-4152-be3d-e05d784ff727",
	"pem": "FAKE_CERTIFICATE_PEM",
	"csr": "FAKE_CERTIFICATE_CSR",
	"links": [
	  {
  		"href": "http://10.63.176.154:9511/v1/certificates/d564b18a-2890-4152-be3d-e05d784ff727",
		  "rel": "self"
	  },
	  {
  		"href": "http://10.63.176.154:9511/certificates/d564b18a-2890-4152-be3d-e05d784ff727",
		  "rel": "bookmark"
	  }
	]
}`

var ExpectedCertificate = certificates.Certificate{
	ClusterUUID: "d564b18a-2890-4152-be3d-e05d784ff727",
	BayUUID:     "d564b18a-2890-4152-be3d-e05d784ff727",
	PEM:         "FAKE_CERTIFICATE",
	Links: []gophercloud.Link{
		{Href: "http://10.63.176.154:9511/v1/certificates/d564b18a-2890-4152-be3d-e05d784ff727", Rel: "self"},
		{Href: "http://10.63.176.154:9511/certificates/d564b18a-2890-4152-be3d-e05d784ff727", Rel: "bookmark"},
	},
}

var ExpectedCreateCertificateResponse = certificates.Certificate{
	ClusterUUID: "d564b18a-2890-4152-be3d-e05d784ff727",
	BayUUID:     "d564b18a-2890-4152-be3d-e05d784ff727",
	PEM:         "FAKE_CERTIFICATE_PEM",
	CSR:         "FAKE_CERTIFICATE_CSR",
	Links: []gophercloud.Link{
		{Href: "http://10.63.176.154:9511/v1/certificates/d564b18a-2890-4152-be3d-e05d784ff727", Rel: "self"},
		{Href: "http://10.63.176.154:9511/certificates/d564b18a-2890-4152-be3d-e05d784ff727", Rel: "bookmark"},
	},
}

func HandleGetCertificateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/certificates/d564b18a-2890-4152-be3d-e05d784ff72", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("OpenStack-API-Minimum-Version", "container-infra 1.1")
		w.Header().Add("OpenStack-API-Maximum-Version", "container-infra 1.6")
		w.Header().Add("OpenStack-API-Version", "container-infra 1.1")
		w.Header().Add("X-OpenStack-Request-Id", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, CertificateResponse)
	})
}

func HandleCreateCertificateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/certificates/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("OpenStack-API-Minimum-Version", "container-infra 1.1")
		w.Header().Add("OpenStack-API-Maximum-Version", "container-infra 1.6")
		w.Header().Add("OpenStack-API-Version", "container-infra 1.1")
		w.Header().Add("X-OpenStack-Request-Id", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprint(w, CreateCertificateResponse)
	})
}

func HandleUpdateCertificateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/certificates/d564b18a-2890-4152-be3d-e05d784ff72",
		func(w http.ResponseWriter, r *http.Request) {
			th.TestMethod(t, r, "PATCH")
			th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

			w.WriteHeader(http.StatusAccepted)
			fmt.Fprintf(w, `{}`)
		})
}
