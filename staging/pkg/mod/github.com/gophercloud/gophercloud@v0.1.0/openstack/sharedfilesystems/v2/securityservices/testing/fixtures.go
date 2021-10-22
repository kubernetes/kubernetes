package testing

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func MockCreateResponse(t *testing.T) {
	th.Mux.HandleFunc("/security-services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, `
        {
            "security_service": {
                "description": "Creating my first Security Service",
                "dns_ip": "10.0.0.0/24",
                "user": "demo",
                "password": "***",
                "type": "kerberos",
                "name": "SecServ1"
            }
        }`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
        {
            "security_service": {
                "status": "new",
                "domain": null,
                "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                "name": "SecServ1",
                "created_at": "2015-09-07T12:19:10.695211",
                "updated_at": null,
                "server": null,
                "dns_ip": "10.0.0.0/24",
                "user": "demo",
                "password": "supersecret",
                "type": "kerberos",
                "id": "3c829734-0679-4c17-9637-801da48c0d5f",
                "description": "Creating my first Security Service"
            }
        }`)
	})
}

func MockDeleteResponse(t *testing.T) {
	th.Mux.HandleFunc("/security-services/securityServiceID", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

func MockListResponse(t *testing.T) {
	th.Mux.HandleFunc("/security-services/detail", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
        {
            "security_services": [
                {
                    "status": "new",
                    "domain": null,
                    "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                    "name": "SecServ1",
                    "created_at": "2015-09-07T12:19:10.000000",
                    "description": "Creating my first Security Service",
                    "updated_at": null,
                    "server": null,
                    "dns_ip": "10.0.0.0/24",
                    "user": "demo",
                    "password": "supersecret",
                    "type": "kerberos",
                    "id": "3c829734-0679-4c17-9637-801da48c0d5f"
                },
                {
                    "status": "new",
                    "domain": null,
                    "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                    "name": "SecServ2",
                    "created_at": "2015-09-07T12:25:03.000000",
                    "description": "Creating my second Security Service",
                    "updated_at": null,
                    "server": null,
                    "dns_ip": "10.0.0.0/24",
                    "user": null,
                    "password": null,
                    "type": "ldap",
                    "id": "5a1d3a12-34a7-4087-8983-50e9ed03509a"
                }
            ]
        }`)
	})
}

func MockFilteredListResponse(t *testing.T) {
	th.Mux.HandleFunc("/security-services/detail", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
        {
            "security_services": [
                {
                    "status": "new",
                    "domain": null,
                    "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                    "name": "SecServ1",
                    "created_at": "2015-09-07T12:19:10.000000",
                    "description": "Creating my first Security Service",
                    "updated_at": null,
                    "server": null,
                    "dns_ip": "10.0.0.0/24",
                    "user": "demo",
                    "password": "supersecret",
                    "type": "kerberos",
                    "id": "3c829734-0679-4c17-9637-801da48c0d5f"
                }
            ]
        }`)
	})
}

func MockGetResponse(t *testing.T) {
	th.Mux.HandleFunc("/security-services/3c829734-0679-4c17-9637-801da48c0d5f", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
        {
            "security_service": {
                "status": "new",
                "domain": null,
                "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                "name": "SecServ1",
                "created_at": "2015-09-07T12:19:10.000000",
                "updated_at": null,
                "server": null,
                "dns_ip": "10.0.0.0/24",
                "user": "demo",
                "password": "supersecret",
                "type": "kerberos",
                "id": "3c829734-0679-4c17-9637-801da48c0d5f",
                "description": "Creating my first Security Service"
            }
        }`)
	})
}

func MockUpdateResponse(t *testing.T) {
	th.Mux.HandleFunc("/security-services/securityServiceID", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
            {
                "security_service": {
                    "status": "new",
                    "domain": null,
                    "project_id": "16e1ab15c35a457e9c2b2aa189f544e1",
                    "name": "SecServ2",
                    "created_at": "2015-09-07T12:19:10.000000",
                    "updated_at": "2015-09-07T12:20:10.000000",
                    "server": null,
                    "dns_ip": "10.0.0.0/24",
                    "user": "demo",
                    "password": "supersecret",
                    "type": "kerberos",
                    "id": "securityServiceID",
                    "description": "Updating my first Security Service"
                }
            }
        `)
	})
}
