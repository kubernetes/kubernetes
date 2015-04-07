package services

import (
	"fmt"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

// HandleListCDNServiceSuccessfully creates an HTTP handler at `/services` on the test handler mux
// that responds with a `List` response.
func HandleListCDNServiceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, `
    {
        "links": [
            {
                "rel": "next",
                "href": "https://www.poppycdn.io/v1.0/services?marker=96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0&limit=20"
            }
        ],
        "services": [
            {
                "id": "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
                "name": "mywebsite.com",
                "domains": [
                    {
                        "domain": "www.mywebsite.com"
                    }
                ],
                "origins": [
                    {
                        "origin": "mywebsite.com",
                        "port": 80,
                        "ssl": false
                    }
                ],
                "caching": [
                    {
                        "name": "default",
                        "ttl": 3600
                    },
                    {
                        "name": "home",
                        "ttl": 17200,
                        "rules": [
                            {
                                "name": "index",
                                "request_url": "/index.htm"
                            }
                        ]
                    },
                    {
                        "name": "images",
                        "ttl": 12800,
                        "rules": [
                            {
                                "name": "images",
                                "request_url": "*.png"
                            }
                        ]
                    }
                ],
                "restrictions": [
                    {
                        "name": "website only",
                        "rules": [
                            {
                                "name": "mywebsite.com",
                                "referrer": "www.mywebsite.com"
                            }
                        ]
                    }
                ],
                "flavor_id": "asia",
                "status": "deployed",
                "errors" : [],
                "links": [
                    {
                        "href": "https://www.poppycdn.io/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
                        "rel": "self"
                    },
                    {
                        "href": "mywebsite.com.cdn123.poppycdn.net",
                        "rel": "access_url"
                    },
                    {
                        "href": "https://www.poppycdn.io/v1.0/flavors/asia",
                        "rel": "flavor"
                    }
                ]
            },
            {
                "id": "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f1",
                "name": "myothersite.com",
                "domains": [
                    {
                        "domain": "www.myothersite.com"
                    }
                ],
                "origins": [
                    {
                        "origin": "44.33.22.11",
                        "port": 80,
                        "ssl": false
                    },
                    {
                        "origin": "77.66.55.44",
                        "port": 80,
                        "ssl": false,
                        "rules": [
                            {
                                "name": "videos",
                                "request_url": "^/videos/*.m3u"
                            }
                        ]
                    }
                ],
                "caching": [
                    {
                        "name": "default",
                        "ttl": 3600
                    }
                ],
                "restrictions": [
                    {}
                ],
                "flavor_id": "europe",
                "status": "deployed",
                "links": [
                    {
                        "href": "https://www.poppycdn.io/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f1",
                        "rel": "self"
                    },
                    {
                        "href": "myothersite.com.poppycdn.net",
                        "rel": "access_url"
                    },
                    {
                        "href": "https://www.poppycdn.io/v1.0/flavors/europe",
                        "rel": "flavor"
                    }
                ]
            }
        ]
    }
    `)
		case "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f1":
			fmt.Fprintf(w, `{
        "services": []
    }`)
		default:
			t.Fatalf("Unexpected marker: [%s]", marker)
		}
	})
}

// HandleCreateCDNServiceSuccessfully creates an HTTP handler at `/services` on the test handler mux
// that responds with a `Create` response.
func HandleCreateCDNServiceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestJSONRequest(t, r, `
      {
        "name": "mywebsite.com",
        "domains": [
            {
                "domain": "www.mywebsite.com"
            },
            {
                "domain": "blog.mywebsite.com"
            }
        ],
        "origins": [
            {
                "origin": "mywebsite.com",
                "port": 80,
                "ssl": false
            }
        ],
        "restrictions": [
                         {
                         "name": "website only",
                         "rules": [
                                   {
                                   "name": "mywebsite.com",
                                   "referrer": "www.mywebsite.com"
                    }
                ]
            }
        ],
        "caching": [
            {
                "name": "default",
                "ttl": 3600
            }
        ],

        "flavor_id": "cdn"
      }
   `)
		w.Header().Add("Location", "https://global.cdn.api.rackspacecloud.com/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0")
		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleGetCDNServiceSuccessfully creates an HTTP handler at `/services/{id}` on the test handler mux
// that responds with a `Get` response.
func HandleGetCDNServiceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
    {
        "id": "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
        "name": "mywebsite.com",
        "domains": [
            {
                "domain": "www.mywebsite.com",
                "protocol": "http"
            }
        ],
        "origins": [
            {
                "origin": "mywebsite.com",
                "port": 80,
                "ssl": false
            }
        ],
        "caching": [
            {
                "name": "default",
                "ttl": 3600
            },
            {
                "name": "home",
                "ttl": 17200,
                "rules": [
                    {
                        "name": "index",
                        "request_url": "/index.htm"
                    }
                ]
            },
            {
                "name": "images",
                "ttl": 12800,
                "rules": [
                    {
                        "name": "images",
                        "request_url": "*.png"
                    }
                ]
            }
        ],
        "restrictions": [
            {
                "name": "website only",
                "rules": [
                    {
                        "name": "mywebsite.com",
                        "referrer": "www.mywebsite.com"
                    }
                ]
            }
        ],
        "flavor_id": "cdn",
        "status": "deployed",
        "errors" : [],
        "links": [
            {
                "href": "https://global.cdn.api.rackspacecloud.com/v1.0/110011/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0",
                "rel": "self"
            },
            {
                "href": "blog.mywebsite.com.cdn1.raxcdn.com",
                "rel": "access_url"
            },
            {
                "href": "https://global.cdn.api.rackspacecloud.com/v1.0/110011/flavors/cdn",
                "rel": "flavor"
            }
        ]
    }
    `)
	})
}

// HandleUpdateCDNServiceSuccessfully creates an HTTP handler at `/services/{id}` on the test handler mux
// that responds with a `Update` response.
func HandleUpdateCDNServiceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestJSONRequest(t, r, `
      [
				{
					"op": "add",
					"path": "/domains/-",
					"value": {"domain": "appended.mocksite4.com"}
				},
				{
					"op": "add",
					"path": "/domains/4",
					"value": {"domain": "inserted.mocksite4.com"}
				},
				{
					"op": "add",
					"path": "/domains",
					"value": [
						{"domain": "bulkadded1.mocksite4.com"},
						{"domain": "bulkadded2.mocksite4.com"}
					]
				},
				{
					"op": "replace",
					"path": "/origins/2",
					"value": {"origin": "44.33.22.11", "port": 80, "ssl": false}
				},
				{
					"op": "replace",
					"path": "/origins",
					"value": [
						{"origin": "44.33.22.11", "port": 80, "ssl": false},
						{"origin": "55.44.33.22", "port": 443, "ssl": true}
					]
				},
				{
					"op": "remove",
					"path": "/caching/8"
				},
				{
					"op": "remove",
					"path": "/caching"
				},
				{
					"op": "replace",
					"path": "/name",
					"value": "differentServiceName"
				}
    ]
   `)
		w.Header().Add("Location", "https://www.poppycdn.io/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0")
		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleDeleteCDNServiceSuccessfully creates an HTTP handler at `/services/{id}` on the test handler mux
// that responds with a `Delete` response.
func HandleDeleteCDNServiceSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}
