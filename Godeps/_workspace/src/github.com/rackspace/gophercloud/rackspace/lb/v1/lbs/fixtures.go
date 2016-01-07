package lbs

import (
	"fmt"
	"net/http"
	"strconv"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func mockListLBResponse(t *testing.T) {
	th.Mux.HandleFunc("/loadbalancers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "loadBalancers":[
      {
         "name":"lb-site1",
         "id":71,
         "protocol":"HTTP",
         "port":80,
         "algorithm":"RANDOM",
         "status":"ACTIVE",
         "nodeCount":3,
         "virtualIps":[
            {
               "id":403,
               "address":"206.55.130.1",
               "type":"PUBLIC",
               "ipVersion":"IPV4"
            }
         ],
         "created":{
            "time":"2010-11-30T03:23:42Z"
         },
         "updated":{
            "time":"2010-11-30T03:23:44Z"
         }
      },
      {
         "name":"lb-site2",
         "id":72,
         "created":{
            "time":"2011-11-30T03:23:42Z"
         },
         "updated":{
            "time":"2011-11-30T03:23:44Z"
         }
      },
      {
         "name":"lb-site3",
         "id":73,
         "created":{
            "time":"2012-11-30T03:23:42Z"
         },
         "updated":{
            "time":"2012-11-30T03:23:44Z"
         }
      }
   ]
}
			`)
	})
}

func mockCreateLBResponse(t *testing.T) {
	th.Mux.HandleFunc("/loadbalancers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
  "loadBalancer": {
    "name": "a-new-loadbalancer",
    "port": 80,
    "protocol": "HTTP",
    "virtualIps": [
      {
        "id": 2341
      },
      {
        "id": 900001
      }
    ],
    "nodes": [
      {
        "address": "10.1.1.1",
        "port": 80,
        "condition": "ENABLED"
      }
    ]
  }
}
		`)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)

		fmt.Fprintf(w, `
{
  "loadBalancer": {
    "name": "a-new-loadbalancer",
    "id": 144,
    "protocol": "HTTP",
    "halfClosed": false,
    "port": 83,
    "algorithm": "RANDOM",
    "status": "BUILD",
    "timeout": 30,
    "cluster": {
      "name": "ztm-n01.staging1.lbaas.rackspace.net"
    },
    "nodes": [
      {
        "address": "10.1.1.1",
        "id": 653,
        "port": 80,
        "status": "ONLINE",
        "condition": "ENABLED",
        "weight": 1
      }
    ],
    "virtualIps": [
      {
        "address": "206.10.10.210",
        "id": 39,
        "type": "PUBLIC",
        "ipVersion": "IPV4"
      },
      {
        "address": "2001:4801:79f1:0002:711b:be4c:0000:0021",
        "id": 900001,
        "type": "PUBLIC",
        "ipVersion": "IPV6"
      }
    ],
    "created": {
      "time": "2010-11-30T03:23:42Z"
    },
    "updated": {
      "time": "2010-11-30T03:23:44Z"
    },
    "connectionLogging": {
      "enabled": false
    }
  }
}
	`)
	})
}

func mockBatchDeleteLBResponse(t *testing.T, ids []int) {
	th.Mux.HandleFunc("/loadbalancers", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		r.ParseForm()

		for k, v := range ids {
			fids := r.Form["id"]
			th.AssertEquals(t, strconv.Itoa(v), fids[k])
		}

		w.WriteHeader(http.StatusAccepted)
	})
}

func mockDeleteLBResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusAccepted)
	})
}

func mockGetLBResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "loadBalancer": {
    "id": 2000,
    "name": "sample-loadbalancer",
    "protocol": "HTTP",
    "port": 80,
    "algorithm": "RANDOM",
    "status": "ACTIVE",
    "timeout": 30,
    "connectionLogging": {
      "enabled": true
    },
    "virtualIps": [
      {
        "id": 1000,
        "address": "206.10.10.210",
        "type": "PUBLIC",
        "ipVersion": "IPV4"
      }
    ],
    "nodes": [
      {
        "id": 1041,
        "address": "10.1.1.1",
        "port": 80,
        "condition": "ENABLED",
        "status": "ONLINE"
      },
      {
        "id": 1411,
        "address": "10.1.1.2",
        "port": 80,
        "condition": "ENABLED",
        "status": "ONLINE"
      }
    ],
    "sessionPersistence": {
      "persistenceType": "HTTP_COOKIE"
    },
    "connectionThrottle": {
      "maxConnections": 100
    },
    "cluster": {
      "name": "c1.dfw1"
    },
    "created": {
      "time": "2010-11-30T03:23:42Z"
    },
    "updated": {
      "time": "2010-11-30T03:23:44Z"
    },
    "sourceAddresses": {
      "ipv6Public": "2001:4801:79f1:1::1/64",
      "ipv4Servicenet": "10.0.0.0",
      "ipv4Public": "10.12.99.28"
    }
  }
}
	`)
	})
}

func mockUpdateLBResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id), func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
	"loadBalancer": {
		"name": "a-new-loadbalancer",
		"protocol": "TCP",
		"halfClosed": true,
		"algorithm": "RANDOM",
		"port": 8080,
		"timeout": 100,
		"httpsRedirect": false
	}
}
		`)

		w.WriteHeader(http.StatusAccepted)
	})
}

func mockListProtocolsResponse(t *testing.T) {
	th.Mux.HandleFunc("/loadbalancers/protocols", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "protocols": [
    {
      "name": "DNS_TCP",
      "port": 53
    },
    {
      "name": "DNS_UDP",
      "port": 53
    },
    {
      "name": "FTP",
      "port": 21
    },
    {
      "name": "HTTP",
      "port": 80
    },
    {
      "name": "HTTPS",
      "port": 443
    },
    {
      "name": "IMAPS",
      "port": 993
    },
    {
      "name": "IMAPv4",
      "port": 143
    },
    {
      "name": "LDAP",
      "port": 389
    },
    {
      "name": "LDAPS",
      "port": 636
    },
    {
      "name": "MYSQL",
      "port": 3306
    },
    {
      "name": "POP3",
      "port": 110
    },
    {
      "name": "POP3S",
      "port": 995
    },
    {
      "name": "SMTP",
      "port": 25
    },
    {
      "name": "TCP",
      "port": 0
    },
    {
      "name": "TCP_CLIENT_FIRST",
      "port": 0
    },
    {
      "name": "UDP",
      "port": 0
    },
    {
      "name": "UDP_STREAM",
      "port": 0
    },
    {
      "name": "SFTP",
      "port": 22
    },
    {
      "name": "TCP_STREAM",
      "port": 0
    }
  ]
}
	`)
	})
}

func mockListAlgorithmsResponse(t *testing.T) {
	th.Mux.HandleFunc("/loadbalancers/algorithms", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "algorithms": [
    {
      "name": "LEAST_CONNECTIONS"
    },
    {
      "name": "RANDOM"
    },
    {
      "name": "ROUND_ROBIN"
    },
    {
      "name": "WEIGHTED_LEAST_CONNECTIONS"
    },
    {
      "name": "WEIGHTED_ROUND_ROBIN"
    }
  ]
}
			`)
	})
}

func mockGetLoggingResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/connectionlogging", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "connectionLogging": {
    "enabled": true
  }
}
			`)
	})
}

func mockEnableLoggingResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/connectionlogging", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
   "connectionLogging":{
      "enabled":true
   }
}
		`)

		w.WriteHeader(http.StatusAccepted)
	})
}

func mockDisableLoggingResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/connectionlogging", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
	"connectionLogging":{
			"enabled":false
	}
}
		`)

		w.WriteHeader(http.StatusAccepted)
	})
}

func mockGetErrorPageResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/errorpage", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "errorpage": {
    "content": "<html>DEFAULT ERROR PAGE</html>"
  }
}
			`)
	})
}

func mockSetErrorPageResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/errorpage", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
	"errorpage": {
		"content": "<html>New error page</html>"
	}
}
		`)

		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
	"errorpage": {
		"content": "<html>New error page</html>"
	}
}
			`)
	})
}

func mockDeleteErrorPageResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/errorpage", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		w.WriteHeader(http.StatusOK)
	})
}

func mockGetStatsResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/stats", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
  "connectTimeOut": 10,
  "connectError": 20,
  "connectFailure": 30,
  "dataTimedOut": 40,
  "keepAliveTimedOut": 50,
  "maxConn": 60,
  "currentConn": 40,
  "connectTimeOutSsl": 10,
  "connectErrorSsl": 20,
  "connectFailureSsl": 30,
  "dataTimedOutSsl": 40,
  "keepAliveTimedOutSsl": 50,
  "maxConnSsl": 60,
  "currentConnSsl": 40
}
			`)
	})
}

func mockGetCachingResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/contentcaching", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, `
{
   "contentCaching": {
      "enabled": true
   }
}
			`)
	})
}

func mockEnableCachingResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/contentcaching", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
   "contentCaching":{
      "enabled":true
   }
}
		`)

		w.WriteHeader(http.StatusAccepted)
	})
}

func mockDisableCachingResponse(t *testing.T, id int) {
	th.Mux.HandleFunc("/loadbalancers/"+strconv.Itoa(id)+"/contentcaching", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		th.TestJSONRequest(t, r, `
{
	"contentCaching":{
			"enabled":false
	}
}
		`)

		w.WriteHeader(http.StatusAccepted)
	})
}
