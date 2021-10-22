package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/objectstorage/v1/containers"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

// ExpectedListInfo is the result expected from a call to `List` when full
// info is requested.
var ExpectedListInfo = []containers.Container{
	{
		Count: 0,
		Bytes: 0,
		Name:  "janeausten",
	},
	{
		Count: 1,
		Bytes: 14,
		Name:  "marktwain",
	},
}

// ExpectedListNames is the result expected from a call to `List` when just
// container names are requested.
var ExpectedListNames = []string{"janeausten", "marktwain"}

// HandleListContainerInfoSuccessfully creates an HTTP handler at `/` on the test handler mux that
// responds with a `List` response when full info is requested.
func HandleListContainerInfoSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Set("Content-Type", "application/json")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, `[
        {
          "count": 0,
          "bytes": 0,
          "name": "janeausten"
        },
        {
          "count": 1,
          "bytes": 14,
          "name": "marktwain"
        }
      ]`)
		case "janeausten":
			fmt.Fprintf(w, `[
				{
					"count": 1,
					"bytes": 14,
					"name": "marktwain"
				}
			]`)
		case "marktwain":
			fmt.Fprintf(w, `[]`)
		default:
			t.Fatalf("Unexpected marker: [%s]", marker)
		}
	})
}

// HandleListContainerNamesSuccessfully creates an HTTP handler at `/` on the test handler mux that
// responds with a `ListNames` response when only container names are requested.
func HandleListContainerNamesSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "text/plain")

		w.Header().Set("Content-Type", "text/plain")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, "janeausten\nmarktwain\n")
		case "janeausten":
			fmt.Fprintf(w, "marktwain\n")
		case "marktwain":
			fmt.Fprintf(w, ``)
		default:
			t.Fatalf("Unexpected marker: [%s]", marker)
		}
	})
}

// HandleCreateContainerSuccessfully creates an HTTP handler at `/testContainer` on the test handler mux that
// responds with a `Create` response.
func HandleCreateContainerSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		w.Header().Add("X-Container-Meta-Foo", "bar")
		w.Header().Set("Content-Length", "0")
		w.Header().Set("Content-Type", "text/html; charset=UTF-8")
		w.Header().Set("Date", "Wed, 17 Aug 2016 19:25:43 GMT")
		w.Header().Set("X-Trans-Id", "tx554ed59667a64c61866f1-0058b4ba37")
		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleDeleteContainerSuccessfully creates an HTTP handler at `/testContainer` on the test handler mux that
// responds with a `Delete` response.
func HandleDeleteContainerSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleUpdateContainerSuccessfully creates an HTTP handler at `/testContainer` on the test handler mux that
// responds with a `Update` response.
func HandleUpdateContainerSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleGetContainerSuccessfully creates an HTTP handler at `/testContainer` on the test handler mux that
// responds with a `Get` response.
func HandleGetContainerSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "HEAD")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("Date", "Wed, 17 Aug 2016 19:25:43 GMT")
		w.Header().Set("X-Container-Bytes-Used", "100")
		w.Header().Set("X-Container-Object-Count", "4")
		w.Header().Set("X-Container-Read", "test")
		w.Header().Set("X-Container-Write", "test2,user4")
		w.Header().Set("X-Timestamp", "1471298837.95721")
		w.Header().Set("X-Trans-Id", "tx554ed59667a64c61866f1-0057b4ba37")
		w.Header().Set("X-Storage-Policy", "test_policy")
		w.WriteHeader(http.StatusNoContent)
	})
}
