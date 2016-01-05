// +build fixtures

package objects

import (
	"crypto/md5"
	"fmt"
	"io"
	"net/http"
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

// HandleDownloadObjectSuccessfully creates an HTTP handler at `/testContainer/testObject` on the test handler mux that
// responds with a `Download` response.
func HandleDownloadObjectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer/testObject", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "Successful download with Gophercloud")
	})
}

// ExpectedListInfo is the result expected from a call to `List` when full
// info is requested.
var ExpectedListInfo = []Object{
	Object{
		Hash:         "451e372e48e0f6b1114fa0724aa79fa1",
		LastModified: "2009-11-10 23:00:00 +0000 UTC",
		Bytes:        14,
		Name:         "goodbye",
		ContentType:  "application/octet-stream",
	},
	Object{
		Hash:         "451e372e48e0f6b1114fa0724aa79fa1",
		LastModified: "2009-11-10 23:00:00 +0000 UTC",
		Bytes:        14,
		Name:         "hello",
		ContentType:  "application/octet-stream",
	},
}

// ExpectedListNames is the result expected from a call to `List` when just
// object names are requested.
var ExpectedListNames = []string{"hello", "goodbye"}

// HandleListObjectsInfoSuccessfully creates an HTTP handler at `/testContainer` on the test handler mux that
// responds with a `List` response when full info is requested.
func HandleListObjectsInfoSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer", func(w http.ResponseWriter, r *http.Request) {
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
        "hash": "451e372e48e0f6b1114fa0724aa79fa1",
        "last_modified": "2009-11-10 23:00:00 +0000 UTC",
        "bytes": 14,
        "name": "goodbye",
        "content_type": "application/octet-stream"
      },
      {
        "hash": "451e372e48e0f6b1114fa0724aa79fa1",
        "last_modified": "2009-11-10 23:00:00 +0000 UTC",
        "bytes": 14,
        "name": "hello",
        "content_type": "application/octet-stream"
      }
    ]`)
		case "hello":
			fmt.Fprintf(w, `[]`)
		default:
			t.Fatalf("Unexpected marker: [%s]", marker)
		}
	})
}

// HandleListObjectNamesSuccessfully creates an HTTP handler at `/testContainer` on the test handler mux that
// responds with a `List` response when only object names are requested.
func HandleListObjectNamesSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "text/plain")

		w.Header().Set("Content-Type", "text/plain")
		r.ParseForm()
		marker := r.Form.Get("marker")
		switch marker {
		case "":
			fmt.Fprintf(w, "hello\ngoodbye\n")
		case "goodbye":
			fmt.Fprintf(w, "")
		default:
			t.Fatalf("Unexpected marker: [%s]", marker)
		}
	})
}

// HandleCreateTextObjectSuccessfully creates an HTTP handler at `/testContainer/testObject` on the test handler mux
// that responds with a `Create` response. A Content-Type of "text/plain" is expected.
func HandleCreateTextObjectSuccessfully(t *testing.T, content string) {
	th.Mux.HandleFunc("/testContainer/testObject", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "text/plain")
		th.TestHeader(t, r, "Accept", "application/json")

		hash := md5.New()
		io.WriteString(hash, content)
		localChecksum := hash.Sum(nil)

		w.Header().Set("ETag", fmt.Sprintf("%x", localChecksum))
		w.WriteHeader(http.StatusCreated)
	})
}

// HandleCreateTypelessObjectSuccessfully creates an HTTP handler at `/testContainer/testObject` on the test handler
// mux that responds with a `Create` response. No Content-Type header may be present in the request, so that server-
// side content-type detection will be triggered properly.
func HandleCreateTypelessObjectSuccessfully(t *testing.T, content string) {
	th.Mux.HandleFunc("/testContainer/testObject", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		if contentType, present := r.Header["Content-Type"]; present {
			t.Errorf("Expected Content-Type header to be omitted, but was %#v", contentType)
		}

		hash := md5.New()
		io.WriteString(hash, content)
		localChecksum := hash.Sum(nil)

		w.Header().Set("ETag", fmt.Sprintf("%x", localChecksum))
		w.WriteHeader(http.StatusCreated)
	})
}

// HandleCopyObjectSuccessfully creates an HTTP handler at `/testContainer/testObject` on the test handler mux that
// responds with a `Copy` response.
func HandleCopyObjectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer/testObject", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "COPY")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "Destination", "/newTestContainer/newTestObject")
		w.WriteHeader(http.StatusCreated)
	})
}

// HandleDeleteObjectSuccessfully creates an HTTP handler at `/testContainer/testObject` on the test handler mux that
// responds with a `Delete` response.
func HandleDeleteObjectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer/testObject", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleUpdateObjectSuccessfully creates an HTTP handler at `/testContainer/testObject` on the test handler mux that
// responds with a `Update` response.
func HandleUpdateObjectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer/testObject", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestHeader(t, r, "X-Object-Meta-Gophercloud-Test", "objects")
		w.WriteHeader(http.StatusAccepted)
	})
}

// HandleGetObjectSuccessfully creates an HTTP handler at `/testContainer/testObject` on the test handler mux that
// responds with a `Get` response.
func HandleGetObjectSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/testContainer/testObject", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "HEAD")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")
		w.Header().Add("X-Object-Meta-Gophercloud-Test", "objects")
		w.WriteHeader(http.StatusNoContent)
	})
}
