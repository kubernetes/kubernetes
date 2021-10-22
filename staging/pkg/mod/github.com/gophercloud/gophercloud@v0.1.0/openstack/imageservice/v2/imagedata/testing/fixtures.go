package testing

import (
	"io/ioutil"
	"net/http"
	"testing"

	th "github.com/gophercloud/gophercloud/testhelper"
	fakeclient "github.com/gophercloud/gophercloud/testhelper/client"
)

// HandlePutImageDataSuccessfully setup
func HandlePutImageDataSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/images/da3b75d9-3f4a-40e7-8a2c-bfab23927dea/file", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fakeclient.TokenID)

		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Unable to read request body: %v", err)
		}

		th.AssertByteArrayEquals(t, []byte{5, 3, 7, 24}, b)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleStageImageDataSuccessfully setup
func HandleStageImageDataSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/images/da3b75d9-3f4a-40e7-8a2c-bfab23927dea/stage", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PUT")
		th.TestHeader(t, r, "X-Auth-Token", fakeclient.TokenID)

		b, err := ioutil.ReadAll(r.Body)
		if err != nil {
			t.Errorf("Unable to read request body: %v", err)
		}

		th.AssertByteArrayEquals(t, []byte{5, 3, 7, 24}, b)

		w.WriteHeader(http.StatusNoContent)
	})
}

// HandleGetImageDataSuccessfully setup
func HandleGetImageDataSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/images/da3b75d9-3f4a-40e7-8a2c-bfab23927dea/file", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fakeclient.TokenID)

		w.WriteHeader(http.StatusOK)

		_, err := w.Write([]byte{34, 87, 0, 23, 23, 23, 56, 255, 254, 0})
		th.AssertNoErr(t, err)
	})
}
