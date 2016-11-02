package node

import (
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"

	certclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion"
	restclient "k8s.io/kubernetes/pkg/client/restclient"
)

func TestPerformTLSBootstrap(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch req.URL.Path {
		default:
			output, err := json.Marshal(nil)
			if err != nil {
				t.Errorf("unexpected encoding error: %v", err)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write(output)
		}
	}))
	defer srv.Close()

	tests := []struct {
		h      string
		expect bool
	}{
		{
			h:      "",
			expect: false,
		},
		{
			h:      "localhost",
			expect: false,
		},
		{
			h:      srv.URL,
			expect: false,
		},
	}
	for _, rt := range tests {
		cd := &ConnectionDetails{}
		r := &restclient.Config{Host: rt.h}
		tmpConfig, err := certclient.NewForConfig(r)
		if err != nil {
			log.Fatal(err)
		}
		cd.CertClient = tmpConfig
		_, actual := PerformTLSBootstrap(cd)
		if (actual == nil) != rt.expect {
			t.Errorf(
				"failed createClients:\n\texpected: %t\n\t  actual: %t",
				rt.expect,
				(actual == nil),
			)
		}
	}
}
