package kubemaster

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"

	"github.com/square/go-jose"

	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
)

func NewDiscoveryEndpoint(params *kubeadmapi.BootstrapParams, caCert string) {

	clusterInfo, err := json.MarshalIndent(kubeadmapi.ClusterInfo{
		Endpoints:              []string{fmt.Sprintf("https://%s:443", params.Discovery.ListenIP)},
		CertificateAuthorities: []string{caCert},
	}, "", "  ")
	if err != nil {
		log.Fatal(err)
	}

	signer, err := jose.NewSigner(jose.SignatureAlgorithm("HS256"), []byte(params.Discovery.BearerToken))
	if err != nil {
		log.Fatal(err)
	}

	http.HandleFunc("/api/v1alpha1/testclusterinfo", func(w http.ResponseWriter, r *http.Request) {

		obj, err := signer.Sign(clusterInfo)
		if err != nil {
			log.Println("Error:", err)
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintln(w, "Error:", err)
			return
		}

		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, obj.FullSerialize())
		return
	})

	listener, err := net.ListenTCP("tcp", &net.TCPAddr{Port: 8081})
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Listening on %v", listener.Addr())
	log.Printf("Shared secret is %q", params.Discovery.BearerToken)
	log.Fatal(http.Serve(listener, nil))

}
