package kubenode

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/square/go-jose"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
)

func RetrieveTrustedClusterInfo(params *kubeadmapi.BootstrapParams) (*clientcmdapi.Config, error) {
	apiServerURL, err := url.Parse(strings.Split(params.Discovery.ApiServerURLs, ",")[0])
	if err != nil {
		return nil, err
	}

	host, port := strings.Split(apiServerURL.Host, ":")[0], 8081

	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s:%d/api/v1alpha1/testclusterinfo", host, port), nil)
	if err != nil {
		return nil, err
	}

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	buf := new(bytes.Buffer)
	io.Copy(buf, res.Body)
	res.Body.Close()

	object, err := jose.ParseSigned(buf.String())
	if err != nil {
		return nil, err
	}

	output, err := object.Verify([]byte(params.Discovery.BearerToken))
	if err != nil {
		return nil, err
	}

	clusterInfo := kubeadmapi.ClusterInfo{}

	if err := json.Unmarshal(output, &clusterInfo); err != nil {
		return nil, err
	}

	fmt.Printf("ClusterInfo: %#v", clusterInfo)

	if len(clusterInfo.CertificateAuthorities) == 0 || len(clusterInfo.Endpoints) == 0 {
		return nil, fmt.Errorf("Cluster discovery object (ClusterInfo) is invalid")
	}

	// TODO figure out what we should do when there is a chain of certificates and more then one API endpoint
	apiServer := clusterInfo.Endpoints[0]
	caCert := []byte(clusterInfo.CertificateAuthorities[0])

	return PerformTLSBootstrap(params, apiServer, caCert)
}
