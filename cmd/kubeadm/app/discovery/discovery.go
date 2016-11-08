package discovery

import (
	"fmt"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

func For(c kubeadmapi.Discovery) (*clientcmdapi.Config, error) {
	switch {
	default:
		return nil, fmt.Errorf("unimplemented")
	}
}
