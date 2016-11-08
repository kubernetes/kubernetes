package https

import (
	"net/url"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
)

func Parse(u *url.URL, c *kubeadm.Discovery) error {
	c.HTTPS = &kubeadm.HTTPSDiscovery{
		URL: u.String(),
	}
	return validation.ValidateDiscovery(c)
}
