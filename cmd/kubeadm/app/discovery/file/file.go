package file

import (
	"net/url"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
)

func Parse(u *url.URL, c *kubeadm.Discovery) error {
	c.File = &kubeadm.FileDiscovery{
		Path: u.Path,
	}
	return validation.ValidateDiscovery(c)
}
