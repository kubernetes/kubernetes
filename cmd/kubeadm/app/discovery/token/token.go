package token

import (
	"net/url"
	"strings"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
)

func Parse(u *url.URL, c *kubeadm.Discovery) error {
	var (
		hosts          []string
		tokenID, token string
	)
	if u.Host != "" {
		hosts = strings.Split(u.Host, ",")
	}
	if p, ok := u.User.Password(); ok {
		tokenID = u.User.Username()
		token = p
	}
	c.Token = &kubeadm.TokenDiscovery{
		TokenID:   tokenID,
		Token:     token,
		Addresses: hosts,
	}
	return validation.ValidateDiscovery(c)
}
