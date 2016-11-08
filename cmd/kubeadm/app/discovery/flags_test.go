package discovery

import (
	"reflect"
	"testing"

	"github.com/davecgh/go-spew/spew"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestParseURL(t *testing.T) {
	cases := []struct {
		url       string
		expect    kubeadm.Discovery
		expectErr bool
	}{
		{
			url: "token://c05de9:ab224260fb3cd718@192.168.0.1:6555,191.168.0.2:6443",
			expect: kubeadm.Discovery{
				Token: &kubeadm.TokenDiscovery{
					TokenID: "c05de9",
					Token:   "ab224260fb3cd718",
					Addresses: []string{
						"192.168.0.1:6555",
						"191.168.0.2:6443",
					},
				},
			},
		},
		{
			url: "file:///foo/bar/baz",
			expect: kubeadm.Discovery{
				File: &kubeadm.FileDiscovery{
					Path: "/foo/bar/baz",
				},
			},
		},
		{
			url: "https://storage.googleapis.com/kubeadm-disco/clusters/217651295213",
			expect: kubeadm.Discovery{
				HTTPS: &kubeadm.HTTPSDiscovery{
					URL: "https://storage.googleapis.com/kubeadm-disco/clusters/217651295213",
				},
			},
		},
	}
	for _, c := range cases {
		var d kubeadm.Discovery
		if err := ParseURL(&d, c.url); err != nil {
			if !c.expectErr {
				t.Errorf("unexpected error parsing discovery url: %v", err)
			}
			continue
		}
		if !reflect.DeepEqual(d, c.expect) {
			t.Errorf("expected discovery config to be equeal but got:\n\ta: %s\n\tb: %s", spew.Sdump(d), spew.Sdump(c.expect))
		}

	}
}
