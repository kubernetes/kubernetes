/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package discovery

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/spf13/pflag"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/file"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/https"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/token"
)

type discoveryValue struct {
	v *kubeadm.Discovery
}

func NewDiscoveryValue(d *kubeadm.Discovery) pflag.Value {
	return &discoveryValue{
		v: d,
	}
}

func (d *discoveryValue) String() string {
	switch {
	case d.v.HTTPS != nil:
		return d.v.HTTPS.URL
	case d.v.File != nil:
		return "file://" + d.v.File.Path
	case d.v.Token != nil:
		return fmt.Sprintf("token://%s:%s@%s", d.v.Token.ID, d.v.Token.Secret, strings.Join(d.v.Token.Addresses, ","))
	default:
		return "unknown"
	}
}

func (d *discoveryValue) Set(s string) error {
	var kd kubeadm.Discovery
	if err := ParseURL(&kd, s); err != nil {
		return err
	}
	*d.v = kd
	return nil
}

func (d *discoveryValue) Type() string {
	return "discovery"
}

func ParseURL(d *kubeadm.Discovery, s string) error {
	u, err := url.Parse(s)
	if err != nil {
		return err
	}
	switch u.Scheme {
	case "https":
		return https.Parse(u, d)
	case "file":
		return file.Parse(u, d)
	case "token":
		// Make sure a valid RFC 3986 URL has been passed and parsed.
		// See https://github.com/kubernetes/kubeadm/issues/95#issuecomment-270431296 for more details.
		if !strings.Contains(s, "@") {
			s := s + "@"
			u, err = url.Parse(s)
			if err != nil {
				return err
			}
		}
		return token.Parse(u, d)
	default:
		return fmt.Errorf("unknown discovery scheme")
	}
}
