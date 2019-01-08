/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"net/url"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// FromApiserverCache modifies <opts> so that the GET request will
// be served from apiserver cache instead of from etcd.
func FromApiserverCache(opts *metav1.GetOptions) {
	opts.ResourceVersion = "0"
}

func parseEndpoint(endpoint string) (string, string, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", "", err
	}

	if u.Scheme == "tcp" {
		return "tcp", u.Host, nil
	} else if u.Scheme == "unix" {
		return "unix", u.Path, nil
	} else if u.Scheme == "" {
		return "", "", fmt.Errorf("Using %q as endpoint is deprecated, please consider using full url format", endpoint)
	} else {
		return u.Scheme, "", fmt.Errorf("protocol %q not supported", u.Scheme)
	}
}

const (
	CPUOvercommitRatioAnnotation = "k8s.qiniu.com/cpu-overcommit-ratio"
	CPUOvercommitRatioMin        = 0.1
	CPUOvercommitRatioMax        = 10
)

// GetCPUOvercommitRatio returns CPU over-commmit ratio of node.
func GetCPUOvercommitRatio(node *v1.Node) float64 {
	if ratio, ok := node.Annotations[CPUOvercommitRatioAnnotation]; ok {
		ratio_f, err := strconv.ParseFloat(ratio, 64)
		if err != nil {
			glog.Errorf("failed to parse ratio data: %s", ratio)
			return 1.0
		}
		if ratio_f < CPUOvercommitRatioMin {
			ratio_f = CPUOvercommitRatioMin
		} else if ratio_f > CPUOvercommitRatioMax {
			ratio_f = CPUOvercommitRatioMax
		}
		return ratio_f
	}
	return 1.0
}
