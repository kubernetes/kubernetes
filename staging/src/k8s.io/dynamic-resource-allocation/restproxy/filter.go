/*
Copyright 2024 The Kubernetes Authors.

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

package restproxy

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	resourceapi "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	scheme "k8s.io/client-go/kubernetes/scheme"
)

var apiPathRE = regexp.MustCompile(`^/apis/([^/]+)/[^/]+/([^/]+)(/[^/]+)?$`)

// FilterDRADriver limits which API server requests a DRA driver may execute
// with the credentials of kubelet.
type FilterDRADriver struct {
	NodeName   string
	DriverName string
}

func (f FilterDRADriver) FilterRequest(ctx context.Context, req *http.Request, apiPath string) error {
	if req.URL == nil {
		return errors.New("missing URL")
	}

	submatches := apiPathRE.FindStringSubmatch(apiPath)
	if len(submatches) == 0 {
		return fmt.Errorf("unsupported API path: %s", apiPath)
	}
	resource := submatches[1] + "/" + submatches[2]
	name := submatches[3]

	switch resource {
	case "resource.k8s.io/resourceslices":
		switch req.Method {
		case http.MethodGet:
			if name == "" {
				// Getting the resourceslices directory is a list.
				//
				// Add or override the field values. DRA drivers could do that
				// themselves, but perhaps they forgot. Doing this is just a
				// courtesy: the node name is checked by the API server, but
				// not the driver name, so in GET or DELETE operations for
				// individual objects, one driver has access to objects of some
				// other driver on the node.
				var options metav1.ListOptions
				if req.URL.RawQuery != "" {
					values, err := url.ParseQuery(req.URL.RawQuery)
					if err != nil {
						return fmt.Errorf("parse raw query: %v", err)
					}

					// The assumption here is that encoding of ListOptions does not depend on
					// the version.
					if err := scheme.ParameterCodec.DecodeParameters(values, resourceapi.SchemeGroupVersion, &options); err != nil {
						return fmt.Errorf("extract ListOptions from raw query: %v", err)
					}
				}

				fields := parseSet(options.FieldSelector)
				fields["nodeName"] = f.NodeName
				fields["driverName"] = f.DriverName
				options.FieldSelector = fields.String()

				values, err := scheme.ParameterCodec.EncodeParameters(&options, resourceapi.SchemeGroupVersion)
				if err != nil {
					return fmt.Errorf("re-encode ListOptions: %v", err)
				}

				req.URL.RawQuery = values.Encode()
			}
		default:
			// All methods are allowed.
		}
	default:
		return fmt.Errorf("unsupported API resource: %s", resource)
	}

	return nil
}

// parseSet is the inverse of [fields.Set.String] (https://github.com/kubernetes/apimachinery/blob/d794766488ac2892197a7cc8d0b4b46b0edbda80/pkg/fields/fields.go#L36-L46).
func parseSet(selector string) fields.Set {
	set := fields.Set{}
	if selector == "" {
		return set
	}
	pairs := strings.Split(selector, ",")
	for _, pair := range pairs {
		kv := strings.SplitN(pair, "=", 2)
		key := kv[0]
		value := ""
		if len(kv) > 1 {
			value = kv[1]
		}
		set[key] = value
	}
	return set
}
