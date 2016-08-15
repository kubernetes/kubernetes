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

// Package healthcheck provides healthcheck URL generators and parsers
package healthcheckparser // import "k8s.io/kubernetes/pkg/util/healthcheckparser"

import (
	"errors"
	"fmt"
	"strings"
)

func GenerateURL(namespace, name string) string {
	return fmt.Sprintf("/healthchecks/svc/%s/%s", namespace, name)
}

func ParseURL(url string) (namespace, name string, err error) {
	parts := strings.Split(url, "/")
	if len(parts) < 5 {
		err = errors.New("Failed to parse url into healthcheck components")
		return
	}
	return parts[3], parts[4], nil
}
