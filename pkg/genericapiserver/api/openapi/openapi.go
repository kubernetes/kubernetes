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

package openapi

import (
	"bytes"
	"fmt"
	"strings"
	"unicode"

	"github.com/emicklei/go-restful"

	"k8s.io/kubernetes/pkg/util"
)

var verbs = util.CreateTrie([]string{"get", "log", "read", "replace", "patch", "delete", "deletecollection", "watch", "connect", "proxy", "list", "create", "patch"})

// ToValidOperationID makes an string a valid op ID (e.g. removing punctuations and whitespaces and make it camel case)
func ToValidOperationID(s string, capitalizeFirstLetter bool) string {
	var buffer bytes.Buffer
	capitalize := capitalizeFirstLetter
	for i, r := range s {
		if unicode.IsLetter(r) || r == '_' || (i != 0 && unicode.IsDigit(r)) {
			if capitalize {
				buffer.WriteRune(unicode.ToUpper(r))
				capitalize = false
			} else {
				buffer.WriteRune(r)
			}
		} else {
			capitalize = true
		}
	}
	return buffer.String()
}

// GetOperationIDAndTags returns a customize operation ID and a list of tags for kubernetes API server's OpenAPI spec to prevent duplicate IDs.
func GetOperationIDAndTags(servePath string, r *restful.Route) (string, []string, error) {
	op := r.Operation
	path := r.Path
	var tags []string
	// TODO: This is hacky, figure out where this name conflict is created and fix it at the root.
	if strings.HasPrefix(path, "/apis/extensions/v1beta1/namespaces/{namespace}/") && strings.HasSuffix(op, "ScaleScale") {
		op = op[:len(op)-10] + strings.Title(strings.Split(path[48:], "/")[0]) + "Scale"
	}
	switch servePath {
	case "/swagger.json":
		prefix, exists := verbs.GetPrefix(op)
		if !exists {
			return op, tags, fmt.Errorf("operation names should start with a verb. Cannot determine operation verb from %v", op)
		}
		op = op[len(prefix):]
		parts := strings.Split(strings.Trim(path, "/"), "/")
		// Assume /api is /apis/core, remove this when we actually server /api/... on /apis/core/...
		if len(parts) >= 1 && parts[0] == "api" {
			parts = append([]string{"apis", "core"}, parts[1:]...)
		}
		if len(parts) >= 2 && parts[0] == "apis" {
			prefix = prefix + ToValidOperationID(strings.TrimSuffix(parts[1], ".k8s.io"), prefix != "")
			tag := ToValidOperationID(strings.TrimSuffix(parts[1], ".k8s.io"), false)
			if len(parts) > 2 {
				prefix = prefix + ToValidOperationID(parts[2], prefix != "")
				tag = tag + "_" + ToValidOperationID(parts[2], false)
			}
			tags = append(tags, tag)
		} else if len(parts) >= 1 {
			tags = append(tags, ToValidOperationID(parts[0], false))
		}
		return prefix + ToValidOperationID(op, prefix != ""), tags, nil
	default:
		return op, tags, nil
	}
}
