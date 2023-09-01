/*
Copyright 2022 The KCP Authors.

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

package client

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/kcp-dev/logicalcluster/v3"
)

// apiRegex matches any string that has /api/ or /apis/ in it.
var apiRegex = regexp.MustCompile(`(/api/|/apis/)`)

// generatePath formats the request path to target the specified cluster.
func generatePath(originalPath string, clusterPath logicalcluster.Path) string {
	// If the originalPath already has cluster.Path() then the path was already modifed and no change needed
	if strings.Contains(originalPath, clusterPath.RequestPath()) {
		return originalPath
	}
	// If the originalPath has /api/ or /apis/ in it, it might be anywhere in the path, so we use a regex to find and
	// replaces /api/ or /apis/ with $cluster/api/ or $cluster/apis/
	if apiRegex.MatchString(originalPath) {
		return apiRegex.ReplaceAllString(originalPath, fmt.Sprintf("%s$1", clusterPath.RequestPath()))
	}

	// Otherwise, we're just prepending /clusters/$name
	path := clusterPath.RequestPath()

	// if the original path is relative, add a / separator
	if len(originalPath) > 0 && originalPath[0] != '/' {
		path += "/"
	}

	// finally append the original path
	path += originalPath

	return path
}
