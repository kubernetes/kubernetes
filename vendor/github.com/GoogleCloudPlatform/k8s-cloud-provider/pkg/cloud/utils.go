/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cloud

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/meta"
)

const (
	gaPrefix    = "https://www.googleapis.com/compute/v1"
	alphaPrefix = "https://www.googleapis.com/compute/alpha"
	betaPrefix  = "https://www.googleapis.com/compute/beta"
)

// ResourceID identifies a GCE resource as parsed from compute resource URL.
type ResourceID struct {
	ProjectID string
	Resource  string
	Key       *meta.Key
}

// Equal returns true if two resource IDs are equal.
func (r *ResourceID) Equal(other *ResourceID) bool {
	if r.ProjectID != other.ProjectID || r.Resource != other.Resource {
		return false
	}
	if r.Key != nil && other.Key != nil {
		return *r.Key == *other.Key
	}
	if r.Key == nil && other.Key == nil {
		return true
	}
	return false
}

// RelativeResourceName returns the relative resource name string
// representing this ResourceID.
func (r *ResourceID) RelativeResourceName() string {
	return RelativeResourceName(r.ProjectID, r.Resource, r.Key)
}

// ResourcePath returns the resource path representing this ResourceID.
func (r *ResourceID) ResourcePath() string {
	return ResourcePath(r.Resource, r.Key)
}

func (r *ResourceID) SelfLink(ver meta.Version) string {
	return SelfLink(ver, r.ProjectID, r.Resource, r.Key)
}

// ParseResourceURL parses resource URLs of the following formats:
//
//   global/<res>/<name>
//   regions/<region>/<res>/<name>
//   zones/<zone>/<res>/<name>
//   projects/<proj>
//   projects/<proj>/global/<res>/<name>
//   projects/<proj>/regions/<region>/<res>/<name>
//   projects/<proj>/zones/<zone>/<res>/<name>
//   [https://www.googleapis.com/compute/<ver>]/projects/<proj>/global/<res>/<name>
//   [https://www.googleapis.com/compute/<ver>]/projects/<proj>/regions/<region>/<res>/<name>
//   [https://www.googleapis.com/compute/<ver>]/projects/<proj>/zones/<zone>/<res>/<name>
func ParseResourceURL(url string) (*ResourceID, error) {
	errNotValid := fmt.Errorf("%q is not a valid resource URL", url)

	// Trim prefix off URL leaving "projects/..."
	projectsIndex := strings.Index(url, "/projects/")
	if projectsIndex >= 0 {
		url = url[projectsIndex+1:]
	}

	parts := strings.Split(url, "/")
	if len(parts) < 2 || len(parts) > 6 {
		return nil, errNotValid
	}

	ret := &ResourceID{}
	scopedName := parts
	if parts[0] == "projects" {
		ret.Resource = "projects"
		ret.ProjectID = parts[1]
		scopedName = parts[2:]

		if len(scopedName) == 0 {
			return ret, nil
		}
	}

	switch scopedName[0] {
	case "global":
		if len(scopedName) != 3 {
			return nil, errNotValid
		}
		ret.Resource = scopedName[1]
		ret.Key = meta.GlobalKey(scopedName[2])
		return ret, nil
	case "regions":
		switch len(scopedName) {
		case 2:
			ret.Resource = "regions"
			ret.Key = meta.GlobalKey(scopedName[1])
			return ret, nil
		case 4:
			ret.Resource = scopedName[2]
			ret.Key = meta.RegionalKey(scopedName[3], scopedName[1])
			return ret, nil
		default:
			return nil, errNotValid
		}
	case "zones":
		switch len(scopedName) {
		case 2:
			ret.Resource = "zones"
			ret.Key = meta.GlobalKey(scopedName[1])
			return ret, nil
		case 4:
			ret.Resource = scopedName[2]
			ret.Key = meta.ZonalKey(scopedName[3], scopedName[1])
			return ret, nil
		default:
			return nil, errNotValid
		}
	}
	return nil, errNotValid
}

func copyViaJSON(dest, src interface{}) error {
	bytes, err := json.Marshal(src)
	if err != nil {
		return err
	}
	return json.Unmarshal(bytes, dest)
}

// ResourcePath returns the path starting from the location.
// Example: regions/us-central1/subnetworks/my-subnet
func ResourcePath(resource string, key *meta.Key) string {
	switch resource {
	case "zones", "regions":
		return fmt.Sprintf("%s/%s", resource, key.Name)
	case "projects":
		return "invalid-resource"
	}

	switch key.Type() {
	case meta.Zonal:
		return fmt.Sprintf("zones/%s/%s/%s", key.Zone, resource, key.Name)
	case meta.Regional:
		return fmt.Sprintf("regions/%s/%s/%s", key.Region, resource, key.Name)
	case meta.Global:
		return fmt.Sprintf("global/%s/%s", resource, key.Name)
	}
	return "invalid-key-type"
}

// RelativeResourceName returns the path starting from project.
// Example: projects/my-project/regions/us-central1/subnetworks/my-subnet
func RelativeResourceName(project, resource string, key *meta.Key) string {
	switch resource {
	case "projects":
		return fmt.Sprintf("projects/%s", project)
	default:
		return fmt.Sprintf("projects/%s/%s", project, ResourcePath(resource, key))
	}
}

// SelfLink returns the self link URL for the given object.
func SelfLink(ver meta.Version, project, resource string, key *meta.Key) string {
	var prefix string
	switch ver {
	case meta.VersionAlpha:
		prefix = alphaPrefix
	case meta.VersionBeta:
		prefix = betaPrefix
	case meta.VersionGA:
		prefix = gaPrefix
	default:
		prefix = "invalid-prefix"
	}

	return fmt.Sprintf("%s/%s", prefix, RelativeResourceName(project, resource, key))

}
