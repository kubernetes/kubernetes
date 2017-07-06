/*
Copyright 2014 The Kubernetes Authors.

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

package libdocker

import (
	"strings"
	"time"

	dockerdigest "github.com/docker/distribution/digest"
	dockerref "github.com/docker/distribution/reference"
	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"
)

// ParseDockerTimestamp parses the timestamp returned by Interface from string to time.Time
func ParseDockerTimestamp(s string) (time.Time, error) {
	// Timestamp returned by Docker is in time.RFC3339Nano format.
	return time.Parse(time.RFC3339Nano, s)
}

// matchImageTagOrSHA checks if the given image specifier is a valid image ref,
// and that it matches the given image. It should fail on things like image IDs
// (config digests) and other digest-only references, but succeed on image names
// (`foo`), tag references (`foo:bar`), and manifest digest references
// (`foo@sha256:xyz`).
func matchImageTagOrSHA(inspected dockertypes.ImageInspect, image string) bool {
	// The image string follows the grammar specified here
	// https://github.com/docker/distribution/blob/master/reference/reference.go#L4
	named, err := dockerref.ParseNamed(image)
	if err != nil {
		glog.V(4).Infof("couldn't parse image reference %q: %v", image, err)
		return false
	}
	_, isTagged := named.(dockerref.Tagged)
	digest, isDigested := named.(dockerref.Digested)
	if !isTagged && !isDigested {
		// No Tag or SHA specified, so just return what we have
		return true
	}

	if isTagged {
		// Check the RepoTags for a match.
		for _, tag := range inspected.RepoTags {
			// An image name (without the tag/digest) can be [hostname '/'] component ['/' component]*
			// Because either the RepoTag or the name *may* contain the
			// hostname or not, we only check for the suffix match.
			if strings.HasSuffix(image, tag) || strings.HasSuffix(tag, image) {
				return true
			}
		}
	}

	if isDigested {
		for _, repoDigest := range inspected.RepoDigests {
			named, err := dockerref.ParseNamed(repoDigest)
			if err != nil {
				glog.V(4).Infof("couldn't parse image RepoDigest reference %q: %v", repoDigest, err)
				continue
			}
			if d, isDigested := named.(dockerref.Digested); isDigested {
				if digest.Digest().Algorithm().String() == d.Digest().Algorithm().String() &&
					digest.Digest().Hex() == d.Digest().Hex() {
					return true
				}
			}
		}

		// process the ID as a digest
		id, err := dockerdigest.ParseDigest(inspected.ID)
		if err != nil {
			glog.V(4).Infof("couldn't parse image ID reference %q: %v", id, err)
			return false
		}
		if digest.Digest().Algorithm().String() == id.Algorithm().String() && digest.Digest().Hex() == id.Hex() {
			return true
		}
	}
	glog.V(4).Infof("Inspected image (%q) does not match %s", inspected.ID, image)
	return false
}

// matchImageIDOnly checks that the given image specifier is a digest-only
// reference, and that it matches the given image.
func matchImageIDOnly(inspected dockertypes.ImageInspect, image string) bool {
	// If the image ref is literally equal to the inspected image's ID,
	// just return true here (this might be the case for Docker 1.9,
	// where we won't have a digest for the ID)
	if inspected.ID == image {
		return true
	}

	// Otherwise, we should try actual parsing to be more correct
	ref, err := dockerref.Parse(image)
	if err != nil {
		glog.V(4).Infof("couldn't parse image reference %q: %v", image, err)
		return false
	}

	digest, isDigested := ref.(dockerref.Digested)
	if !isDigested {
		glog.V(4).Infof("the image reference %q was not a digest reference", image)
		return false
	}

	id, err := dockerdigest.ParseDigest(inspected.ID)
	if err != nil {
		glog.V(4).Infof("couldn't parse image ID reference %q: %v", id, err)
		return false
	}

	if digest.Digest().Algorithm().String() == id.Algorithm().String() && digest.Digest().Hex() == id.Hex() {
		return true
	}

	glog.V(4).Infof("The reference %s does not directly refer to the given image's ID (%q)", image, inspected.ID)
	return false
}

// isImageNotFoundError returns whether the err is caused by image not found in docker
// TODO: Use native error tester once ImageNotFoundError is supported in docker-engine client(eg. ImageRemove())
func isImageNotFoundError(err error) bool {
	if err != nil {
		return strings.Contains(err.Error(), "No such image:")
	}
	return false
}
