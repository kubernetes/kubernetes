/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rkt

import (
	"fmt"
	"strings"
)

// image stores the appc image information.
// TODO(yifan): Replace with schema.ImageManifest.
type image struct {
	// The hash of the image, it must be universal unique. (e.g. sha512-xxx)
	id string
	// The name of the image manifest.
	name string
	// The version of the image. (e.g. v2.0.8, latest)
	version string
}

// parseString creates the image struct by parsing the string in the result of 'rkt images',
// the input looks like:
//
// sha512-91e98d7f1679a097c878203c9659f2a26ae394656b3147963324c61fa3832f15	coreos.com/etcd:v2.0.9
//
func (im *image) parseString(input string) error {
	idName := strings.Split(strings.TrimSpace(input), "\t")
	if len(idName) != 2 {
		return fmt.Errorf("invalid image information from 'rkt images': %q", input)
	}
	nameVersion := strings.Split(idName[1], ":")
	if len(nameVersion) != 2 {
		return fmt.Errorf("cannot parse the app name: %q", nameVersion)
	}
	im.id, im.name, im.version = idName[0], nameVersion[0], nameVersion[1]
	return nil
}
