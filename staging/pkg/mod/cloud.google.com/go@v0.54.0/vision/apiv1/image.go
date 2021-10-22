// Copyright 2017, Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package vision

import (
	"io"
	"io/ioutil"

	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"
)

// NewImageFromReader reads the bytes of an image from r.
func NewImageFromReader(r io.Reader) (*pb.Image, error) {
	bytes, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return &pb.Image{Content: bytes}, nil
}

// NewImageFromURI returns an image that refers to an object in Google Cloud Storage
// (when the uri is of the form "gs://BUCKET/OBJECT") or at a public URL.
func NewImageFromURI(uri string) *pb.Image {
	return &pb.Image{Source: &pb.ImageSource{ImageUri: uri}}
}
