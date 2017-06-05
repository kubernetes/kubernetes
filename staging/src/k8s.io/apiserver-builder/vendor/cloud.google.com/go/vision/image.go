// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
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

// An Image represents the contents of an image to run detection algorithms on,
// along with metadata. Images may be described by their raw bytes, or by a
// reference to a a Google Cloude Storage (GCS) object.
type Image struct {
	// Exactly one of content and gcsURI will be non-zero.
	content []byte // raw image bytes
	gcsURI  string // URI of the form "gs://BUCKET/OBJECT"

	// Rect is a rectangle on the Earth's surface represented by the
	// image. It is optional.
	Rect *LatLngRect

	// LanguageHints is a list of languages to use for text detection. In most
	// cases, leaving this field nil yields the best results since it enables
	// automatic language detection. For languages based on the Latin alphabet,
	// setting LanguageHints is not needed. In rare cases, when the language of
	// the text in the image is known, setting a hint will help get better
	// results (although it will be a significant hindrance if the hint is
	// wrong). Text detection returns an error if one or more of the specified
	// languages is not one of the supported languages (See
	// https://cloud.google.com/translate/v2/translate-reference#supported_languages).
	LanguageHints []string
}

// NewImageFromReader reads the bytes of an image from rc, then closes rc.
//
// You may optionally set Rect and LanguageHints on the returned Image before
// using it.
func NewImageFromReader(r io.ReadCloser) (*Image, error) {
	bytes, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	if err := r.Close(); err != nil {
		return nil, err
	}
	return &Image{content: bytes}, nil
}

// NewImageFromGCS returns an image that refers to an object in Google Cloud Storage.
// gcsPath must be a valid Google Cloud Storage URI of the form "gs://BUCKET/OBJECT".
//
// You may optionally set Rect and LanguageHints on the returned Image before
// using it.
func NewImageFromGCS(gcsURI string) *Image {
	return &Image{gcsURI: gcsURI}
}

// toProtos converts the Image to the two underlying API protos it represents,
// pb.Image and pb.ImageContext.
func (img *Image) toProtos() (*pb.Image, *pb.ImageContext) {
	var pimg *pb.Image
	switch {
	case img.content != nil:
		pimg = &pb.Image{Content: img.content}
	case img.gcsURI != "":
		pimg = &pb.Image{Source: &pb.ImageSource{GcsImageUri: img.gcsURI}}
	}

	var pctx *pb.ImageContext
	if img.Rect != nil || len(img.LanguageHints) > 0 {
		pctx = &pb.ImageContext{
			LatLongRect:   img.Rect.toProto(),
			LanguageHints: img.LanguageHints,
		}
	}

	return pimg, pctx
}
