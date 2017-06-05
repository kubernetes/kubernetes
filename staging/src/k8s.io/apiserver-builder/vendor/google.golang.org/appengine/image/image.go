// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Package image provides image services.
package image // import "google.golang.org/appengine/image"

import (
	"fmt"
	"net/url"

	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/image"
)

type ServingURLOptions struct {
	Secure bool // whether the URL should use HTTPS

	// Size must be between zero and 1600.
	// If Size is non-zero, a resized version of the image is served,
	// and Size is the served image's longest dimension. The aspect ratio is preserved.
	// If Crop is true the image is cropped from the center instead of being resized.
	Size int
	Crop bool
}

// ServingURL returns a URL that will serve an image from Blobstore.
func ServingURL(c context.Context, key appengine.BlobKey, opts *ServingURLOptions) (*url.URL, error) {
	req := &pb.ImagesGetUrlBaseRequest{
		BlobKey: (*string)(&key),
	}
	if opts != nil && opts.Secure {
		req.CreateSecureUrl = &opts.Secure
	}
	res := &pb.ImagesGetUrlBaseResponse{}
	if err := internal.Call(c, "images", "GetUrlBase", req, res); err != nil {
		return nil, err
	}

	// The URL may have suffixes added to dynamically resize or crop:
	// - adding "=s32" will serve the image resized to 32 pixels, preserving the aspect ratio.
	// - adding "=s32-c" is the same as "=s32" except it will be cropped.
	u := *res.Url
	if opts != nil && opts.Size > 0 {
		u += fmt.Sprintf("=s%d", opts.Size)
		if opts.Crop {
			u += "-c"
		}
	}
	return url.Parse(u)
}

// DeleteServingURL deletes the serving URL for an image.
func DeleteServingURL(c context.Context, key appengine.BlobKey) error {
	req := &pb.ImagesDeleteUrlBaseRequest{
		BlobKey: (*string)(&key),
	}
	res := &pb.ImagesDeleteUrlBaseResponse{}
	return internal.Call(c, "images", "DeleteUrlBase", req, res)
}

func init() {
	internal.RegisterErrorCodeMap("images", pb.ImagesServiceError_ErrorCode_name)
}
