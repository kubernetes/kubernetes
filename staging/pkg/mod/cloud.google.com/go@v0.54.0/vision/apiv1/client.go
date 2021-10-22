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
	"context"

	gax "github.com/googleapis/gax-go/v2"
	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// AnnotateImage runs image detection and annotation for a single image.
func (c *ImageAnnotatorClient) AnnotateImage(ctx context.Context, req *pb.AnnotateImageRequest, opts ...gax.CallOption) (*pb.AnnotateImageResponse, error) {
	res, err := c.BatchAnnotateImages(ctx, &pb.BatchAnnotateImagesRequest{
		Requests: []*pb.AnnotateImageRequest{req},
	}, opts...)
	if err != nil {
		return nil, err
	}
	return res.Responses[0], nil
}

// Called for a single image and a single feature.
func (c *ImageAnnotatorClient) annotateOne(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, ftype pb.Feature_Type, maxResults int, opts []gax.CallOption) (*pb.AnnotateImageResponse, error) {
	res, err := c.AnnotateImage(ctx, &pb.AnnotateImageRequest{
		Image:        img,
		ImageContext: ictx,
		Features:     []*pb.Feature{{Type: ftype, MaxResults: int32(maxResults)}},
	}, opts...)
	if err != nil {
		return nil, err
	}
	// When there is only one image and one feature, the response's Error field is
	// unambiguously about that one detection, so we "promote" it to the error return
	// value.
	// res.Error is a google.rpc.Status. Convert to a Go error. Use a gRPC
	// error because it preserves the code as a separate field.
	// TODO(jba): preserve the details field.
	if res.Error != nil {
		return nil, status.Errorf(codes.Code(res.Error.Code), "%s", res.Error.Message)
	}
	return res, nil
}

// DetectFaces performs face detection on the image.
// At most maxResults results are returned.
func (c *ImageAnnotatorClient) DetectFaces(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, maxResults int, opts ...gax.CallOption) ([]*pb.FaceAnnotation, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_FACE_DETECTION, maxResults, opts)
	if err != nil {
		return nil, err
	}
	return res.FaceAnnotations, nil
}

// DetectLandmarks performs landmark detection on the image.
// At most maxResults results are returned.
func (c *ImageAnnotatorClient) DetectLandmarks(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, maxResults int, opts ...gax.CallOption) ([]*pb.EntityAnnotation, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_LANDMARK_DETECTION, maxResults, opts)
	if err != nil {
		return nil, err
	}
	return res.LandmarkAnnotations, nil
}

// DetectLogos performs logo detection on the image.
// At most maxResults results are returned.
func (c *ImageAnnotatorClient) DetectLogos(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, maxResults int, opts ...gax.CallOption) ([]*pb.EntityAnnotation, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_LOGO_DETECTION, maxResults, opts)
	if err != nil {
		return nil, err
	}
	return res.LogoAnnotations, nil
}

// DetectLabels performs label detection on the image.
// At most maxResults results are returned.
func (c *ImageAnnotatorClient) DetectLabels(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, maxResults int, opts ...gax.CallOption) ([]*pb.EntityAnnotation, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_LABEL_DETECTION, maxResults, opts)
	if err != nil {
		return nil, err
	}
	return res.LabelAnnotations, nil
}

// DetectTexts performs text detection on the image.
// At most maxResults results are returned.
func (c *ImageAnnotatorClient) DetectTexts(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, maxResults int, opts ...gax.CallOption) ([]*pb.EntityAnnotation, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_TEXT_DETECTION, maxResults, opts)
	if err != nil {
		return nil, err
	}
	return res.TextAnnotations, nil
}

// DetectDocumentText performs full text (OCR) detection on the image.
func (c *ImageAnnotatorClient) DetectDocumentText(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, opts ...gax.CallOption) (*pb.TextAnnotation, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_DOCUMENT_TEXT_DETECTION, 0, opts)
	if err != nil {
		return nil, err
	}
	return res.FullTextAnnotation, nil
}

// DetectSafeSearch performs safe-search detection on the image.
func (c *ImageAnnotatorClient) DetectSafeSearch(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, opts ...gax.CallOption) (*pb.SafeSearchAnnotation, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_SAFE_SEARCH_DETECTION, 0, opts)
	if err != nil {
		return nil, err
	}
	return res.SafeSearchAnnotation, nil
}

// DetectImageProperties computes properties of the image.
func (c *ImageAnnotatorClient) DetectImageProperties(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, opts ...gax.CallOption) (*pb.ImageProperties, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_IMAGE_PROPERTIES, 0, opts)
	if err != nil {
		return nil, err
	}
	return res.ImagePropertiesAnnotation, nil
}

// DetectWeb computes a web annotation on the image.
func (c *ImageAnnotatorClient) DetectWeb(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, opts ...gax.CallOption) (*pb.WebDetection, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_WEB_DETECTION, 0, opts)
	if err != nil {
		return nil, err
	}
	return res.WebDetection, nil
}

// CropHints computes crop hints for the image.
func (c *ImageAnnotatorClient) CropHints(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, opts ...gax.CallOption) (*pb.CropHintsAnnotation, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_CROP_HINTS, 0, opts)
	if err != nil {
		return nil, err
	}
	return res.CropHintsAnnotation, nil
}

// LocalizeObject runs the localizer for object detection.
func (c *ImageAnnotatorClient) LocalizeObjects(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, opts ...gax.CallOption) ([]*pb.LocalizedObjectAnnotation, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_OBJECT_LOCALIZATION, 0, opts)
	if err != nil {
		return nil, err
	}
	return res.LocalizedObjectAnnotations, nil
}

// ProductSearch searches the image for products.
func (c *ImageAnnotatorClient) ProductSearch(ctx context.Context, img *pb.Image, ictx *pb.ImageContext, opts ...gax.CallOption) (*pb.ProductSearchResults, error) {
	res, err := c.annotateOne(ctx, img, ictx, pb.Feature_PRODUCT_SEARCH, 0, opts)
	if err != nil {
		return nil, err
	}
	return res.ProductSearchResults, nil
}
