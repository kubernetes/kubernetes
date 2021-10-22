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
	"fmt"
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"
	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"
	"google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

var batchResponse = &pb.BatchAnnotateImagesResponse{
	Responses: []*pb.AnnotateImageResponse{{
		FaceAnnotations: []*pb.FaceAnnotation{
			{RollAngle: 1}, {RollAngle: 2}},
		LandmarkAnnotations:       []*pb.EntityAnnotation{{Mid: "landmark"}},
		LogoAnnotations:           []*pb.EntityAnnotation{{Mid: "logo"}},
		LabelAnnotations:          []*pb.EntityAnnotation{{Mid: "label"}},
		TextAnnotations:           []*pb.EntityAnnotation{{Mid: "text"}},
		FullTextAnnotation:        &pb.TextAnnotation{Text: "full"},
		SafeSearchAnnotation:      &pb.SafeSearchAnnotation{Spoof: pb.Likelihood_POSSIBLE},
		ImagePropertiesAnnotation: &pb.ImageProperties{DominantColors: &pb.DominantColorsAnnotation{}},
		CropHintsAnnotation:       &pb.CropHintsAnnotation{CropHints: []*pb.CropHint{{Confidence: 0.5}}},
		WebDetection:              &pb.WebDetection{WebEntities: []*pb.WebDetection_WebEntity{{EntityId: "web"}}},
	}},
}

// Verify that all the "shortcut" methods use the underlying
// BatchAnnotateImages RPC correctly.
func TestClientMethods(t *testing.T) {
	ctx := context.Background()
	c, err := NewImageAnnotatorClient(ctx, clientOpt)
	if err != nil {
		t.Fatal(err)
	}

	mockImageAnnotator.resps = []proto.Message{batchResponse}
	img := &pb.Image{Source: &pb.ImageSource{ImageUri: "http://foo.jpg"}}
	ictx := &pb.ImageContext{LanguageHints: []string{"en", "fr"}}
	req := &pb.AnnotateImageRequest{
		Image:        img,
		ImageContext: ictx,
		Features: []*pb.Feature{
			{Type: pb.Feature_LABEL_DETECTION, MaxResults: 3},
			{Type: pb.Feature_FACE_DETECTION, MaxResults: 4},
		},
	}

	for i, test := range []struct {
		call         func() (interface{}, error)
		wantFeatures []*pb.Feature
		wantRes      interface{}
	}{
		{
			func() (interface{}, error) { return c.AnnotateImage(ctx, req) },
			req.Features, batchResponse.Responses[0],
		},
		{
			func() (interface{}, error) { return c.DetectFaces(ctx, img, ictx, 2) },
			[]*pb.Feature{{Type: pb.Feature_FACE_DETECTION, MaxResults: 2}},
			batchResponse.Responses[0].FaceAnnotations,
		},
		{
			func() (interface{}, error) { return c.DetectLandmarks(ctx, img, ictx, 2) },
			[]*pb.Feature{{Type: pb.Feature_LANDMARK_DETECTION, MaxResults: 2}},
			batchResponse.Responses[0].LandmarkAnnotations,
		},
		{
			func() (interface{}, error) { return c.DetectLogos(ctx, img, ictx, 2) },
			[]*pb.Feature{{Type: pb.Feature_LOGO_DETECTION, MaxResults: 2}},
			batchResponse.Responses[0].LogoAnnotations,
		},
		{
			func() (interface{}, error) { return c.DetectLabels(ctx, img, ictx, 2) },
			[]*pb.Feature{{Type: pb.Feature_LABEL_DETECTION, MaxResults: 2}},
			batchResponse.Responses[0].LabelAnnotations,
		},
		{
			func() (interface{}, error) { return c.DetectTexts(ctx, img, ictx, 2) },
			[]*pb.Feature{{Type: pb.Feature_TEXT_DETECTION, MaxResults: 2}},
			batchResponse.Responses[0].TextAnnotations,
		},
		{
			func() (interface{}, error) { return c.DetectDocumentText(ctx, img, ictx) },
			[]*pb.Feature{{Type: pb.Feature_DOCUMENT_TEXT_DETECTION, MaxResults: 0}},
			batchResponse.Responses[0].FullTextAnnotation,
		},
		{
			func() (interface{}, error) { return c.DetectSafeSearch(ctx, img, ictx) },
			[]*pb.Feature{{Type: pb.Feature_SAFE_SEARCH_DETECTION, MaxResults: 0}},
			batchResponse.Responses[0].SafeSearchAnnotation,
		},
		{
			func() (interface{}, error) { return c.DetectImageProperties(ctx, img, ictx) },
			[]*pb.Feature{{Type: pb.Feature_IMAGE_PROPERTIES, MaxResults: 0}},
			batchResponse.Responses[0].ImagePropertiesAnnotation,
		},
		{
			func() (interface{}, error) { return c.DetectWeb(ctx, img, ictx) },
			[]*pb.Feature{{Type: pb.Feature_WEB_DETECTION, MaxResults: 0}},
			batchResponse.Responses[0].WebDetection,
		},
		{
			func() (interface{}, error) { return c.CropHints(ctx, img, ictx) },
			[]*pb.Feature{{Type: pb.Feature_CROP_HINTS, MaxResults: 0}},
			batchResponse.Responses[0].CropHintsAnnotation,
		},
		{
			func() (interface{}, error) { return c.LocalizeObjects(ctx, img, ictx) },
			[]*pb.Feature{{Type: pb.Feature_OBJECT_LOCALIZATION, MaxResults: 0}},
			batchResponse.Responses[0].LocalizedObjectAnnotations,
		},
		{
			func() (interface{}, error) { return c.ProductSearch(ctx, img, ictx) },
			[]*pb.Feature{{Type: pb.Feature_PRODUCT_SEARCH, MaxResults: 0}},
			batchResponse.Responses[0].ProductSearchResults,
		},
	} {
		mockImageAnnotator.reqs = nil
		res, err := test.call()
		if err != nil {
			t.Fatal(err)
		}
		got := mockImageAnnotator.reqs[0]
		want := &pb.BatchAnnotateImagesRequest{
			Requests: []*pb.AnnotateImageRequest{{
				Image:        img,
				ImageContext: ictx,
				Features:     test.wantFeatures,
			}},
		}
		if !testEqual(got, want) {
			t.Errorf("#%d:\ngot  %v\nwant %v", i, got, want)
		}
		if got, want := res, test.wantRes; !testEqual(got, want) {
			t.Errorf("#%d:\ngot  %v\nwant %v", i, got, want)
		}
	}

}

func testEqual(a, b interface{}) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	t := reflect.TypeOf(a)
	if t != reflect.TypeOf(b) {
		return false
	}
	if am, ok := a.(proto.Message); ok {
		return proto.Equal(am, b.(proto.Message))
	}
	if t.Kind() != reflect.Slice {
		panic(fmt.Sprintf("testEqual can only handle proto.Message and slices, got %s", t))
	}
	va := reflect.ValueOf(a)
	vb := reflect.ValueOf(b)
	if va.Len() != vb.Len() {
		return false
	}
	for i := 0; i < va.Len(); i++ {
		if !testEqual(va.Index(i).Interface(), vb.Index(i).Interface()) {
			return false
		}
	}
	return true
}

func TestAnnotateOneError(t *testing.T) {
	ctx := context.Background()
	c, err := NewImageAnnotatorClient(ctx, clientOpt)
	if err != nil {
		t.Fatal(err)
	}
	mockImageAnnotator.resps = []proto.Message{
		&pb.BatchAnnotateImagesResponse{
			Responses: []*pb.AnnotateImageResponse{{
				Error: &status.Status{Code: int32(codes.NotFound), Message: "not found"},
			}},
		},
	}

	_, err = c.annotateOne(ctx,
		&pb.Image{Source: &pb.ImageSource{ImageUri: "http://foo.jpg"}},
		nil, pb.Feature_LOGO_DETECTION, 1, nil)
	if c := grpc.Code(err); c != codes.NotFound {
		t.Errorf("got %v, want NotFound", c)
	}
}
