// Copyright 2016 Google LLC
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

package vision_test

import (
	"context"
	"fmt"
	"os"

	vision "cloud.google.com/go/vision/apiv1"
	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"
)

func Example_NewImageFromReader() {
	f, err := os.Open("path/to/image.jpg")
	if err != nil {
		// TODO: handle error.
	}
	img, err := vision.NewImageFromReader(f)
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(img)
}

func Example_NewImageFromURI() {
	img := vision.NewImageFromURI("gs://my-bucket/my-image.png")
	fmt.Println(img)
}

func ExampleImageAnnotatorClient_AnnotateImage() {
	ctx := context.Background()
	c, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	res, err := c.AnnotateImage(ctx, &pb.AnnotateImageRequest{
		Image: vision.NewImageFromURI("gs://my-bucket/my-image.png"),
		Features: []*pb.Feature{
			{Type: pb.Feature_LANDMARK_DETECTION, MaxResults: 5},
			{Type: pb.Feature_LABEL_DETECTION, MaxResults: 3},
		},
	})
	if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use res.
	_ = res
}

func Example_FaceFromLandmarks() {
	ctx := context.Background()
	c, err := vision.NewImageAnnotatorClient(ctx)
	if err != nil {
		// TODO: Handle error.
	}
	resp, err := c.BatchAnnotateImages(ctx, &pb.BatchAnnotateImagesRequest{
		Requests: []*pb.AnnotateImageRequest{
			{
				Image: vision.NewImageFromURI("gs://bucket/image.jpg"),
				Features: []*pb.Feature{{
					Type:       pb.Feature_FACE_DETECTION,
					MaxResults: 5,
				}},
			},
		},
	})
	if err != nil {
		// TODO: Handle error.
	}
	res := resp.Responses[0]
	if res.Error != nil {
		// TODO: Handle error.
	}
	for _, a := range res.FaceAnnotations {
		face := vision.FaceFromLandmarks(a.Landmarks)
		fmt.Println(face.Nose.Tip)
		fmt.Println(face.Eyes.Left.Pupil)
	}
}
