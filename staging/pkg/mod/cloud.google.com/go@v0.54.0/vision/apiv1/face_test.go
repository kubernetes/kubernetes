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

package vision

import (
	"testing"

	"cloud.google.com/go/internal/testutil"
	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"
)

func TestFaceFromLandmarks(t *testing.T) {
	landmarks := []*pb.FaceAnnotation_Landmark{
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_EYE,
			Position: &pb.Position{X: 1192, Y: 575, Z: 0},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_EYE,
			Position: &pb.Position{X: 1479, Y: 571, Z: -9},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_OF_LEFT_EYEBROW,
			Position: &pb.Position{X: 1097, Y: 522, Z: 27},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_OF_LEFT_EYEBROW,
			Position: &pb.Position{X: 1266, Y: 521, Z: -61},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_OF_RIGHT_EYEBROW,
			Position: &pb.Position{X: 1402, Y: 520, Z: -66},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_OF_RIGHT_EYEBROW,
			Position: &pb.Position{X: 1571, Y: 519, Z: 10},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_MIDPOINT_BETWEEN_EYES,
			Position: &pb.Position{X: 1331, Y: 566, Z: -66},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_NOSE_TIP,
			Position: &pb.Position{X: 1329, Y: 743, Z: -137},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_UPPER_LIP,
			Position: &pb.Position{X: 1330, Y: 836, Z: -66},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LOWER_LIP,
			Position: &pb.Position{X: 1334, Y: 954, Z: -36},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_MOUTH_LEFT,
			Position: &pb.Position{X: 1186, Y: 867, Z: 27},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_MOUTH_RIGHT,
			Position: &pb.Position{X: 1484, Y: 857, Z: 19},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_MOUTH_CENTER,
			Position: &pb.Position{X: 1332, Y: 894, Z: -41},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_NOSE_BOTTOM_RIGHT,
			Position: &pb.Position{X: 1432, Y: 750, Z: -26},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_NOSE_BOTTOM_LEFT,
			Position: &pb.Position{X: 1236, Y: 755, Z: -20},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_NOSE_BOTTOM_CENTER,
			Position: &pb.Position{X: 1332, Y: 783, Z: -70},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_EYE_TOP_BOUNDARY,
			Position: &pb.Position{X: 1193, Y: 561, Z: -20},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_EYE_RIGHT_CORNER,
			Position: &pb.Position{X: 1252, Y: 581, Z: -1},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_EYE_BOTTOM_BOUNDARY,
			Position: &pb.Position{X: 1190, Y: 593, Z: -1},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_EYE_LEFT_CORNER,
			Position: &pb.Position{X: 1133, Y: 584, Z: 28},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_EYE_PUPIL,
			Position: &pb.Position{X: 1189, Y: 580, Z: -8},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_EYE_TOP_BOUNDARY,
			Position: &pb.Position{X: 1474, Y: 561, Z: -30},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_EYE_RIGHT_CORNER,
			Position: &pb.Position{X: 1536, Y: 581, Z: 15},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_EYE_BOTTOM_BOUNDARY,
			Position: &pb.Position{X: 1481, Y: 590, Z: -11},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_EYE_LEFT_CORNER,
			Position: &pb.Position{X: 1424, Y: 579, Z: -6},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_EYE_PUPIL,
			Position: &pb.Position{X: 1478, Y: 580, Z: -18},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_EYEBROW_UPPER_MIDPOINT,
			Position: &pb.Position{X: 1181, Y: 482, Z: -40},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_EYEBROW_UPPER_MIDPOINT,
			Position: &pb.Position{X: 1485, Y: 482, Z: -50},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_LEFT_EAR_TRAGION,
			Position: &pb.Position{X: 1027, Y: 696, Z: 361},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_RIGHT_EAR_TRAGION,
			Position: &pb.Position{X: 1666, Y: 695, Z: 339},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_FOREHEAD_GLABELLA,
			Position: &pb.Position{X: 1332, Y: 514, Z: -75},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_CHIN_GNATHION,
			Position: &pb.Position{X: 1335, Y: 1058, Z: 6},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_CHIN_LEFT_GONION,
			Position: &pb.Position{X: 1055, Y: 882, Z: 257},
		},
		{
			Type:     pb.FaceAnnotation_Landmark_CHIN_RIGHT_GONION,
			Position: &pb.Position{X: 1631, Y: 881, Z: 238},
		},
	}
	want := &FaceLandmarks{
		Eyebrows: Eyebrows{
			Left: Eyebrow{
				Top:   &pb.Position{X: 1181, Y: 482, Z: -40},
				Left:  &pb.Position{X: 1097, Y: 522, Z: 27},
				Right: &pb.Position{X: 1266, Y: 521, Z: -61},
			},
			Right: Eyebrow{
				Top:   &pb.Position{X: 1485, Y: 482, Z: -50},
				Left:  &pb.Position{X: 1402, Y: 520, Z: -66},
				Right: &pb.Position{X: 1571, Y: 519, Z: 10},
			},
		},
		Eyes: Eyes{
			Left: Eye{
				Left:   &pb.Position{X: 1133, Y: 584, Z: 28},
				Right:  &pb.Position{X: 1252, Y: 581, Z: -1},
				Top:    &pb.Position{X: 1193, Y: 561, Z: -20},
				Bottom: &pb.Position{X: 1190, Y: 593, Z: -1},
				Center: &pb.Position{X: 1192, Y: 575, Z: 0},
				Pupil:  &pb.Position{X: 1189, Y: 580, Z: -8},
			},
			Right: Eye{
				Left:   &pb.Position{X: 1424, Y: 579, Z: -6},
				Right:  &pb.Position{X: 1536, Y: 581, Z: 15},
				Top:    &pb.Position{X: 1474, Y: 561, Z: -30},
				Bottom: &pb.Position{X: 1481, Y: 590, Z: -11},
				Center: &pb.Position{X: 1479, Y: 571, Z: -9},
				Pupil:  &pb.Position{X: 1478, Y: 580, Z: -18},
			},
		},
		Ears: Ears{
			Left:  &pb.Position{X: 1027, Y: 696, Z: 361},
			Right: &pb.Position{X: 1666, Y: 695, Z: 339},
		},
		Nose: Nose{
			Left:   &pb.Position{X: 1236, Y: 755, Z: -20},
			Right:  &pb.Position{X: 1432, Y: 750, Z: -26},
			Top:    &pb.Position{X: 1331, Y: 566, Z: -66},
			Bottom: &pb.Position{X: 1332, Y: 783, Z: -70},
			Tip:    &pb.Position{X: 1329, Y: 743, Z: -137},
		},
		Mouth: Mouth{
			Left:     &pb.Position{X: 1186, Y: 867, Z: 27},
			Center:   &pb.Position{X: 1332, Y: 894, Z: -41},
			Right:    &pb.Position{X: 1484, Y: 857, Z: 19},
			UpperLip: &pb.Position{X: 1330, Y: 836, Z: -66},
			LowerLip: &pb.Position{X: 1334, Y: 954, Z: -36},
		},
		Chin: Chin{
			Left:   &pb.Position{X: 1055, Y: 882, Z: 257},
			Center: &pb.Position{X: 1335, Y: 1058, Z: 6},
			Right:  &pb.Position{X: 1631, Y: 881, Z: 238},
		},
		Forehead: &pb.Position{X: 1332, Y: 514, Z: -75},
	}

	got := FaceFromLandmarks(landmarks)
	if diff := testutil.Diff(got, want); diff != "" {
		t.Error(diff)
	}
}
