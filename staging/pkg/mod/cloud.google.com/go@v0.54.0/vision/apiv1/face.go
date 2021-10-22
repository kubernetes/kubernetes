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
	"log"

	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"
)

// FaceLandmarks contains the positions of facial features detected by the service.
type FaceLandmarks struct {
	Eyebrows Eyebrows
	Eyes     Eyes
	Ears     Ears
	Nose     Nose
	Mouth    Mouth
	Chin     Chin
	Forehead *pb.Position
}

// Eyebrows represents a face's eyebrows.
type Eyebrows struct {
	Left, Right Eyebrow
}

// Eyebrow represents a face's eyebrow.
type Eyebrow struct {
	Top, Left, Right *pb.Position
}

// Eyes represents a face's eyes.
type Eyes struct {
	Left, Right Eye
}

// Eye represents a face's eye.
type Eye struct {
	Left, Right, Top, Bottom, Center, Pupil *pb.Position
}

// Ears represents a face's ears.
type Ears struct {
	Left, Right *pb.Position
}

// Nose represents a face's nose.
type Nose struct {
	Left, Right, Top, Bottom, Tip *pb.Position
}

// Mouth represents a face's mouth.
type Mouth struct {
	Left, Center, Right, UpperLip, LowerLip *pb.Position
}

// Chin represents a face's chin.
type Chin struct {
	Left, Center, Right *pb.Position
}

// FaceFromLandmarks converts the list of face landmarks returned by the service
// to a FaceLandmarks struct.
func FaceFromLandmarks(landmarks []*pb.FaceAnnotation_Landmark) *FaceLandmarks {
	face := &FaceLandmarks{}
	for _, lm := range landmarks {
		switch lm.Type {
		case pb.FaceAnnotation_Landmark_LEFT_OF_LEFT_EYEBROW:
			face.Eyebrows.Left.Left = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_OF_LEFT_EYEBROW:
			face.Eyebrows.Left.Right = lm.Position
		case pb.FaceAnnotation_Landmark_LEFT_OF_RIGHT_EYEBROW:
			face.Eyebrows.Right.Left = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_OF_RIGHT_EYEBROW:
			face.Eyebrows.Right.Right = lm.Position
		case pb.FaceAnnotation_Landmark_LEFT_EYEBROW_UPPER_MIDPOINT:
			face.Eyebrows.Left.Top = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_EYEBROW_UPPER_MIDPOINT:
			face.Eyebrows.Right.Top = lm.Position
		case pb.FaceAnnotation_Landmark_MIDPOINT_BETWEEN_EYES:
			face.Nose.Top = lm.Position
		case pb.FaceAnnotation_Landmark_NOSE_TIP:
			face.Nose.Tip = lm.Position
		case pb.FaceAnnotation_Landmark_UPPER_LIP:
			face.Mouth.UpperLip = lm.Position
		case pb.FaceAnnotation_Landmark_LOWER_LIP:
			face.Mouth.LowerLip = lm.Position
		case pb.FaceAnnotation_Landmark_MOUTH_LEFT:
			face.Mouth.Left = lm.Position
		case pb.FaceAnnotation_Landmark_MOUTH_RIGHT:
			face.Mouth.Right = lm.Position
		case pb.FaceAnnotation_Landmark_MOUTH_CENTER:
			face.Mouth.Center = lm.Position
		case pb.FaceAnnotation_Landmark_NOSE_BOTTOM_RIGHT:
			face.Nose.Right = lm.Position
		case pb.FaceAnnotation_Landmark_NOSE_BOTTOM_LEFT:
			face.Nose.Left = lm.Position
		case pb.FaceAnnotation_Landmark_NOSE_BOTTOM_CENTER:
			face.Nose.Bottom = lm.Position
		case pb.FaceAnnotation_Landmark_LEFT_EYE:
			face.Eyes.Left.Center = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_EYE:
			face.Eyes.Right.Center = lm.Position
		case pb.FaceAnnotation_Landmark_LEFT_EYE_TOP_BOUNDARY:
			face.Eyes.Left.Top = lm.Position
		case pb.FaceAnnotation_Landmark_LEFT_EYE_RIGHT_CORNER:
			face.Eyes.Left.Right = lm.Position
		case pb.FaceAnnotation_Landmark_LEFT_EYE_BOTTOM_BOUNDARY:
			face.Eyes.Left.Bottom = lm.Position
		case pb.FaceAnnotation_Landmark_LEFT_EYE_LEFT_CORNER:
			face.Eyes.Left.Left = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_TOP_BOUNDARY:
			face.Eyes.Right.Top = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_RIGHT_CORNER:
			face.Eyes.Right.Right = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_BOTTOM_BOUNDARY:
			face.Eyes.Right.Bottom = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_LEFT_CORNER:
			face.Eyes.Right.Left = lm.Position
		case pb.FaceAnnotation_Landmark_LEFT_EYE_PUPIL:
			face.Eyes.Left.Pupil = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_PUPIL:
			face.Eyes.Right.Pupil = lm.Position
		case pb.FaceAnnotation_Landmark_LEFT_EAR_TRAGION:
			face.Ears.Left = lm.Position
		case pb.FaceAnnotation_Landmark_RIGHT_EAR_TRAGION:
			face.Ears.Right = lm.Position
		case pb.FaceAnnotation_Landmark_FOREHEAD_GLABELLA:
			face.Forehead = lm.Position
		case pb.FaceAnnotation_Landmark_CHIN_GNATHION:
			face.Chin.Center = lm.Position
		case pb.FaceAnnotation_Landmark_CHIN_LEFT_GONION:
			face.Chin.Left = lm.Position
		case pb.FaceAnnotation_Landmark_CHIN_RIGHT_GONION:
			face.Chin.Right = lm.Position
		default:
			log.Printf("vision: ignoring unknown face annotation landmark %s", lm.Type)
		}
	}
	return face
}
