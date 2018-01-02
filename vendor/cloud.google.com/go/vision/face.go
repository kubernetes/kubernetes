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
	"log"

	"github.com/golang/geo/r3"
	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"
)

// FaceLandmarks contains the positions of facial features detected by the service.
// TODO(jba): write doc for all
type FaceLandmarks struct {
	Eyebrows Eyebrows
	Eyes     Eyes
	Ears     Ears
	Nose     Nose
	Mouth    Mouth
	Chin     Chin
	Forehead *r3.Vector
}

type Eyebrows struct {
	Left, Right Eyebrow
}

type Eyebrow struct {
	Top, Left, Right *r3.Vector
}

type Eyes struct {
	Left, Right Eye
}

type Eye struct {
	Left, Right, Top, Bottom, Center, Pupil *r3.Vector
}

type Ears struct {
	Left, Right *r3.Vector
}

type Nose struct {
	Left, Right, Top, Bottom, Tip *r3.Vector
}

type Mouth struct {
	Left, Center, Right, UpperLip, LowerLip *r3.Vector
}

type Chin struct {
	Left, Center, Right *r3.Vector
}

// FaceLikelihoods  expresses the likelihood of various aspects of a face.
type FaceLikelihoods struct {
	// Joy is the likelihood that the face expresses joy.
	Joy Likelihood

	// Sorrow is the likelihood that the face expresses sorrow.
	Sorrow Likelihood

	// Anger is the likelihood that the face expresses anger.
	Anger Likelihood

	// Surprise is the likelihood that the face expresses surprise.
	Surprise Likelihood

	// UnderExposed is the likelihood that the face is under-exposed.
	UnderExposed Likelihood

	// Blurred is the likelihood that the face is blurred.
	Blurred Likelihood

	// Headwear is the likelihood that the face has headwear.
	Headwear Likelihood
}

func populateFaceLandmarks(landmarks []*pb.FaceAnnotation_Landmark, face *FaceLandmarks) {
	for _, lm := range landmarks {
		pos := &r3.Vector{
			X: float64(lm.Position.X),
			Y: float64(lm.Position.Y),
			Z: float64(lm.Position.Z),
		}
		switch lm.Type {
		case pb.FaceAnnotation_Landmark_LEFT_OF_LEFT_EYEBROW:
			face.Eyebrows.Left.Left = pos
		case pb.FaceAnnotation_Landmark_RIGHT_OF_LEFT_EYEBROW:
			face.Eyebrows.Left.Right = pos
		case pb.FaceAnnotation_Landmark_LEFT_OF_RIGHT_EYEBROW:
			face.Eyebrows.Right.Left = pos
		case pb.FaceAnnotation_Landmark_RIGHT_OF_RIGHT_EYEBROW:
			face.Eyebrows.Right.Right = pos
		case pb.FaceAnnotation_Landmark_LEFT_EYEBROW_UPPER_MIDPOINT:
			face.Eyebrows.Left.Top = pos
		case pb.FaceAnnotation_Landmark_RIGHT_EYEBROW_UPPER_MIDPOINT:
			face.Eyebrows.Right.Top = pos
		case pb.FaceAnnotation_Landmark_MIDPOINT_BETWEEN_EYES:
			face.Nose.Top = pos
		case pb.FaceAnnotation_Landmark_NOSE_TIP:
			face.Nose.Tip = pos
		case pb.FaceAnnotation_Landmark_UPPER_LIP:
			face.Mouth.UpperLip = pos
		case pb.FaceAnnotation_Landmark_LOWER_LIP:
			face.Mouth.LowerLip = pos
		case pb.FaceAnnotation_Landmark_MOUTH_LEFT:
			face.Mouth.Left = pos
		case pb.FaceAnnotation_Landmark_MOUTH_RIGHT:
			face.Mouth.Right = pos
		case pb.FaceAnnotation_Landmark_MOUTH_CENTER:
			face.Mouth.Center = pos
		case pb.FaceAnnotation_Landmark_NOSE_BOTTOM_RIGHT:
			face.Nose.Right = pos
		case pb.FaceAnnotation_Landmark_NOSE_BOTTOM_LEFT:
			face.Nose.Left = pos
		case pb.FaceAnnotation_Landmark_NOSE_BOTTOM_CENTER:
			face.Nose.Bottom = pos
		case pb.FaceAnnotation_Landmark_LEFT_EYE:
			face.Eyes.Left.Center = pos
		case pb.FaceAnnotation_Landmark_RIGHT_EYE:
			face.Eyes.Right.Center = pos
		case pb.FaceAnnotation_Landmark_LEFT_EYE_TOP_BOUNDARY:
			face.Eyes.Left.Top = pos
		case pb.FaceAnnotation_Landmark_LEFT_EYE_RIGHT_CORNER:
			face.Eyes.Left.Right = pos
		case pb.FaceAnnotation_Landmark_LEFT_EYE_BOTTOM_BOUNDARY:
			face.Eyes.Left.Bottom = pos
		case pb.FaceAnnotation_Landmark_LEFT_EYE_LEFT_CORNER:
			face.Eyes.Left.Left = pos
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_TOP_BOUNDARY:
			face.Eyes.Right.Top = pos
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_RIGHT_CORNER:
			face.Eyes.Right.Right = pos
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_BOTTOM_BOUNDARY:
			face.Eyes.Right.Bottom = pos
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_LEFT_CORNER:
			face.Eyes.Right.Left = pos
		case pb.FaceAnnotation_Landmark_LEFT_EYE_PUPIL:
			face.Eyes.Left.Pupil = pos
		case pb.FaceAnnotation_Landmark_RIGHT_EYE_PUPIL:
			face.Eyes.Right.Pupil = pos
		case pb.FaceAnnotation_Landmark_LEFT_EAR_TRAGION:
			face.Ears.Left = pos
		case pb.FaceAnnotation_Landmark_RIGHT_EAR_TRAGION:
			face.Ears.Right = pos
		case pb.FaceAnnotation_Landmark_FOREHEAD_GLABELLA:
			face.Forehead = pos
		case pb.FaceAnnotation_Landmark_CHIN_GNATHION:
			face.Chin.Center = pos
		case pb.FaceAnnotation_Landmark_CHIN_LEFT_GONION:
			face.Chin.Left = pos
		case pb.FaceAnnotation_Landmark_CHIN_RIGHT_GONION:
			face.Chin.Right = pos
		default:
			log.Printf("vision: ignoring unknown face annotation landmark %s", lm.Type)
		}
	}
}
