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
	"image"

	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

// Annotations contains all the annotations performed by the API on a single image.
// A nil field indicates either that the corresponding feature was not requested,
// or that annotation failed for that feature.
type Annotations struct {
	// Faces holds the results of face detection.
	Faces []*FaceAnnotation
	// Landmarks holds the results of landmark detection.
	Landmarks []*EntityAnnotation
	// Logos holds the results of logo detection.
	Logos []*EntityAnnotation
	// Labels holds the results of label detection.
	Labels []*EntityAnnotation
	// Texts holds the results of text detection.
	Texts []*EntityAnnotation
	// SafeSearch holds the results of safe-search detection.
	SafeSearch *SafeSearchAnnotation
	// ImageProps contains properties of the annotated image.
	ImageProps *ImageProps

	// If non-nil, then one or more of the attempted annotations failed.
	// Non-nil annotations are guaranteed to be correct, even if Error is
	// non-nil.
	Error error
}

func annotationsFromProto(res *pb.AnnotateImageResponse) *Annotations {
	as := &Annotations{}
	for _, a := range res.FaceAnnotations {
		as.Faces = append(as.Faces, faceAnnotationFromProto(a))
	}
	for _, a := range res.LandmarkAnnotations {
		as.Landmarks = append(as.Landmarks, entityAnnotationFromProto(a))
	}
	for _, a := range res.LogoAnnotations {
		as.Logos = append(as.Logos, entityAnnotationFromProto(a))
	}
	for _, a := range res.LabelAnnotations {
		as.Labels = append(as.Labels, entityAnnotationFromProto(a))
	}
	for _, a := range res.TextAnnotations {
		as.Texts = append(as.Texts, entityAnnotationFromProto(a))
	}
	as.SafeSearch = safeSearchAnnotationFromProto(res.SafeSearchAnnotation)
	as.ImageProps = imagePropertiesFromProto(res.ImagePropertiesAnnotation)
	if res.Error != nil {
		// res.Error is a google.rpc.Status. Convert to a Go error. Use a gRPC
		// error because it preserves the code as a separate field.
		// TODO(jba): preserve the details field.
		as.Error = grpc.Errorf(codes.Code(res.Error.Code), "%s", res.Error.Message)
	}
	return as
}

// A FaceAnnotation describes the results of face detection on an image.
type FaceAnnotation struct {
	// BoundingPoly is the bounding polygon around the face. The coordinates of
	// the bounding box are in the original image's scale, as returned in
	// ImageParams. The bounding box is computed to "frame" the face in
	// accordance with human expectations. It is based on the landmarker
	// results. Note that one or more x and/or y coordinates may not be
	// generated in the BoundingPoly (the polygon will be unbounded) if only a
	// partial face appears in the image to be annotated.
	BoundingPoly []image.Point

	// FDBoundingPoly is tighter than BoundingPoly, and
	// encloses only the skin part of the face. Typically, it is used to
	// eliminate the face from any image analysis that detects the "amount of
	// skin" visible in an image. It is not based on the landmarker results, only
	// on the initial face detection, hence the fd (face detection) prefix.
	FDBoundingPoly []image.Point

	// Landmarks are detected face landmarks.
	Face FaceLandmarks

	// RollAngle indicates the amount of clockwise/anti-clockwise rotation of
	// the face relative to the image vertical, about the axis perpendicular to
	// the face. Range [-180,180].
	RollAngle float32

	// PanAngle is the yaw angle: the leftward/rightward angle that the face is
	// pointing, relative to the vertical plane perpendicular to the image. Range
	// [-180,180].
	PanAngle float32

	// TiltAngle is the pitch angle: the upwards/downwards angle that the face is
	// pointing relative to the image's horizontal plane. Range [-180,180].
	TiltAngle float32

	// DetectionConfidence is the detection confidence. The range is [0, 1].
	DetectionConfidence float32

	// LandmarkingConfidence is the face landmarking confidence. The range is [0, 1].
	LandmarkingConfidence float32

	// Likelihoods expresses the likelihood of various aspects of the face.
	Likelihoods *FaceLikelihoods
}

func faceAnnotationFromProto(pfa *pb.FaceAnnotation) *FaceAnnotation {
	fa := &FaceAnnotation{
		BoundingPoly:          boundingPolyFromProto(pfa.BoundingPoly),
		FDBoundingPoly:        boundingPolyFromProto(pfa.FdBoundingPoly),
		RollAngle:             pfa.RollAngle,
		PanAngle:              pfa.PanAngle,
		TiltAngle:             pfa.TiltAngle,
		DetectionConfidence:   pfa.DetectionConfidence,
		LandmarkingConfidence: pfa.LandmarkingConfidence,
		Likelihoods: &FaceLikelihoods{
			Joy:          Likelihood(pfa.JoyLikelihood),
			Sorrow:       Likelihood(pfa.SorrowLikelihood),
			Anger:        Likelihood(pfa.AngerLikelihood),
			Surprise:     Likelihood(pfa.SurpriseLikelihood),
			UnderExposed: Likelihood(pfa.UnderExposedLikelihood),
			Blurred:      Likelihood(pfa.BlurredLikelihood),
			Headwear:     Likelihood(pfa.HeadwearLikelihood),
		},
	}
	populateFaceLandmarks(pfa.Landmarks, &fa.Face)
	return fa
}

// An EntityAnnotation describes the results of a landmark, label, logo or text
// detection on an image.
type EntityAnnotation struct {
	// ID is an opaque entity ID. Some IDs might be available in Knowledge Graph(KG).
	// For more details on KG please see:
	// https://developers.google.com/knowledge-graph/
	ID string

	// Locale is the language code for the locale in which the entity textual
	// description (next field) is expressed.
	Locale string

	// Description is the entity textual description, expressed in the language of Locale.
	Description string

	// Score is the overall score of the result. Range [0, 1].
	Score float32

	// Confidence is the accuracy of the entity detection in an image.
	// For example, for an image containing the Eiffel Tower, this field represents
	// the confidence that there is a tower in the query image. Range [0, 1].
	Confidence float32

	// Topicality is the relevancy of the ICA (Image Content Annotation) label to the
	// image. For example, the relevancy of 'tower' to an image containing
	// 'Eiffel Tower' is likely higher than an image containing a distant towering
	// building, though the confidence that there is a tower may be the same.
	// Range [0, 1].
	Topicality float32

	// BoundingPoly is the image region to which this entity belongs. Not filled currently
	// for label detection. For text detection, BoundingPolys
	// are produced for the entire text detected in an image region, followed by
	// BoundingPolys for each word within the detected text.
	BoundingPoly []image.Point

	// Locations contains the location information for the detected entity.
	// Multiple LatLng structs can be present since one location may indicate the
	// location of the scene in the query image, and another the location of the
	// place where the query image was taken. Location information is usually
	// present for landmarks.
	Locations []LatLng

	// Properties are additional optional Property fields.
	// For example a different kind of score or string that qualifies the entity.
	Properties []Property
}

func entityAnnotationFromProto(e *pb.EntityAnnotation) *EntityAnnotation {
	var locs []LatLng
	for _, li := range e.Locations {
		locs = append(locs, latLngFromProto(li.LatLng))
	}
	var props []Property
	for _, p := range e.Properties {
		props = append(props, propertyFromProto(p))
	}
	return &EntityAnnotation{
		ID:           e.Mid,
		Locale:       e.Locale,
		Description:  e.Description,
		Score:        e.Score,
		Confidence:   e.Confidence,
		Topicality:   e.Topicality,
		BoundingPoly: boundingPolyFromProto(e.BoundingPoly),
		Locations:    locs,
		Properties:   props,
	}
}

// SafeSearchAnnotation describes the results of a SafeSearch detection on an image.
type SafeSearchAnnotation struct {
	// Adult is the likelihood that the image contains adult content.
	Adult Likelihood

	// Spoof is the likelihood that an obvious modification was made to the
	// image's canonical version to make it appear funny or offensive.
	Spoof Likelihood

	// Medical is the likelihood that this is a medical image.
	Medical Likelihood

	// Violence is the likelihood that this image represents violence.
	Violence Likelihood
}

func safeSearchAnnotationFromProto(s *pb.SafeSearchAnnotation) *SafeSearchAnnotation {
	if s == nil {
		return nil
	}
	return &SafeSearchAnnotation{
		Adult:    Likelihood(s.Adult),
		Spoof:    Likelihood(s.Spoof),
		Medical:  Likelihood(s.Medical),
		Violence: Likelihood(s.Violence),
	}
}

// ImageProps describes properties of the image itself, like the dominant colors.
type ImageProps struct {
	// DominantColors describes the dominant colors of the image.
	DominantColors []*ColorInfo
}

func imagePropertiesFromProto(ip *pb.ImageProperties) *ImageProps {
	if ip == nil || ip.DominantColors == nil {
		return nil
	}
	var cinfos []*ColorInfo
	for _, ci := range ip.DominantColors.Colors {
		cinfos = append(cinfos, colorInfoFromProto(ci))
	}
	return &ImageProps{DominantColors: cinfos}
}
