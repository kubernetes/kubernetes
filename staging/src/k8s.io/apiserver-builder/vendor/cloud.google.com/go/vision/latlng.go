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
	pb "google.golang.org/genproto/googleapis/cloud/vision/v1"
	llpb "google.golang.org/genproto/googleapis/type/latlng"
)

// A LatLng is a point on the Earth's surface, represented with a latitude and longitude.
type LatLng struct {
	// Lat is the latitude in degrees. It must be in the range [-90.0, +90.0].
	Lat float64
	// Lng is the longitude in degrees. It must be in the range [-180.0, +180.0].
	Lng float64
}

func (l LatLng) toProto() *llpb.LatLng {
	return &llpb.LatLng{
		Latitude:  l.Lat,
		Longitude: l.Lng,
	}
}

func latLngFromProto(ll *llpb.LatLng) LatLng {
	return LatLng{
		Lat: ll.Latitude,
		Lng: ll.Longitude,
	}
}

// A LatLngRect is a rectangular area on the Earth's surface, represented by a
// minimum and maximum latitude and longitude.
type LatLngRect struct {
	Min, Max LatLng
}

func (r *LatLngRect) toProto() *pb.LatLongRect {
	if r == nil {
		return nil
	}
	return &pb.LatLongRect{
		MinLatLng: r.Min.toProto(),
		MaxLatLng: r.Max.toProto(),
	}
}
