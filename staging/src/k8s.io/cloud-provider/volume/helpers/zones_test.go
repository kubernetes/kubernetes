/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package helpers

import (
	"hash/fnv"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestZonesToSet(t *testing.T) {
	functionUnderTest := "ZonesToSet"
	// First part: want an error
	sliceOfZones := []string{"", ",", "us-east-1a, , us-east-1d", ", us-west-1b", "us-west-2b,"}
	for _, zones := range sliceOfZones {
		if got, err := ZonesToSet(zones); err == nil {
			t.Errorf("%v(%v) returned (%v), want (%v)", functionUnderTest, zones, got, "an error")
		}
	}

	// Second part: want no error
	tests := []struct {
		zones string
		want  sets.String
	}{
		{
			zones: "us-east-1a",
			want:  sets.String{"us-east-1a": sets.Empty{}},
		},
		{
			zones: "us-east-1a, us-west-2a",
			want: sets.String{
				"us-east-1a": sets.Empty{},
				"us-west-2a": sets.Empty{},
			},
		},
	}
	for _, tt := range tests {
		if got, err := ZonesToSet(tt.zones); err != nil || !got.Equal(tt.want) {
			t.Errorf("%v(%v) returned (%v), want (%v)", functionUnderTest, tt.zones, got, tt.want)
		}
	}
}

func TestChooseZonesForVolume(t *testing.T) {
	checkFnv32(t, "henley", 1180403676)
	// 1180403676 mod 3 == 0, so the offset from "henley" is 0, which makes it easier to verify this by inspection

	// A few others
	checkFnv32(t, "henley-", 2652299129)
	checkFnv32(t, "henley-a", 1459735322)
	checkFnv32(t, "", 2166136261)

	tests := []struct {
		Zones      sets.String
		VolumeName string
		NumZones   uint32
		Expected   sets.String
	}{
		// Test for PVC names that don't have a dash
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */, "b"),
		},
		// Tests for PVC names that end in - number, but don't look like statefulset PVCs
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-0",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-0",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-1",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 1 == 1 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-1",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 1 + 1(startingIndex) == 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-2",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley") + 2 == 2 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-2",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("henley") + 2 + 2(startingIndex) == 4 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-3",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") + 3 == 3 === 0 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-3",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") + 3 + 3(startingIndex) == 6 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-4",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 4 == 4 === 1 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-4",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 4 + 4(startingIndex) == 8 */, "a"),
		},
		// Tests for PVC names that are edge cases
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley-") = 2652299129 === 2 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley-") = 2652299129 === 2 mod 3 = 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-a",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley-a") = 1459735322 === 2 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "henley-a",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley-a") = 1459735322 === 2 mod 3 = 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium--1",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("") + 1 == 2166136261 + 1 === 2 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium--1",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("") + 1 + 1(startingIndex) == 2166136261 + 1 + 1 === 3 mod 3 = 0 */, "b"),
		},
		// Tests for PVC names for simple StatefulSet cases
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-1",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 1 == 1 */),
		},
		// Tests for PVC names for simple StatefulSet cases
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-1",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 1 + 1(startingIndex) == 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "loud-henley-1",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 1 == 1 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "loud-henley-1",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 1 + 1(startingIndex) == 2 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "quiet-henley-2",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley") + 2 == 2 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "quiet-henley-2",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("henley") + 2 + 2(startingIndex) == 4 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-2",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley") + 2 == 2 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-2",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("henley") + 2 + 2(startingIndex) == 4 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-3",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") + 3 == 3 === 0 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-3",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") + 3 + 3(startingIndex) == 6 === 6 mod 3 = 0 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-4",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 4 == 4 === 1 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley-4",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 4 + 4(startingIndex) == 8 === 2 mod 3 */, "a"),
		},
		// Tests for statefulsets (or claims) with dashes in the names
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-alpha-henley-2",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("henley") + 2 == 2 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-alpha-henley-2",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("henley") + 2 + 2(startingIndex) == 4 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-beta-henley-3",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("henley") + 3 == 3 === 0 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-beta-henley-3",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") + 3 + 3(startingIndex) == 6 === 0 mod 3 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-gamma-henley-4",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("henley") + 4 == 4 === 1 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-gamma-henley-4",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") + 4 + 4(startingIndex) == 8 === 2 mod 3 */, "a"),
		},
		// Tests for statefulsets name ending in -
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--2",
			NumZones:   1,
			Expected:   sets.NewString("a" /* hash("") + 2 == 0 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--2",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("") + 2 + 2(startingIndex) == 2 mod 3 */, "a"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--3",
			NumZones:   1,
			Expected:   sets.NewString("b" /* hash("") + 3 == 1 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--3",
			NumZones:   2,
			Expected:   sets.NewString("b" /* hash("") + 3 + 3(startingIndex) == 1 mod 3 */, "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--4",
			NumZones:   1,
			Expected:   sets.NewString("c" /* hash("") + 4 == 2 mod 3 */),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--4",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("") + 4 + 4(startingIndex) == 0 mod 3 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--4",
			NumZones:   3,
			Expected:   sets.NewString("c" /* hash("") + 4 == 2 mod 3 */, "a", "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c"),
			VolumeName: "medium-henley--4",
			NumZones:   4,
			Expected:   sets.NewString("c" /* hash("") + 4 + 9(startingIndex) == 2 mod 3 */, "a", "b", "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-0",
			NumZones:   2,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */, "b"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-1",
			NumZones:   2,
			Expected:   sets.NewString("c" /* hash("henley") == 0 + 2 */, "d"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-2",
			NumZones:   2,
			Expected:   sets.NewString("e" /* hash("henley") == 0 + 2 + 2(startingIndex) */, "f"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-3",
			NumZones:   2,
			Expected:   sets.NewString("g" /* hash("henley") == 0 + 2 + 4(startingIndex) */, "h"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-0",
			NumZones:   3,
			Expected:   sets.NewString("a" /* hash("henley") == 0 */, "b", "c"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-1",
			NumZones:   3,
			Expected:   sets.NewString("d" /* hash("henley") == 0 + 1 + 2(startingIndex) */, "e", "f"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-2",
			NumZones:   3,
			Expected:   sets.NewString("g" /* hash("henley") == 0 + 2 + 4(startingIndex) */, "h", "i"),
		},
		{
			Zones:      sets.NewString("a", "b", "c", "d", "e", "f", "g", "h", "i"),
			VolumeName: "henley-3",
			NumZones:   3,
			Expected:   sets.NewString("a" /* hash("henley") == 0 + 3 + 6(startingIndex) */, "b", "c"),
		},
		// Test for no zones
		{
			Zones:      sets.NewString(),
			VolumeName: "henley-1",
			Expected:   sets.NewString(),
		},
		{
			Zones:      nil,
			VolumeName: "henley-2",
			Expected:   sets.NewString(),
		},
	}

	for _, test := range tests {
		actual := ChooseZonesForVolume(test.Zones, test.VolumeName, test.NumZones)

		if !actual.Equal(test.Expected) {
			t.Errorf("Test %v failed, expected zone %#v, actual %#v", test, test.Expected, actual)
		}
	}
}

func TestSelectZoneForVolume(t *testing.T) {

	nodeWithGATopologyLabels := &v1.Node{}
	nodeWithGATopologyLabels.Labels = map[string]string{v1.LabelTopologyZone: "zoneX"}

	nodeWithBetaTopologyLabels := &v1.Node{}
	nodeWithBetaTopologyLabels.Labels = map[string]string{v1.LabelFailureDomainBetaZone: "zoneY"}

	nodeWithNoLabels := &v1.Node{}

	tests := []struct {
		// Parameters passed by test to SelectZoneForVolume
		Name              string
		ZonePresent       bool
		Zone              string
		ZonesPresent      bool
		Zones             string
		ZonesWithNodes    string
		Node              *v1.Node
		AllowedTopologies []v1.TopologySelectorTerm
		// Expectations around returned zone from SelectZoneForVolume
		Reject             bool   // expect error due to validation failing
		ExpectSpecificZone bool   // expect returned zone to specifically match a single zone (rather than one from a set)
		ExpectedZone       string // single zone that should perfectly match returned zone (requires ExpectSpecificZone to be true)
		ExpectedZones      string // set of zones one of whose members should match returned zone (requires ExpectSpecificZone to be false)
	}{
		// NEGATIVE TESTS

		// Zone and Zones are both specified [Fail]
		// [1] Node irrelevant
		// [2] Zone and Zones parameters presents
		// [3] AllowedTopologies irrelevant
		{
			Name:         "Nil_Node_with_Zone_Zones_parameters_present",
			ZonePresent:  true,
			Zone:         "zoneX",
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			Reject:       true,
		},

		// Node has no zone labels [Fail]
		// [1] Node with no zone labels
		// [2] Zone/Zones parameter irrelevant
		// [3] AllowedTopologies irrelevant
		{
			Name:   "Node_with_no_Zone_labels",
			Node:   nodeWithNoLabels,
			Reject: true,
		},

		// Node with Zone labels as well as Zone parameter specified [Fail]
		// [1] Node with zone labels
		// [2] Zone parameter specified
		// [3] AllowedTopologies irrelevant
		{
			Name:        "Node_with_Zone_labels_and_Zone_parameter_present",
			Node:        nodeWithGATopologyLabels,
			ZonePresent: true,
			Zone:        "zoneX",
			Reject:      true,
		},

		// Node with Zone labels as well as Zones parameter specified [Fail]
		// [1] Node with zone labels
		// [2] Zones parameter specified
		// [3] AllowedTopologies irrelevant
		{
			Name:         "Node_with_Zone_labels_and_Zones_parameter_present",
			Node:         nodeWithGATopologyLabels,
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			Reject:       true,
		},

		// Zone parameter as well as AllowedTopologies specified [Fail]
		// [1] nil Node
		// [2] Zone parameter specified
		// [3] AllowedTopologies specified
		{
			Name:        "Nil_Node_and_Zone_parameter_and_Allowed_Topology_term",
			Node:        nil,
			ZonePresent: true,
			Zone:        "zoneX",
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
			},
			Reject: true,
		},

		// Zones parameter as well as AllowedTopologies specified [Fail]
		// [1] nil Node
		// [2] Zones parameter specified
		// [3] AllowedTopologies specified
		{
			Name:         "Nil_Node_and_Zones_parameter_and_Allowed_Topology_term",
			Node:         nil,
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
			},
			Reject: true,
		},

		// Key specified in AllowedTopologies is not LabelTopologyZone [Fail]
		// [1] nil Node
		// [2] no Zone/Zones parameter
		// [3] AllowedTopologies with invalid key specified
		{
			Name: "Nil_Node_and_Invalid_Allowed_Topology_Key",
			Node: nil,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    "invalid_key",
							Values: []string{"zoneX"},
						},
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneY"},
						},
					},
				},
			},
			Reject: true,
		},

		// AllowedTopologies without keys specifying LabelTopologyZone [Fail]
		// [1] nil Node
		// [2] no Zone/Zones parameter
		// [3] Invalid AllowedTopologies
		{
			Name: "Nil_Node_and_Invalid_AllowedTopologies",
			Node: nil,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{},
				},
			},
			Reject: true,
		},

		// POSITIVE TESTS WITH VolumeScheduling ENABLED

		// Select zone from active zones [Pass]
		// [1] nil Node
		// [2] no Zone parameter specified
		// [3] no AllowedTopologies
		{
			Name:           "Nil_Node_and_No_Zone_Zones_parameter_and_no_Allowed_topologies_and_VolumeScheduling_enabled",
			Node:           nil,
			ZonesWithNodes: "zoneX,zoneY",
			Reject:         false,
			ExpectedZones:  "zoneX,zoneY",
		},

		// Select zone from single zone parameter [Pass]
		// [1] nil Node
		// [2] Zone parameter specified
		// [3] no AllowedTopology specified
		{
			Name:               "Nil_Node_and_Zone_parameter_present_and_VolumeScheduling_enabled",
			ZonePresent:        true,
			Zone:               "zoneX",
			Node:               nil,
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
		},

		// Select zone from zones parameter [Pass]
		// [1] nil Node
		// [2] Zones parameter specified
		// [3] no AllowedTopology
		{
			Name:          "Nil_Node_and_Zones_parameter_present_and_VolumeScheduling_enabled",
			ZonesPresent:  true,
			Zones:         "zoneX,zoneY",
			Node:          nil,
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// Select zone from node ga label [Pass]
		// [1] Node with zone labels
		// [2] no zone/zones parameters
		// [3] no AllowedTopology
		{
			Name:               "Node_with_GA_Zone_labels_and_VolumeScheduling_enabled",
			Node:               nodeWithGATopologyLabels,
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
		},

		// Select zone from node beta label [Pass]
		// [1] Node with zone labels
		// [2] no zone/zones parameters
		// [3] no AllowedTopology
		{
			Name:               "Node_with_Beta_Zone_labels_and_VolumeScheduling_enabled",
			Node:               nodeWithBetaTopologyLabels,
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneY",
		},

		// Select zone from node label [Pass]
		// [1] Node with zone labels
		// [2] no Zone/Zones parameters
		// [3] AllowedTopology with single term with multiple values specified (ignored)
		{
			Name: "Node_with_Zone_labels_and_Multiple_Allowed_Topology_values_and_VolumeScheduling_enabled",
			Node: nodeWithGATopologyLabels,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneZ", "zoneY"},
						},
					},
				},
			},
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
		},

		// Select Zone from AllowedTopologies [Pass]
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with single term with multiple values specified
		{
			Name: "Nil_Node_with_Multiple_Allowed_Topology_values_and_VolumeScheduling_enabled",
			Node: nil,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX", "zoneY"},
						},
					},
				},
			},
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// Select Zone from AllowedTopologies using Beta labels [Pass]
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with single term with multiple values specified
		{
			Name: "Nil_Node_with_Multiple_Allowed_Topology_values_Beta_label_and_VolumeScheduling_enabled",
			Node: nil,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelFailureDomainBetaZone,
							Values: []string{"zoneX", "zoneY"},
						},
					},
				},
			},
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// Select zone from AllowedTopologies [Pass]
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with multiple terms specified
		{
			Name: "Nil_Node_and_Multiple_Allowed_Topology_terms_and_VolumeScheduling_enabled",
			Node: nil,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneY"},
						},
					},
				},
			},
			Reject:        false,
			ExpectedZones: "zoneX,zoneY",
		},

		// Select Zone from AllowedTopologies [Pass]
		// Note: Dual replica with same AllowedTopologies will fail: Nil_Node_and_Single_Allowed_Topology_term_value_and_Dual_replicas
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with single term and value specified
		{
			Name: "Nil_Node_and_Single_Allowed_Topology_term_value_and_VolumeScheduling_enabled",
			Node: nil,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
			},
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			var zonesParameter, zonesWithNodes sets.String
			var err error

			if test.Zones != "" {
				zonesParameter, err = ZonesToSet(test.Zones)
				if err != nil {
					t.Fatalf("Could not convert Zones to a set: %s. This is a test error %s", test.Zones, test.Name)
				}
			}

			if test.ZonesWithNodes != "" {
				zonesWithNodes, err = ZonesToSet(test.ZonesWithNodes)
				if err != nil {
					t.Fatalf("Could not convert specified ZonesWithNodes to a set: %s. This is a test error %s", test.ZonesWithNodes, test.Name)
				}
			}

			zone, err := SelectZoneForVolume(test.ZonePresent, test.ZonesPresent, test.Zone, zonesParameter, zonesWithNodes, test.Node, test.AllowedTopologies, test.Name)

			if test.Reject && err == nil {
				t.Errorf("Unexpected zone from SelectZoneForVolume for %s", zone)
			}

			if !test.Reject {
				if err != nil {
					t.Errorf("Unexpected error from SelectZoneForVolume for %s; Error: %v", test.Name, err)
				}

				if test.ExpectSpecificZone {
					if zone != test.ExpectedZone {
						t.Errorf("Expected zone %v does not match obtained zone %v for %s", test.ExpectedZone, zone, test.Name)
					}
				}

				if test.ExpectedZones != "" {
					expectedZones, err := ZonesToSet(test.ExpectedZones)
					if err != nil {
						t.Fatalf("Could not convert ExpectedZones to a set: %s. This is a test error", test.ExpectedZones)
					}
					if !expectedZones.Has(zone) {
						t.Errorf("Obtained zone %s not member of expectedZones %s", zone, expectedZones)
					}
				}
			}
		})
	}
}

func TestSelectZonesForVolume(t *testing.T) {

	nodeWithZoneLabels := &v1.Node{}
	nodeWithZoneLabels.Labels = map[string]string{v1.LabelTopologyZone: "zoneX"}

	nodeWithNoLabels := &v1.Node{}

	tests := []struct {
		// Parameters passed by test to SelectZonesForVolume
		Name              string
		ReplicaCount      uint32
		ZonePresent       bool
		Zone              string
		ZonesPresent      bool
		Zones             string
		ZonesWithNodes    string
		Node              *v1.Node
		AllowedTopologies []v1.TopologySelectorTerm
		// Expectations around returned zones from SelectZonesForVolume
		Reject              bool   // expect error due to validation failing
		ExpectSpecificZones bool   // expect set of returned zones to be equal to ExpectedZones (rather than subset of ExpectedZones)
		ExpectedZones       string // set of zones that is a superset of returned zones or equal to returned zones (if ExpectSpecificZones is set)
		ExpectSpecificZone  bool   // expect set of returned zones to include ExpectedZone as a member
		ExpectedZone        string // zone that should be a member of returned zones
	}{
		// NEGATIVE TESTS

		// Zone and Zones are both specified [Fail]
		// [1] Node irrelevant
		// [2] Zone and Zones parameters presents
		// [3] AllowedTopologies irrelevant
		// [4] ReplicaCount irrelevant
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Nil_Node_with_Zone_Zones_parameters_present",
			ZonePresent:  true,
			Zone:         "zoneX",
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			Reject:       true,
		},

		// Node has no zone labels [Fail]
		// [1] Node with no zone labels
		// [2] Zone/Zones parameter irrelevant
		// [3] AllowedTopologies irrelevant
		// [4] ReplicaCount irrelevant
		// [5] ZonesWithNodes irrelevant
		{
			Name:   "Node_with_no_Zone_labels",
			Node:   nodeWithNoLabels,
			Reject: true,
		},

		// Node with Zone labels as well as Zone parameter specified [Fail]
		// [1] Node with zone labels
		// [2] Zone parameter specified
		// [3] AllowedTopologies irrelevant
		// [4] ReplicaCount irrelevant
		// [5] ZonesWithNodes irrelevant
		{
			Name:        "Node_with_Zone_labels_and_Zone_parameter_present",
			Node:        nodeWithZoneLabels,
			ZonePresent: true,
			Zone:        "zoneX",
			Reject:      true,
		},

		// Node with Zone labels as well as Zones parameter specified [Fail]
		// [1] Node with zone labels
		// [2] Zones parameter specified
		// [3] AllowedTopologies irrelevant
		// [4] ReplicaCount irrelevant
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Node_with_Zone_labels_and_Zones_parameter_present",
			Node:         nodeWithZoneLabels,
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			Reject:       true,
		},

		// Zone parameter as well as AllowedTopologies specified [Fail]
		// [1] nil Node
		// [2] Zone parameter specified
		// [3] AllowedTopologies specified
		// [4] ReplicaCount irrelevant
		// [5] ZonesWithNodes irrelevant
		{
			Name:        "Nil_Node_and_Zone_parameter_and_Allowed_Topology_term",
			Node:        nil,
			ZonePresent: true,
			Zone:        "zoneX",
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
			},
			Reject: true,
		},

		// Zones parameter as well as AllowedTopologies specified [Fail]
		// [1] nil Node
		// [2] Zones parameter specified
		// [3] AllowedTopologies specified
		// [4] ReplicaCount irrelevant
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Nil_Node_and_Zones_parameter_and_Allowed_Topology_term",
			Node:         nil,
			ZonesPresent: true,
			Zones:        "zoneX,zoneY",
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
			},
			Reject: true,
		},

		// Key specified in AllowedTopologies is not LabelTopologyZone [Fail]
		// [1] nil Node
		// [2] no Zone/Zones parameter
		// [3] AllowedTopologies with invalid key specified
		// [4] ReplicaCount irrelevant
		// [5] ZonesWithNodes irrelevant
		{
			Name: "Nil_Node_and_Invalid_Allowed_Topology_Key",
			Node: nil,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    "invalid_key",
							Values: []string{"zoneX"},
						},
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneY"},
						},
					},
				},
			},
			Reject: true,
		},

		// AllowedTopologies without keys specifying LabelTopologyZone [Fail]
		// [1] nil Node
		// [2] no Zone/Zones parameter
		// [3] Invalid AllowedTopologies
		// [4] ReplicaCount irrelevant
		// [5] ZonesWithNodes irrelevant
		{
			Name: "Nil_Node_and_Invalid_AllowedTopologies",
			Node: nil,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{},
				},
			},
			Reject: true,
		},

		// Zone specified with ReplicaCount > 1 [Fail]
		// [1] nil Node
		// [2] Zone/Zones parameter specified
		// [3] AllowedTopologies irrelevant
		// [4] ReplicaCount > 1
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Zone_and_Replicas_mismatched",
			Node:         nil,
			ZonePresent:  true,
			Zone:         "zoneX",
			ReplicaCount: 2,
			Reject:       true,
		},

		// Not enough zones in Zones parameter to satisfy ReplicaCount [Fail]
		// [1] nil Node
		// [2] Zone/Zones parameter specified
		// [3] AllowedTopologies irrelevant
		// [4] ReplicaCount > zones
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Zones_and_Replicas_mismatched",
			Node:         nil,
			ZonesPresent: true,
			Zones:        "zoneX",
			ReplicaCount: 2,
			Reject:       true,
		},

		// Not enough zones in ZonesWithNodes to satisfy ReplicaCount [Fail]
		// [1] nil Node
		// [2] no Zone/Zones parameter specified
		// [3] AllowedTopologies irrelevant
		// [4] ReplicaCount > ZonesWithNodes
		// [5] ZonesWithNodes specified but not enough
		{
			Name:           "Zones_and_Replicas_mismatched",
			Node:           nil,
			ZonesWithNodes: "zoneX",
			ReplicaCount:   2,
			Reject:         true,
		},

		// Not enough zones in AllowedTopologies to satisfy ReplicaCount [Fail]
		// [1] nil Node
		// [2] Zone/Zones parameter specified
		// [3] Invalid AllowedTopologies
		// [4] ReplicaCount > zones
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "AllowedTopologies_and_Replicas_mismatched",
			Node:         nil,
			ReplicaCount: 2,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
			},
			Reject: true,
		},

		// POSITIVE TESTS

		// Select zones from zones parameter [Pass]
		// [1] nil Node (Node irrelevant)
		// [2] Zones parameter specified
		// [3] no AllowedTopologies
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:          "Zones_parameter_Dual_replica_Superset",
			ZonesPresent:  true,
			Zones:         "zoneW,zoneX,zoneY,zoneZ",
			ReplicaCount:  2,
			Reject:        false,
			ExpectedZones: "zoneW,zoneX,zoneY,zoneZ",
		},

		// Select zones from zones parameter [Pass]
		// [1] nil Node (Node irrelevant)
		// [2] Zones parameter specified
		// [3] no AllowedTopologies
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:                "Zones_parameter_Dual_replica_Match",
			ZonesPresent:        true,
			Zones:               "zoneX,zoneY",
			ReplicaCount:        2,
			Reject:              false,
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneX,zoneY",
		},

		// Select zones from active zones with nodes [Pass]
		// [1] nil Node (Node irrelevant)
		// [2] no Zone parameter
		// [3] no AllowedTopologies
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes specified
		{
			Name:           "Active_zones_Nil_node_Dual_replica_Superset",
			ZonesWithNodes: "zoneW,zoneX,zoneY,zoneZ",
			ReplicaCount:   2,
			Reject:         false,
			ExpectedZones:  "zoneW,zoneX,zoneY,zoneZ",
		},

		// Select zones from active zones [Pass]
		// [1] nil Node (Node irrelevant)
		// [2] no Zone[s] parameter
		// [3] no AllowedTopologies
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes specified
		{
			Name:                "Active_zones_Nil_node_Dual_replica_Match",
			ZonesWithNodes:      "zoneW,zoneX",
			ReplicaCount:        2,
			Reject:              false,
			ExpectSpecificZone:  true,
			ExpectedZone:        "zoneX",
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneW,zoneX",
		},

		// Select zones from node label and active zones [Pass]
		// [1] Node with zone labels
		// [2] no Zone parameter
		// [3] no AllowedTopologies
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:               "Active_zones_Node_with_Zone_labels_Dual_replica_Superset",
			Node:               nodeWithZoneLabels,
			ZonesWithNodes:     "zoneW,zoneX,zoneY,zoneZ",
			ReplicaCount:       2,
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
			ExpectedZones:      "zoneW,zoneX,zoneY,zoneZ",
		},

		// Select zones from node label and active zones [Pass]
		// [1] Node with zone labels
		// [2] no Zone[s] parameter
		// [3] no AllowedTopologies
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:                "Active_zones_Node_with_Zone_labels_Dual_replica_Match",
			Node:                nodeWithZoneLabels,
			ZonesWithNodes:      "zoneW,zoneX",
			ReplicaCount:        2,
			Reject:              false,
			ExpectSpecificZone:  true,
			ExpectedZone:        "zoneX",
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneW,zoneX",
		},

		// Select zones from node label and AllowedTopologies [Pass]
		// [1] Node with zone labels
		// [2] no Zone/Zones parameters
		// [3] AllowedTopologies with single term with multiple values specified
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		// Note: the test Name suffix is used as the pvcname and it's suffix is important to influence ChooseZonesForVolume
		// to NOT pick zoneX from AllowedTopologies if zoneFromNode is incorrectly set or not set at all.
		{
			Name:         "Node_with_Zone_labels_and_Multiple_Allowed_Topology_values_Superset-1",
			Node:         nodeWithZoneLabels,
			ReplicaCount: 2,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneV", "zoneW", "zoneX", "zoneY", "zoneZ"},
						},
					},
				},
			},
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
			ExpectedZones:      "zoneV,zoneW,zoneX,zoneY,zoneZ",
		},

		// Select zones from node label and AllowedTopologies [Pass]
		// [1] Node with zone labels
		// [2] no Zone/Zones parameters
		// [3] AllowedTopologies with single term with multiple values specified
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Node_with_Zone_labels_and_Multiple_Allowed_Topology_values_Match",
			Node:         nodeWithZoneLabels,
			ReplicaCount: 2,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX", "zoneY"},
						},
					},
				},
			},
			Reject:              false,
			ExpectSpecificZone:  true,
			ExpectedZone:        "zoneX",
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneX,zoneY",
		},

		// Select Zones from AllowedTopologies [Pass]
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with single term with multiple values specified
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Nil_Node_with_Multiple_Allowed_Topology_values_Superset",
			Node:         nil,
			ReplicaCount: 2,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX", "zoneY", "zoneZ"},
						},
					},
				},
			},
			Reject:        false,
			ExpectedZones: "zoneX,zoneY,zoneZ",
		},

		// Select Zones from AllowedTopologies [Pass]
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with single term with multiple values specified
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Nil_Node_with_Multiple_Allowed_Topology_values_Match",
			Node:         nil,
			ReplicaCount: 2,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX", "zoneY"},
						},
					},
				},
			},
			Reject:              false,
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneX,zoneY",
		},

		// Select zones from node label and AllowedTopologies [Pass]
		// [1] Node with zone labels
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with multiple terms specified
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		// Note: the test Name is used as the pvcname and it's hash is important to influence ChooseZonesForVolume
		// to NOT pick zoneX from AllowedTopologies of 5 zones. If zoneFromNode (zoneX) is incorrectly set or
		// not set at all in SelectZonesForVolume it will be detected.
		{
			Name:         "Node_with_Zone_labels_and_Multiple_Allowed_Topology_terms_Superset-2",
			Node:         nodeWithZoneLabels,
			ReplicaCount: 2,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneV"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneW"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneY"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneZ"},
						},
					},
				},
			},
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
			ExpectedZones:      "zoneV,zoneW,zoneX,zoneY,zoneZ",
		},

		// Select zones from node label and AllowedTopologies [Pass]
		// [1] Node with zone labels
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with multiple terms specified
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Node_with_Zone_labels_and_Multiple_Allowed_Topology_terms_Match",
			Node:         nodeWithZoneLabels,
			ReplicaCount: 2,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneY"},
						},
					},
				},
			},
			Reject:              false,
			ExpectSpecificZone:  true,
			ExpectedZone:        "zoneX",
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneX,zoneY",
		},

		// Select zones from AllowedTopologies [Pass]
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with multiple terms specified
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Nil_Node_and_Multiple_Allowed_Topology_terms_Superset",
			Node:         nil,
			ReplicaCount: 2,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneY"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneZ"},
						},
					},
				},
			},
			Reject:        false,
			ExpectedZones: "zoneX,zoneY,zoneZ",
		},

		// Select zones from AllowedTopologies [Pass]
		// [1] nil Node
		// [2] no Zone/Zones parametes specified
		// [3] AllowedTopologies with multiple terms specified
		// [4] ReplicaCount specified
		// [5] ZonesWithNodes irrelevant
		{
			Name:         "Nil_Node_and_Multiple_Allowed_Topology_terms_Match",
			Node:         nil,
			ReplicaCount: 2,
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneX"},
						},
					},
				},
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    v1.LabelTopologyZone,
							Values: []string{"zoneY"},
						},
					},
				},
			},
			Reject:              false,
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneX,zoneY",
		},
	}

	for _, test := range tests {
		var zonesParameter, zonesWithNodes sets.String
		var err error

		if test.Zones != "" {
			zonesParameter, err = ZonesToSet(test.Zones)
			if err != nil {
				t.Errorf("Could not convert Zones to a set: %s. This is a test error %s", test.Zones, test.Name)
				continue
			}
		}

		if test.ZonesWithNodes != "" {
			zonesWithNodes, err = ZonesToSet(test.ZonesWithNodes)
			if err != nil {
				t.Errorf("Could not convert specified ZonesWithNodes to a set: %s. This is a test error %s", test.ZonesWithNodes, test.Name)
				continue
			}
		}

		zones, err := SelectZonesForVolume(test.ZonePresent, test.ZonesPresent, test.Zone, zonesParameter, zonesWithNodes, test.Node, test.AllowedTopologies, test.Name, test.ReplicaCount)

		if test.Reject && err == nil {
			t.Errorf("Unexpected zones from SelectZonesForVolume in %s. Zones: %v", test.Name, zones)
			continue
		}

		if !test.Reject {
			if err != nil {
				t.Errorf("Unexpected error from SelectZonesForVolume in %s; Error: %v", test.Name, err)
				continue
			}

			if uint32(zones.Len()) != test.ReplicaCount {
				t.Errorf("Number of elements in returned zones %v does not equal replica count %d in %s", zones, test.ReplicaCount, test.Name)
				continue
			}

			expectedZones, err := ZonesToSet(test.ExpectedZones)
			if err != nil {
				t.Errorf("Could not convert ExpectedZones to a set: %v. This is a test error in %s", test.ExpectedZones, test.Name)
				continue
			}

			if test.ExpectSpecificZones && !zones.Equal(expectedZones) {
				t.Errorf("Expected zones %v does not match obtained zones %v in %s", expectedZones, zones, test.Name)
				continue
			}

			if test.ExpectSpecificZone && !zones.Has(test.ExpectedZone) {
				t.Errorf("Expected zone %s not found in obtained zones %v in %s", test.ExpectedZone, zones, test.Name)
				continue
			}

			if !expectedZones.IsSuperset(zones) {
				t.Errorf("Obtained zones %v not subset of expectedZones %v in %s", zones, expectedZones, test.Name)
			}
		}
	}
}

func TestChooseZonesForVolumeIncludingZone(t *testing.T) {
	tests := []struct {
		// Parameters passed by test to chooseZonesForVolumeIncludingZone
		Name          string
		ReplicaCount  uint32
		Zones         string
		ZoneToInclude string
		// Expectations around returned zones from chooseZonesForVolumeIncludingZone
		Reject              bool   // expect error due to validation failing
		ExpectSpecificZones bool   // expect set of returned zones to be equal to ExpectedZones (rather than subset of ExpectedZones)
		ExpectedZones       string // set of zones that is a superset of returned zones or equal to returned zones (if ExpectSpecificZones is set)
		ExpectSpecificZone  bool   // expect set of returned zones to include ExpectedZone as a member
		ExpectedZone        string // zone that should be a member of returned zones
	}{
		// NEGATIVE TESTS

		// Not enough zones specified to fit ReplicaCount [Fail]
		{
			Name:         "Too_Few_Zones",
			ReplicaCount: 2,
			Zones:        "zoneX",
			Reject:       true,
		},

		// Invalid ReplicaCount [Fail]
		{
			Name:         "Zero_Replica_Count",
			ReplicaCount: 0,
			Zones:        "zoneY,zoneZ",
			Reject:       true,
		},

		// Invalid ZoneToInclude [Fail]
		{
			Name:          "ZoneToInclude_Not_In_Zones_Single_replica_Dual_zones",
			ReplicaCount:  1,
			ZoneToInclude: "zoneX",
			Zones:         "zoneY,zoneZ",
			Reject:        true,
		},

		// Invalid ZoneToInclude [Fail]
		{
			Name:          "ZoneToInclude_Not_In_Zones_Single_replica_Single_zone",
			ReplicaCount:  1,
			ZoneToInclude: "zoneX",
			Zones:         "zoneY",
			Reject:        true,
		},

		// Invalid ZoneToInclude [Fail]
		{
			Name:          "ZoneToInclude_Not_In_Zones_Dual_replica_Multiple_zones",
			ReplicaCount:  2,
			ZoneToInclude: "zoneX",
			Zones:         "zoneY,zoneZ,zoneW",
			Reject:        true,
		},

		// POSITIVE TESTS

		// Pick any one zone from Zones
		{
			Name:          "No_zones_to_include_and_Single_replica_and_Superset",
			ReplicaCount:  1,
			Zones:         "zoneX,zoneY,zoneZ",
			Reject:        false,
			ExpectedZones: "zoneX,zoneY,zoneZ",
		},

		// Pick any two zones from Zones
		{
			Name:          "No_zones_to_include_and_Dual_replicas_and_Superset",
			ReplicaCount:  2,
			Zones:         "zoneW,zoneX,zoneY,zoneZ",
			Reject:        false,
			ExpectedZones: "zoneW,zoneX,zoneY,zoneZ",
		},

		// Pick the two zones from Zones
		{
			Name:                "No_zones_to_include_and_Dual_replicas_and_Match",
			ReplicaCount:        2,
			Zones:               "zoneX,zoneY",
			Reject:              false,
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneX,zoneY",
		},

		// Pick one zone from ZoneToInclude (other zones ignored)
		{
			Name:                "Include_zone_and_Single_replica_and_Match",
			ReplicaCount:        1,
			Zones:               "zoneW,zoneX,zoneY,zoneZ",
			ZoneToInclude:       "zoneX",
			Reject:              false,
			ExpectSpecificZone:  true,
			ExpectedZone:        "zoneX",
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneX",
		},

		// Pick one zone from ZoneToInclude (other zones ignored)
		{
			Name:                "Include_zone_and_single_replica_and_Match",
			ReplicaCount:        1,
			Zones:               "zoneX",
			ZoneToInclude:       "zoneX",
			Reject:              false,
			ExpectSpecificZone:  true,
			ExpectedZone:        "zoneX",
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneX",
		},

		// Pick one zone from Zones and the other from ZoneToInclude
		{
			Name:               "Include_zone_and_dual_replicas_and_Superset",
			ReplicaCount:       2,
			Zones:              "zoneW,zoneX,zoneY,zoneZ",
			ZoneToInclude:      "zoneX",
			Reject:             false,
			ExpectSpecificZone: true,
			ExpectedZone:       "zoneX",
			ExpectedZones:      "zoneW,zoneX,zoneY,zoneZ",
		},

		// Pick one zone from Zones and the other from ZoneToInclude
		{
			Name:                "Include_zone_and_dual_replicas_and_Match",
			ReplicaCount:        2,
			Zones:               "zoneX,zoneY",
			ZoneToInclude:       "zoneX",
			Reject:              false,
			ExpectSpecificZone:  true,
			ExpectedZone:        "zoneX",
			ExpectSpecificZones: true,
			ExpectedZones:       "zoneX,zoneY",
		},
	}
	for _, test := range tests {
		zonesParameter, err := ZonesToSet(test.Zones)
		if err != nil {
			t.Errorf("Could not convert Zones to a set: %s. This is a test error %s", test.Zones, test.Name)
			continue
		}

		zones, err := chooseZonesForVolumeIncludingZone(zonesParameter, test.Name, test.ZoneToInclude, test.ReplicaCount)
		if test.Reject && err == nil {
			t.Errorf("Unexpected zones from chooseZonesForVolumeIncludingZone in %s. Zones: %v", test.Name, zones)
			continue
		}
		if !test.Reject {
			if err != nil {
				t.Errorf("Unexpected error from chooseZonesForVolumeIncludingZone in %s; Error: %v", test.Name, err)
				continue
			}

			if uint32(zones.Len()) != test.ReplicaCount {
				t.Errorf("Number of elements in returned zones %v does not equal replica count %d in %s", zones, test.ReplicaCount, test.Name)
				continue
			}

			expectedZones, err := ZonesToSet(test.ExpectedZones)
			if err != nil {
				t.Errorf("Could not convert ExpectedZones to a set: %v. This is a test error in %s", test.ExpectedZones, test.Name)
				continue
			}

			if test.ExpectSpecificZones && !zones.Equal(expectedZones) {
				t.Errorf("Expected zones %v does not match obtained zones %v in %s", expectedZones, zones, test.Name)
				continue
			}

			if test.ExpectSpecificZone && !zones.Has(test.ExpectedZone) {
				t.Errorf("Expected zone %s not found in obtained zones %v in %s", test.ExpectedZone, zones, test.Name)
				continue
			}

			if !expectedZones.IsSuperset(zones) {
				t.Errorf("Obtained zones %v not subset of expectedZones %v in %s", zones, expectedZones, test.Name)
			}
		}
	}
}

func checkFnv32(t *testing.T, s string, expected uint32) {
	h := fnv.New32()
	h.Write([]byte(s))
	h.Sum32()

	if h.Sum32() != expected {
		t.Fatalf("hash of %q was %v, expected %v", s, h.Sum32(), expected)
	}
}
