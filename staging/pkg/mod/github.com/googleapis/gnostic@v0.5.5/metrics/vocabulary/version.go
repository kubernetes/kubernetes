// Copyright 2020 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package vocabulary

import (
	metrics "github.com/googleapis/gnostic/metrics"
)

// fillVersionProto takes a newer and older version of a vocabularies and utilizes the
// difference operation to find new and deleted terms. Those terms are used to create
// a new Version structure which is then returned.
func fillVersionProto(oldVersion, newVersion *metrics.Vocabulary, oldName, newName string) *metrics.Version {
	newTerms := Difference([]*metrics.Vocabulary{newVersion, oldVersion})
	deletedTerms := Difference([]*metrics.Vocabulary{oldVersion, newVersion})
	version := &metrics.Version{
		NewTerms:         newTerms,
		DeletedTerms:     deletedTerms,
		Name:             newName,
		NewTermCount:     int32(length(newTerms)),
		DeletedTermCount: int32(length(deletedTerms)),
	}
	return version
}

// Version implements the difference and union operation amongst a list of
// vocabularies that represent different versions of the same API. This
// function utilizes the VersionHistory proto struct, and creates a new version
// struct for each comparison between vocabularies.
func Version(v []*metrics.Vocabulary, versionNames []string, directory string) *metrics.VersionHistory {
	versions := make([]*metrics.Version, 0)
	for i := 0; i < len(v)-1; i++ {
		versions = append(versions, fillVersionProto(v[i], v[i+1], versionNames[i], versionNames[i+1]))
	}
	versionHistory := &metrics.VersionHistory{
		Versions: versions,
		Name:     directory,
	}
	return versionHistory
}
