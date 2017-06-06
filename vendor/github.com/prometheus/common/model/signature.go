// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"sort"
)

// SeparatorByte is a byte that cannot occur in valid UTF-8 sequences and is
// used to separate label names, label values, and other strings from each other
// when calculating their combined hash value (aka signature aka fingerprint).
const SeparatorByte byte = 255

var (
	// cache the signature of an empty label set.
	emptyLabelSignature = hashNew()
)

// LabelsToSignature returns a quasi-unique signature (i.e., fingerprint) for a
// given label set. (Collisions are possible but unlikely if the number of label
// sets the function is applied to is small.)
func LabelsToSignature(labels map[string]string) uint64 {
	if len(labels) == 0 {
		return emptyLabelSignature
	}

	labelNames := make([]string, 0, len(labels))
	for labelName := range labels {
		labelNames = append(labelNames, labelName)
	}
	sort.Strings(labelNames)

	sum := hashNew()
	for _, labelName := range labelNames {
		sum = hashAdd(sum, labelName)
		sum = hashAddByte(sum, SeparatorByte)
		sum = hashAdd(sum, labels[labelName])
		sum = hashAddByte(sum, SeparatorByte)
	}
	return sum
}

// labelSetToFingerprint works exactly as LabelsToSignature but takes a LabelSet as
// parameter (rather than a label map) and returns a Fingerprint.
func labelSetToFingerprint(ls LabelSet) Fingerprint {
	if len(ls) == 0 {
		return Fingerprint(emptyLabelSignature)
	}

	labelNames := make(LabelNames, 0, len(ls))
	for labelName := range ls {
		labelNames = append(labelNames, labelName)
	}
	sort.Sort(labelNames)

	sum := hashNew()
	for _, labelName := range labelNames {
		sum = hashAdd(sum, string(labelName))
		sum = hashAddByte(sum, SeparatorByte)
		sum = hashAdd(sum, string(ls[labelName]))
		sum = hashAddByte(sum, SeparatorByte)
	}
	return Fingerprint(sum)
}

// labelSetToFastFingerprint works similar to labelSetToFingerprint but uses a
// faster and less allocation-heavy hash function, which is more susceptible to
// create hash collisions. Therefore, collision detection should be applied.
func labelSetToFastFingerprint(ls LabelSet) Fingerprint {
	if len(ls) == 0 {
		return Fingerprint(emptyLabelSignature)
	}

	var result uint64
	for labelName, labelValue := range ls {
		sum := hashNew()
		sum = hashAdd(sum, string(labelName))
		sum = hashAddByte(sum, SeparatorByte)
		sum = hashAdd(sum, string(labelValue))
		result ^= sum
	}
	return Fingerprint(result)
}

// SignatureForLabels works like LabelsToSignature but takes a Metric as
// parameter (rather than a label map) and only includes the labels with the
// specified LabelNames into the signature calculation. The labels passed in
// will be sorted by this function.
func SignatureForLabels(m Metric, labels ...LabelName) uint64 {
	if len(labels) == 0 {
		return emptyLabelSignature
	}

	sort.Sort(LabelNames(labels))

	sum := hashNew()
	for _, label := range labels {
		sum = hashAdd(sum, string(label))
		sum = hashAddByte(sum, SeparatorByte)
		sum = hashAdd(sum, string(m[label]))
		sum = hashAddByte(sum, SeparatorByte)
	}
	return sum
}

// SignatureWithoutLabels works like LabelsToSignature but takes a Metric as
// parameter (rather than a label map) and excludes the labels with any of the
// specified LabelNames from the signature calculation.
func SignatureWithoutLabels(m Metric, labels map[LabelName]struct{}) uint64 {
	if len(m) == 0 {
		return emptyLabelSignature
	}

	labelNames := make(LabelNames, 0, len(m))
	for labelName := range m {
		if _, exclude := labels[labelName]; !exclude {
			labelNames = append(labelNames, labelName)
		}
	}
	if len(labelNames) == 0 {
		return emptyLabelSignature
	}
	sort.Sort(labelNames)

	sum := hashNew()
	for _, labelName := range labelNames {
		sum = hashAdd(sum, string(labelName))
		sum = hashAddByte(sum, SeparatorByte)
		sum = hashAdd(sum, string(m[labelName]))
		sum = hashAddByte(sum, SeparatorByte)
	}
	return sum
}
