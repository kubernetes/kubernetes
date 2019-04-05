/*
Copyright 2016 Google Inc. All Rights Reserved.
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
// Type information for attributes.

package edit

import (
	buildpb "github.com/bazelbuild/buildtools/build_proto"
	"github.com/bazelbuild/buildtools/lang"
	"github.com/bazelbuild/buildtools/tables"
)

var typeOf = lang.TypeOf

// IsList returns true for all attributes whose type is a list.
func IsList(attr string) bool {
	overrideValue, isOverridden := tables.IsListArg[attr]
	if isOverridden {
		return overrideValue
	}
	// It stands to reason that a sortable list must be a list.
	isSortableList := tables.IsSortableListArg[attr]
	if isSortableList {
		return true
	}
	ty := typeOf[attr]
	return ty == buildpb.Attribute_STRING_LIST ||
		ty == buildpb.Attribute_LABEL_LIST ||
		ty == buildpb.Attribute_OUTPUT_LIST ||
		ty == buildpb.Attribute_FILESET_ENTRY_LIST ||
		ty == buildpb.Attribute_INTEGER_LIST ||
		ty == buildpb.Attribute_LICENSE ||
		ty == buildpb.Attribute_DISTRIBUTION_SET
}

// IsIntList returns true for all attributes whose type is an int list.
func IsIntList(attr string) bool {
	return typeOf[attr] == buildpb.Attribute_INTEGER_LIST
}

// IsString returns true for all attributes whose type is a string or a label.
func IsString(attr string) bool {
	ty := typeOf[attr]
	return ty == buildpb.Attribute_LABEL ||
		ty == buildpb.Attribute_STRING ||
		ty == buildpb.Attribute_OUTPUT
}

// IsStringDict returns true for all attributes whose type is a string dictionary.
func IsStringDict(attr string) bool {
	return typeOf[attr] == buildpb.Attribute_STRING_DICT
}

// ContainsLabels returns true for all attributes whose type is a label or a label list.
func ContainsLabels(attr string) bool {
	ty := typeOf[attr]
	return ty == buildpb.Attribute_LABEL_LIST ||
		ty == buildpb.Attribute_LABEL
}
