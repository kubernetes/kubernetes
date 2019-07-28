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
// Tables about what Buildifier can and cannot edit.
// Perhaps eventually this will be
// derived from the BUILD encyclopedia.

package tables

// IsLabelArg: a named argument to a rule call is considered to have a value
// that can be treated as a label or list of labels if the name
// is one of these names. There is a separate blacklist for
// rule-specific exceptions.
var IsLabelArg = map[string]bool{
	"app_target":         true,
	"appdir":             true,
	"base_package":       true,
	"build_deps":         true,
	"cc_deps":            true,
	"ccdeps":             true,
	"common_deps":        true,
	"compile_deps":       true,
	"compiler":           true,
	"data":               true,
	"default_visibility": true,
	"dep":                true,
	"deps":               true,
	"deps_java":          true,
	"dont_depend_on":     true,
	"env_deps":           true,
	"envscripts":         true,
	"exported_deps":      true,
	"exports":            true,
	"externs_list":       true,
	"files":              true,
	"globals":            true,
	"implementation":     true,
	"implements":         true,
	"includes":           true,
	"interface":          true,
	"jar":                true,
	"jars":               true,
	"javadeps":           true,
	"lib_deps":           true,
	"library":            true,
	"malloc":             true,
	"model":              true,
	"mods":               true,
	"module_deps":        true,
	"module_target":      true,
	"of":                 true,
	"plugins":            true,
	"proto_deps":         true,
	"proto_target":       true,
	"protos":             true,
	"resource":           true,
	"resources":          true,
	"runtime_deps":       true,
	"scope":              true,
	"shared_deps":        true,
	"similar_deps":       true,
	"source_jar":         true,
	"src":                true,
	"srcs":               true,
	"stripped_targets":   true,
	"suites":             true,
	"swigdeps":           true,
	"target":             true,
	"target_devices":     true,
	"target_platforms":   true,
	"template":           true,
	"test":               true,
	"tests":              true,
	"tests_deps":         true,
	"tool":               true,
	"tools":              true,
	"visibility":         true,
}

// LabelBlacklist is the list of call arguments that cannot be
// shortened, because they are not interpreted using the same
// rules as for other labels.
var LabelBlacklist = map[string]bool{
	// Shortening this can cause visibility checks to fail.
	"package_group.includes": true,
}

// By default, edit.types.IsList consults lang.TypeOf to determine if an arg is a list.
// You may override this using IsListArg. Specifying a name here overrides any value
// in lang.TypeOf.
var IsListArg = map[string]bool{}

// IsSortableListArg: a named argument to a rule call is considered to be a sortable list
// if the name is one of these names. There is a separate blacklist for
// rule-specific exceptions.
var IsSortableListArg = map[string]bool{
	"cc_deps":             true,
	"common_deps":         true,
	"compile_deps":        true,
	"configs":             true,
	"constraints":         true,
	"data":                true,
	"default_visibility":  true,
	"deps":                true,
	"deps_java":           true,
	"exported_deps":       true,
	"exports":             true,
	"filegroups":          true,
	"files":               true,
	"hdrs":                true,
	"imports":             true,
	"includes":            true,
	"inherits":            true,
	"javadeps":            true,
	"lib_deps":            true,
	"module_deps":         true,
	"out":                 true,
	"outs":                true,
	"packages":            true,
	"plugin_modules":      true,
	"proto_deps":          true,
	"protos":              true,
	"pubs":                true,
	"resources":           true,
	"runtime_deps":        true,
	"shared_deps":         true,
	"similar_deps":        true,
	"srcs":                true,
	"swigdeps":            true,
	"swig_includes":       true,
	"tags":                true,
	"tests":               true,
	"tools":               true,
	"to_start_extensions": true,
	"visibility":          true,
}

// SortableBlacklist records specific rule arguments that must not be reordered.
var SortableBlacklist = map[string]bool{
	"genrule.outs": true,
	"genrule.srcs": true,
}

// SortableWhitelist records specific rule arguments that are guaranteed
// to be reorderable, because bazel re-sorts the list itself after reading the BUILD file.
var SortableWhitelist = map[string]bool{
	"cc_inc_library.hdrs":      true,
	"cc_library.hdrs":          true,
	"java_library.srcs":        true,
	"java_library.resources":   true,
	"java_binary.srcs":         true,
	"java_binary.resources":    true,
	"java_test.srcs":           true,
	"java_test.resources":      true,
	"java_library.constraints": true,
	"java_import.constraints":  true,
}

// NamePriority maps an argument name to its sorting priority.
//
// NOTE(bazel-team): These are the old buildifier rules. It is likely that this table
// will change, perhaps swapping in a separate table for each call,
// derived from the order used in the Build Encyclopedia.
var NamePriority = map[string]int{
	"name":              -99,
	"gwt_name":          -98,
	"package_name":      -97,
	"visible_node_name": -96, // for boq_initial_css_modules and boq_jswire_test_suite
	"size":              -95,
	"timeout":           -94,
	"testonly":          -93,
	"src":               -92,
	"srcdir":            -91,
	"srcs":              -90,
	"out":               -89,
	"outs":              -88,
	"hdrs":              -87,
	"has_services":      -86, // before api versions, for proto
	"include":           -85, // before exclude, for glob
	"of":                -84, // for check_dependencies
	"baseline":          -83, // for searchbox_library
	// All others sort here, at 0.
	"destdir":        1,
	"exports":        2,
	"runtime_deps":   3,
	"deps":           4,
	"implementation": 5,
	"implements":     6,
	"alwayslink":     7,
}

var StripLabelLeadingSlashes = false

var ShortenAbsoluteLabelsToRelative = false

var FormatBzlFiles = false

// OverrideTables allows a user of the build package to override the special-case rules. The user-provided tables replace the built-in tables.
func OverrideTables(labelArg, blacklist, listArg, sortableListArg, sortBlacklist, sortWhitelist map[string]bool, namePriority map[string]int, stripLabelLeadingSlashes, shortenAbsoluteLabelsToRelative bool) {
	IsLabelArg = labelArg
	LabelBlacklist = blacklist
	IsListArg = listArg
	IsSortableListArg = sortableListArg
	SortableBlacklist = sortBlacklist
	SortableWhitelist = sortWhitelist
	NamePriority = namePriority
	StripLabelLeadingSlashes = stripLabelLeadingSlashes
	ShortenAbsoluteLabelsToRelative = shortenAbsoluteLabelsToRelative
}

// MergeTables allows a user of the build package to override the special-case rules. The user-provided tables are merged into the built-in tables.
func MergeTables(labelArg, blacklist, listArg, sortableListArg, sortBlacklist, sortWhitelist map[string]bool, namePriority map[string]int, stripLabelLeadingSlashes, shortenAbsoluteLabelsToRelative bool) {
	for k, v := range labelArg {
		IsLabelArg[k] = v
	}
	for k, v := range blacklist {
		LabelBlacklist[k] = v
	}
	for k, v := range listArg {
		IsListArg[k] = v
	}
	for k, v := range sortableListArg {
		IsSortableListArg[k] = v
	}
	for k, v := range sortBlacklist {
		SortableBlacklist[k] = v
	}
	for k, v := range sortWhitelist {
		SortableWhitelist[k] = v
	}
	for k, v := range namePriority {
		NamePriority[k] = v
	}
	StripLabelLeadingSlashes = stripLabelLeadingSlashes || StripLabelLeadingSlashes
	ShortenAbsoluteLabelsToRelative = shortenAbsoluteLabelsToRelative || ShortenAbsoluteLabelsToRelative
}
