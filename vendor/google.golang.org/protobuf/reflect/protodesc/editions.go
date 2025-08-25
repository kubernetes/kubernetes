// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protodesc

import (
	"fmt"
	"os"
	"sync"

	"google.golang.org/protobuf/internal/editiondefaults"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/gofeaturespb"
)

var defaults = &descriptorpb.FeatureSetDefaults{}
var defaultsCacheMu sync.Mutex
var defaultsCache = make(map[filedesc.Edition]*descriptorpb.FeatureSet)

func init() {
	err := proto.Unmarshal(editiondefaults.Defaults, defaults)
	if err != nil {
		fmt.Fprintf(os.Stderr, "unmarshal editions defaults: %v\n", err)
		os.Exit(1)
	}
}

func fromEditionProto(epb descriptorpb.Edition) filedesc.Edition {
	return filedesc.Edition(epb)
}

func toEditionProto(ed filedesc.Edition) descriptorpb.Edition {
	switch ed {
	case filedesc.EditionUnknown:
		return descriptorpb.Edition_EDITION_UNKNOWN
	case filedesc.EditionProto2:
		return descriptorpb.Edition_EDITION_PROTO2
	case filedesc.EditionProto3:
		return descriptorpb.Edition_EDITION_PROTO3
	case filedesc.Edition2023:
		return descriptorpb.Edition_EDITION_2023
	case filedesc.Edition2024:
		return descriptorpb.Edition_EDITION_2024
	default:
		panic(fmt.Sprintf("unknown value for edition: %v", ed))
	}
}

func getFeatureSetFor(ed filedesc.Edition) *descriptorpb.FeatureSet {
	defaultsCacheMu.Lock()
	defer defaultsCacheMu.Unlock()
	if def, ok := defaultsCache[ed]; ok {
		return def
	}
	edpb := toEditionProto(ed)
	if defaults.GetMinimumEdition() > edpb || defaults.GetMaximumEdition() < edpb {
		// This should never happen protodesc.(FileOptions).New would fail when
		// initializing the file descriptor.
		// This most likely means the embedded defaults were not updated.
		fmt.Fprintf(os.Stderr, "internal error: unsupported edition %v (did you forget to update the embedded defaults (i.e. the bootstrap descriptor proto)?)\n", edpb)
		os.Exit(1)
	}
	fsed := defaults.GetDefaults()[0]
	// Using a linear search for now.
	// Editions are guaranteed to be sorted and thus we could use a binary search.
	// Given that there are only a handful of editions (with one more per year)
	// there is not much reason to use a binary search.
	for _, def := range defaults.GetDefaults() {
		if def.GetEdition() <= edpb {
			fsed = def
		} else {
			break
		}
	}
	fs := proto.Clone(fsed.GetFixedFeatures()).(*descriptorpb.FeatureSet)
	proto.Merge(fs, fsed.GetOverridableFeatures())
	defaultsCache[ed] = fs
	return fs
}

// mergeEditionFeatures merges the parent and child feature sets. This function
// should be used when initializing Go descriptors from descriptor protos which
// is why the parent is a filedesc.EditionsFeatures (Go representation) while
// the child is a descriptorproto.FeatureSet (protoc representation).
// Any feature set by the child overwrites what is set by the parent.
func mergeEditionFeatures(parentDesc protoreflect.Descriptor, child *descriptorpb.FeatureSet) filedesc.EditionFeatures {
	var parentFS filedesc.EditionFeatures
	switch p := parentDesc.(type) {
	case *filedesc.File:
		parentFS = p.L1.EditionFeatures
	case *filedesc.Message:
		parentFS = p.L1.EditionFeatures
	default:
		panic(fmt.Sprintf("unknown parent type %T", parentDesc))
	}
	if child == nil {
		return parentFS
	}
	if fp := child.FieldPresence; fp != nil {
		parentFS.IsFieldPresence = *fp == descriptorpb.FeatureSet_LEGACY_REQUIRED ||
			*fp == descriptorpb.FeatureSet_EXPLICIT
		parentFS.IsLegacyRequired = *fp == descriptorpb.FeatureSet_LEGACY_REQUIRED
	}
	if et := child.EnumType; et != nil {
		parentFS.IsOpenEnum = *et == descriptorpb.FeatureSet_OPEN
	}

	if rfe := child.RepeatedFieldEncoding; rfe != nil {
		parentFS.IsPacked = *rfe == descriptorpb.FeatureSet_PACKED
	}

	if utf8val := child.Utf8Validation; utf8val != nil {
		parentFS.IsUTF8Validated = *utf8val == descriptorpb.FeatureSet_VERIFY
	}

	if me := child.MessageEncoding; me != nil {
		parentFS.IsDelimitedEncoded = *me == descriptorpb.FeatureSet_DELIMITED
	}

	if jf := child.JsonFormat; jf != nil {
		parentFS.IsJSONCompliant = *jf == descriptorpb.FeatureSet_ALLOW
	}

	// We must not use proto.GetExtension(child, gofeaturespb.E_Go)
	// because that only works for messages we generated, but not for
	// dynamicpb messages. See golang/protobuf#1669.
	//
	// Further, we harden this code against adversarial inputs: a
	// service which accepts descriptors from a possibly malicious
	// source shouldn't crash.
	goFeatures := child.ProtoReflect().Get(gofeaturespb.E_Go.TypeDescriptor())
	if !goFeatures.IsValid() {
		return parentFS
	}
	gf, ok := goFeatures.Interface().(protoreflect.Message)
	if !ok {
		return parentFS
	}
	// gf.Interface() could be *dynamicpb.Message or *gofeaturespb.GoFeatures.
	fields := gf.Descriptor().Fields()

	if fd := fields.ByNumber(genid.GoFeatures_LegacyUnmarshalJsonEnum_field_number); fd != nil &&
		!fd.IsList() &&
		fd.Kind() == protoreflect.BoolKind &&
		gf.Has(fd) {
		parentFS.GenerateLegacyUnmarshalJSON = gf.Get(fd).Bool()
	}

	if fd := fields.ByNumber(genid.GoFeatures_StripEnumPrefix_field_number); fd != nil &&
		!fd.IsList() &&
		fd.Kind() == protoreflect.EnumKind &&
		gf.Has(fd) {
		parentFS.StripEnumPrefix = int(gf.Get(fd).Enum())
	}

	if fd := fields.ByNumber(genid.GoFeatures_ApiLevel_field_number); fd != nil &&
		!fd.IsList() &&
		fd.Kind() == protoreflect.EnumKind &&
		gf.Has(fd) {
		parentFS.APILevel = int(gf.Get(fd).Enum())
	}

	return parentFS
}

// initFileDescFromFeatureSet initializes editions related fields in fd based
// on fs. If fs is nil it is assumed to be an empty featureset and all fields
// will be initialized with the appropriate default. fd.L1.Edition must be set
// before calling this function.
func initFileDescFromFeatureSet(fd *filedesc.File, fs *descriptorpb.FeatureSet) {
	dfs := getFeatureSetFor(fd.L1.Edition)
	// initialize the featureset with the defaults
	fd.L1.EditionFeatures = mergeEditionFeatures(fd, dfs)
	// overwrite any options explicitly specified
	fd.L1.EditionFeatures = mergeEditionFeatures(fd, fs)
}
