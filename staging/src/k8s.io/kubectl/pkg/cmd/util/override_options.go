/*
Copyright 2021 The Kubernetes Authors.

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

package util

import (
	"fmt"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
)

type OverrideType string

const (
	// OverrideTypeJSON will use an RFC6902 JSON Patch to alter the generated output
	OverrideTypeJSON OverrideType = "json"

	// OverrideTypeMerge will use an RFC7396 JSON Merge Patch to alter the generated output
	OverrideTypeMerge OverrideType = "merge"

	// OverrideTypeStrategic will use a Strategic Merge Patch to alter the generated output
	OverrideTypeStrategic OverrideType = "strategic"
)

const DefaultOverrideType = OverrideTypeMerge

type OverrideOptions struct {
	Overrides    string
	OverrideType OverrideType
}

func (o *OverrideOptions) AddOverrideFlags(cmd *cobra.Command) {
	cmd.Flags().StringVar(&o.Overrides, "overrides", "", i18n.T("An inline JSON override for the generated object. If this is non-empty, it is used to override the generated object. Requires that the object supply a valid apiVersion field."))
	cmd.Flags().StringVar((*string)(&o.OverrideType), "override-type", string(DefaultOverrideType), fmt.Sprintf("The method used to override the generated object: %s, %s, or %s.", OverrideTypeJSON, OverrideTypeMerge, OverrideTypeStrategic))
}

func (o *OverrideOptions) NewOverrider(dataStruct runtime.Object) *Overrider {
	return &Overrider{
		Options:    o,
		DataStruct: dataStruct,
	}
}

type Overrider struct {
	Options    *OverrideOptions
	DataStruct runtime.Object
}

func (o *Overrider) Apply(obj runtime.Object) (runtime.Object, error) {
	if len(o.Options.Overrides) == 0 {
		return obj, nil
	}

	codec := runtime.NewCodec(scheme.DefaultJSONEncoder(), scheme.Codecs.UniversalDecoder(scheme.Scheme.PrioritizedVersionsAllGroups()...))

	var overrideType OverrideType
	if len(o.Options.OverrideType) == 0 {
		overrideType = DefaultOverrideType
	} else {
		overrideType = o.Options.OverrideType
	}

	switch overrideType {
	case OverrideTypeJSON:
		return JSONPatch(codec, obj, o.Options.Overrides)
	case OverrideTypeMerge:
		return Merge(codec, obj, o.Options.Overrides)
	case OverrideTypeStrategic:
		return StrategicMerge(codec, obj, o.Options.Overrides, o.DataStruct)
	default:
		return nil, fmt.Errorf("invalid override type: %v", overrideType)
	}
}
