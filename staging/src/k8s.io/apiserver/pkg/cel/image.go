/*
Copyright 2025 The Kubernetes Authors.

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

package cel

import (
	"fmt"
	"reflect"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

var (
	ImageType = cel.ObjectType("kubernetes.Image")
)

type ImageReference struct {
	Image      string `json:"image,omitempty"`
	Registry   string `json:"registry,omitempty"`
	Repository string `json:"repository,omitempty"`
	Identifier string `json:"identifier,omitempty"`
	Tag        string `json:"tag,omitempty"`
	Digest     string `json:"digest,omitempty"`
}

func (i ImageReference) String() string {
	return i.Image
}

type Image struct {
	ImageReference
}

func (v Image) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	if reflect.TypeOf(v.ImageReference).AssignableTo(typeDesc) {
		return v.ImageReference, nil
	}
	if reflect.TypeOf("").AssignableTo(typeDesc) {
		return v.ImageReference.String(), nil
	}
	return nil, fmt.Errorf("type conversion error from 'Image' to '%v'", typeDesc)
}

func (v Image) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case ImageType:
		return v
	case types.TypeType:
		return ImageType
	default:
		return types.NewErr("type conversion error from '%s' to '%s'", ImageType, typeVal)
	}
}

func (v Image) Equal(other ref.Val) ref.Val {
	img, ok := other.(Image)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	return types.Bool(reflect.DeepEqual(v.ImageReference, img.ImageReference))
}

func (v Image) Type() ref.Type {
	return ImageType
}

func (v Image) Value() interface{} {
	return v.ImageReference
}
