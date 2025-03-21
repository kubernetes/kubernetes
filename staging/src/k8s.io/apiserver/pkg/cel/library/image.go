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

package library

import (
	"fmt"
	//  Import the crypto sha256 algorithm for the image parser to work
	_ "crypto/sha256"
	//  Import the crypto/sha512 algorithm for the image parser to work with 384 and 512 sha hashes
	_ "crypto/sha512"

	"github.com/distribution/reference"
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"

	apiservercel "k8s.io/apiserver/pkg/cel"
)

const (
	// DefaultTag is the tag name that will be used if no tag provided
	DefaultTag = "latest"
)

// Image provides a CEL function library extension for parsing image references.
//
// image
//
// Converts a string to an image reference or returns an error if the image is not a valid image reference. Refer
// to github.com/opencontainers/image-spec documentation for information on accepted image references.
// An optional "normalize" argument can be passed to enable normalization. Normalization parses the image
// into a normalized image. An example normalized image is "docker.io/library/ubuntu" for image "ubuntu".
//
//	image(<string>) <Image>
//
// Examples:
//
//	image('registry.k8s.io/kube-apiserver-arm64:latest') // Returns an Image
//	image('nginx') // Returns an Image
//	image('registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2') // Returns an Image
//	image('registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2') // Returns an Image
//	image('registry.k8s.io//kube-apiserver-arm64') // error
//	image(0) // error
///	image('ubuntu') // Returns Image for "ubuntu".
///	image('ubuntu', true) // Applies normalization and returns components of "docker.io/library/ubuntu".
///	image('ubuntu', false) // Returns Image for "ubuntu".
//	image('registry.k8s.io/kube-apiserver-arm64:latest', true) // Returns an Image
//
// isImage
//
// Returns true if a string is a valid image. isImage returns true if and
// only if image reference parsing does not result in error.
//
//	isImage(<string>) <bool>
//
// Examples:
//
//	isImage('registry.k8s.io/kube-apiserver-arm64:latest') // Returns true
//	isImage('nginx') // Returns true
//	isImage('registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2') // Returns true
//	isImage('registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2') // Returns true
//	isImage('registry.k8s.io//kube-apiserver-arm64') // Returns false
//	isImage(0) // Returns false
//
// <Image>.containsDigest:
//
// Returns true if the Image has a digest.
//
// Examples:
//
//	image("image("registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").containsDigest()").containsDigest() // Returns true
//	image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").containsDigest()) // Returns true
//	image("registry.k8s.io/kube-apiserver-arm64:latest").containsDigest() // Returns false
//	image("registry.k8s.io/kube-apiserver-arm64").containsDigest() // Returns false
//
// Image components:
//
//   - registry: Returns the registry component of the image reference.
//
//   - repository: Returns the repository component of the image reference.
//
//   - tag: Returns the registry component of the image reference. Defaults to latest if both tag and digest are not present.
//
//   - digest: Returns the digest component of the image reference.
//
// Examples:
//
// image("kube-apiserver-arm64:testtag").registry() // Returns ""
// image("registry.k8s.io/kube-apiserver-arm64:latest").registry() // Returns "registry.k8s.io"
// image("registry.k8s.io/kube-apiserver-arm64").repository() // Returns "kube-apiserver-arm64"
// image("registry.k8s.io/kube-apiserver-arm64").tag() // Returns "latest"
// image("registry.k8s.io/kube-apiserver-arm64:testtag").tag() // Returns "testtag"
// image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").digest()) // Returns "sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"
// image("registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").digest()) // Returns "sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"
//
// <Image>.identifier
//
// Returns the tag or digest of the image. identifer() returns the digest if both tag and digest are provided
//
// Examples:
//
// image("registry.k8s.io/kube-apiserver-arm64:testtag").identifier() // Returns "testtag"
// image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").identifier() // Returns "sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"
// image("registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").identifier() // Returns "sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"

func Image() cel.EnvOption {
	return cel.Lib(imageLib)
}

var imageLib = &imageLibType{}

type imageLibType struct{}

func (*imageLibType) LibraryName() string {
	return "kubernetes.Image"
}

func (*imageLibType) Types() []*cel.Type {
	return []*cel.Type{apiservercel.ImageType}
}

func (*imageLibType) declarations() map[string][]cel.FunctionOpt {
	return map[string][]cel.FunctionOpt{
		"image": {
			cel.Overload("string_to_image", []*cel.Type{cel.StringType}, apiservercel.ImageType, cel.UnaryBinding((stringToImage))),
			cel.Overload("string_bool_to_image", []*cel.Type{cel.StringType, cel.BoolType}, apiservercel.ImageType, cel.BinaryBinding((stringToImageNormalized))),
		},
		"isImage": {
			cel.Overload("is_image_string", []*cel.Type{cel.StringType}, cel.BoolType, cel.UnaryBinding(isImage)),
		},
		"containsDigest": {
			cel.MemberOverload("image_contains_digest", []*cel.Type{apiservercel.ImageType}, cel.BoolType, cel.UnaryBinding(imageContainsDigest)),
		},
		"registry": {
			cel.MemberOverload("image_registry", []*cel.Type{apiservercel.ImageType}, cel.StringType, cel.UnaryBinding(imageRegistry)),
		},
		"repository": {
			cel.MemberOverload("image_repository", []*cel.Type{apiservercel.ImageType}, cel.StringType, cel.UnaryBinding(imageRepository)),
		},
		"identifier": {
			cel.MemberOverload("image_identifier", []*cel.Type{apiservercel.ImageType}, cel.StringType, cel.UnaryBinding(imageIdentifier)),
		},
		"tag": {
			cel.MemberOverload("image_tag", []*cel.Type{apiservercel.ImageType}, cel.StringType, cel.UnaryBinding(imageTag)),
		},
		"digest": {
			cel.MemberOverload("image_digest", []*cel.Type{apiservercel.ImageType}, cel.StringType, cel.UnaryBinding(imageDigest)),
		},
	}
}

func (i *imageLibType) CompileOptions() []cel.EnvOption {
	imageLibraryDecls := i.declarations()
	options := make([]cel.EnvOption, 0, len(imageLibraryDecls))
	for name, overloads := range imageLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	return options
}

func (*imageLibType) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func isImage(arg ref.Val) ref.Val {
	str, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	_, err := parseImageRef(str)
	if err != nil {
		return types.Bool(false)
	}

	return types.Bool(true)
}

func stringToImage(arg ref.Val) ref.Val {
	return stringToImageNormalized(arg, types.Bool(false))
}

func stringToImageNormalized(arg ref.Val, normalizeArg ref.Val) ref.Val {
	str, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	normalize, ok := normalizeArg.Value().(bool)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	var v apiservercel.ImageReference
	var err error
	if normalize {
		v, err = parseImageRefNormalized(str)
	} else {
		v, err = parseImageRef(str)
	}
	if err != nil {
		return types.WrapErr(err)
	}

	return apiservercel.Image{ImageReference: v}
}

func imageContainsDigest(arg ref.Val) ref.Val {
	v, ok := arg.Value().(apiservercel.ImageReference)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.Bool(len(v.Digest) != 0)
}

func imageRegistry(arg ref.Val) ref.Val {
	v, ok := arg.Value().(apiservercel.ImageReference)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(v.Registry)
}

func imageRepository(arg ref.Val) ref.Val {
	v, ok := arg.Value().(apiservercel.ImageReference)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(v.Repository)
}

func imageIdentifier(arg ref.Val) ref.Val {
	v, ok := arg.Value().(apiservercel.ImageReference)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(v.Identifier)
}

func imageTag(arg ref.Val) ref.Val {
	v, ok := arg.Value().(apiservercel.ImageReference)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(v.Tag)
}

func imageDigest(arg ref.Val) ref.Val {
	v, ok := arg.Value().(apiservercel.ImageReference)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(v.Digest)
}

func namedToRef(ref reference.Named, normalize bool) apiservercel.ImageReference {
	var img apiservercel.ImageReference
	img.Registry = reference.Domain(ref)
	img.Repository = reference.Path(ref)
	img.Image = ref.String()

	if tagged, ok := ref.(reference.Tagged); ok {
		img.Tag = tagged.Tag()
	}
	if digested, ok := ref.(reference.Digested); ok {
		img.Digest = digested.Digest().String()
	}

	if normalize && len(img.Tag) == 0 && len(img.Digest) == 0 {
		img.Tag = DefaultTag
		img.Image += ":latest"
	}

	if len(img.Digest) > 0 {
		img.Identifier = img.Digest
	} else {
		img.Identifier = img.Tag
	}

	return img
}

func parseImageRef(image string) (apiservercel.ImageReference, error) {
	ref, err := reference.Parse(image)
	if err != nil {
		return apiservercel.ImageReference{}, fmt.Errorf("failed to parse image: %s, err: %w", image, err)
	}

	if named, ok := ref.(reference.Named); ok {
		return namedToRef(named, false), nil
	} else {
		return apiservercel.ImageReference{}, fmt.Errorf("failed to parse image reference %s", ref.String())
	}
}

func parseImageRefNormalized(image string) (apiservercel.ImageReference, error) {
	ref, err := reference.ParseNormalizedNamed(image)
	if err != nil {
		return apiservercel.ImageReference{}, fmt.Errorf("failed to parse image: %s, err: %w", image, err)
	}

	return namedToRef(ref, true), nil
}
