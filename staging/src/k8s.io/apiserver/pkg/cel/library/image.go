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
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/go-containerregistry/pkg/name"

	apiservercel "k8s.io/apiserver/pkg/cel"
)

// Image provides a CEL function library extension for parsing image references.
//
// image
//
// Converts a string to an image reference or returns an error if the image is not a valid image reference. Refer
// to github.com/opencontainers/image-spec documentation for information on accepted image references.
//
//	image(<string>) <Image>
//
// Examples:
//
//	image('registry.k8s.io/kube-apiserver-arm64:latest') // returns an Image
//	image('nginx') // returns an Image
//	image('registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2') // returns an Image
//	image('registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2') // returns an Image
//	image('registry.k8s.io//kube-apiserver-arm64') // error
//	image(0) // error
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
//	isImage('registry.k8s.io/kube-apiserver-arm64:latest') // returns true
//	isImage('nginx') // returns true
//	isImage('registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2') // returns true
//	isImage('registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2') // returns true
//	isImage('registry.k8s.io//kube-apiserver-arm64') // returns false
//	isImage(0) // returns false
//
// <Image>.containsDigest:
//
// Returns true if the Image has a digest.
//
// Examples:
//
//	image("image("registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").containsDigest()").containsDigest() // returns true
//	image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").containsDigest()) // returns true
//	image("registry.k8s.io/kube-apiserver-arm64:latest").containsDigest() // returns false
//	image("registry.k8s.io/kube-apiserver-arm64").containsDigest() // returns false
//
// Image components:
//
//   - registry: Returns the registry component of the image reference. Defaults to index.docker.io if not present.
//
//   - repository: Returns the repository component of the image reference.
//
//   - tag: Returns the registry component of the image reference. Defaults to latest if both tag and digest are not present.
//
//   - digest: Returns the digest component of the image reference.
//
// Examples:
//
// image("kube-apiserver-arm64:testtag").registry() // returns "index.docker.io"
// image("registry.k8s.io/kube-apiserver-arm64:latest").registry() // returns "registry.k8s.io"
// image("registry.k8s.io/kube-apiserver-arm64").repository() // returns "kube-apiserver-arm64"
// image("registry.k8s.io/kube-apiserver-arm64").tag() // returns "latest"
// image("registry.k8s.io/kube-apiserver-arm64:testtag").tag() // returns "testtag"
// image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").digest()) // returns "sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"
// image("registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").digest()) // returns "sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"
//
// <Image>.identifier
//
// Returns the tag or digest of the image. identifer() returns the digest if both tag and digest are provided
//
// Examples:
//
// image("registry.k8s.io/kube-apiserver-arm64:testtag").identifier() // returns "testtag"
// image("registry.k8s.io/kube-apiserver-arm64@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").identifier() // returns "sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"
// image("registry.k8s.io/kube-apiserver-arm64:latest@sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2").identifier() // returns "sha256:6aefddb645ee6963afd681b1845c661d0ea4c3b20ab9db86d9e753b203d385f2"

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

	_, err := name.ParseReference(str)
	if err != nil {
		return types.Bool(false)
	}

	return types.Bool(true)
}

func stringToImage(arg ref.Val) ref.Val {
	str, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	v, err := name.ParseReference(str)
	if err != nil {
		return types.WrapErr(err)
	}

	return apiservercel.Image{ImageReference: ConvertToImageRef(v)}
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

func ConvertToImageRef(ref name.Reference) apiservercel.ImageReference {
	var img apiservercel.ImageReference
	img.Image = ref.String()
	img.Registry = ref.Context().RegistryStr()
	img.Repository = ref.Context().RepositoryStr()
	img.Identifier = ref.Identifier()

	if _, ok := ref.(name.Tag); ok {
		img.Tag = ref.Identifier()
	} else {
		img.Digest = ref.Identifier()
	}

	return img
}
