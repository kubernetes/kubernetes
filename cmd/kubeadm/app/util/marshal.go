/*
Copyright 2017 The Kubernetes Authors.

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
	"bufio"
	"bytes"
	"io"

	"github.com/pkg/errors"
	"sigs.k8s.io/yaml"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// MarshalToYaml marshals an object into yaml.
func MarshalToYaml(obj runtime.Object, gv schema.GroupVersion) ([]byte, error) {
	return MarshalToYamlForCodecs(obj, gv, clientsetscheme.Codecs)
}

// MarshalToYamlForCodecs marshals an object into yaml using the specified codec
// TODO: Is specifying the gv really needed here?
// TODO: Can we support json out of the box easily here?
func MarshalToYamlForCodecs(obj runtime.Object, gv schema.GroupVersion, codecs serializer.CodecFactory) ([]byte, error) {
	mediaType := "application/yaml"
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return []byte{}, errors.Errorf("unsupported media type %q", mediaType)
	}

	encoder := codecs.EncoderForVersion(info.Serializer, gv)
	return runtime.Encode(encoder, obj)
}

// UnmarshalFromYaml unmarshals yaml into an object.
func UnmarshalFromYaml(buffer []byte, gv schema.GroupVersion) (runtime.Object, error) {
	return UnmarshalFromYamlForCodecs(buffer, gv, clientsetscheme.Codecs)
}

// UnmarshalFromYamlForCodecs unmarshals yaml into an object using the specified codec
// TODO: Is specifying the gv really needed here?
// TODO: Can we support json out of the box easily here?
func UnmarshalFromYamlForCodecs(buffer []byte, gv schema.GroupVersion, codecs serializer.CodecFactory) (runtime.Object, error) {
	mediaType := "application/yaml"
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, errors.Errorf("unsupported media type %q", mediaType)
	}

	decoder := codecs.DecoderToVersion(info.Serializer, gv)
	return runtime.Decode(decoder, buffer)
}

// SplitYAMLDocuments reads the YAML bytes per-document, unmarshals the TypeMeta information from each document
// and returns a map between the GroupVersionKind of the document and the document bytes
func SplitYAMLDocuments(yamlBytes []byte) (map[schema.GroupVersionKind][]byte, error) {
	gvkmap := map[schema.GroupVersionKind][]byte{}
	knownKinds := map[string]bool{}
	errs := []error{}
	buf := bytes.NewBuffer(yamlBytes)
	reader := utilyaml.NewYAMLReader(bufio.NewReader(buf))
	for {
		typeMetaInfo := runtime.TypeMeta{}
		// Read one YAML document at a time, until io.EOF is returned
		b, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		if len(b) == 0 {
			break
		}
		// Deserialize the TypeMeta information of this byte slice
		if err := yaml.Unmarshal(b, &typeMetaInfo); err != nil {
			return nil, err
		}
		// Require TypeMeta information to be present
		if len(typeMetaInfo.APIVersion) == 0 || len(typeMetaInfo.Kind) == 0 {
			errs = append(errs, errors.New("invalid configuration: kind and apiVersion is mandatory information that needs to be specified in all YAML documents"))
			continue
		}
		// Check whether the kind has been registered before. If it has, throw an error
		if known := knownKinds[typeMetaInfo.Kind]; known {
			errs = append(errs, errors.Errorf("invalid configuration: kind %q is specified twice in YAML file", typeMetaInfo.Kind))
			continue
		}
		knownKinds[typeMetaInfo.Kind] = true

		// Build a GroupVersionKind object from the deserialized TypeMeta object
		gv, err := schema.ParseGroupVersion(typeMetaInfo.APIVersion)
		if err != nil {
			errs = append(errs, errors.Wrap(err, "unable to parse apiVersion"))
			continue
		}
		gvk := gv.WithKind(typeMetaInfo.Kind)

		// Save the mapping between the gvk and the bytes that object consists of
		gvkmap[gvk] = b
	}
	if err := errorsutil.NewAggregate(errs); err != nil {
		return nil, err
	}
	return gvkmap, nil
}

// GroupVersionKindsFromBytes parses the bytes and returns a gvk slice
func GroupVersionKindsFromBytes(b []byte) ([]schema.GroupVersionKind, error) {
	gvkmap, err := SplitYAMLDocuments(b)
	if err != nil {
		return nil, err
	}
	gvks := []schema.GroupVersionKind{}
	for gvk := range gvkmap {
		gvks = append(gvks, gvk)
	}
	return gvks, nil
}

// GroupVersionKindsHasKind returns whether the following gvk slice contains the kind given as a parameter
func GroupVersionKindsHasKind(gvks []schema.GroupVersionKind, kind string) bool {
	for _, gvk := range gvks {
		if gvk.Kind == kind {
			return true
		}
	}
	return false
}

// GroupVersionKindsHasClusterConfiguration returns whether the following gvk slice contains a ClusterConfiguration object
func GroupVersionKindsHasClusterConfiguration(gvks ...schema.GroupVersionKind) bool {
	return GroupVersionKindsHasKind(gvks, constants.ClusterConfigurationKind)
}

// GroupVersionKindsHasInitConfiguration returns whether the following gvk slice contains a InitConfiguration object
func GroupVersionKindsHasInitConfiguration(gvks ...schema.GroupVersionKind) bool {
	return GroupVersionKindsHasKind(gvks, constants.InitConfigurationKind)
}

// GroupVersionKindsHasJoinConfiguration returns whether the following gvk slice contains a JoinConfiguration object
func GroupVersionKindsHasJoinConfiguration(gvks ...schema.GroupVersionKind) bool {
	return GroupVersionKindsHasKind(gvks, constants.JoinConfigurationKind)
}
