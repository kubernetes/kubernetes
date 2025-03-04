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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	yamlserializer "k8s.io/apimachinery/pkg/runtime/serializer/yaml"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
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
	const mediaType = runtime.ContentTypeYAML
	info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return []byte{}, errors.Errorf("unsupported media type %q", mediaType)
	}

	encoder := codecs.EncoderForVersion(info.Serializer, gv)
	return runtime.Encode(encoder, obj)
}

// UniversalUnmarshal unmarshals YAML or JSON into a runtime.Object using the universal deserializer.
func UniversalUnmarshal(buffer []byte) (runtime.Object, error) {
	codecs := clientsetscheme.Codecs
	decoder := codecs.UniversalDeserializer()
	obj, _, err := decoder.Decode(buffer, nil, nil)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to decode %s into runtime.Object", buffer)
	}
	return obj, nil
}

// SplitConfigDocuments reads the YAML/JSON bytes per-document, unmarshals the TypeMeta information from each document
// and returns a map between the GroupVersionKind of the document and the document bytes
func SplitConfigDocuments(documentBytes []byte) (kubeadmapi.DocumentMap, error) {
	gvkmap := kubeadmapi.DocumentMap{}
	knownKinds := map[string]bool{}
	errs := []error{}
	buf := bytes.NewBuffer(documentBytes)
	reader := utilyaml.NewYAMLReader(bufio.NewReader(buf))
	for {
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
		gvk, err := yamlserializer.DefaultMetaFactory.Interpret(b)
		if err != nil {
			return nil, err
		}
		if len(gvk.Group) == 0 || len(gvk.Version) == 0 || len(gvk.Kind) == 0 {
			return nil, errors.Errorf("invalid configuration for GroupVersionKind %+v: kind and apiVersion is mandatory information that must be specified", gvk)
		}

		// Check whether the kind has been registered before. If it has, throw an error
		if known := knownKinds[gvk.Kind]; known {
			errs = append(errs, errors.Errorf("invalid configuration: kind %q is specified twice in YAML file", gvk.Kind))
			continue
		}
		knownKinds[gvk.Kind] = true

		// Save the mapping between the gvk and the bytes that object consists of
		gvkmap[*gvk] = b
	}
	if err := errorsutil.NewAggregate(errs); err != nil {
		return nil, err
	}
	return gvkmap, nil
}

// GroupVersionKindsFromBytes parses the bytes and returns a gvk slice
func GroupVersionKindsFromBytes(b []byte) ([]schema.GroupVersionKind, error) {
	gvkmap, err := SplitConfigDocuments(b)
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

// GroupVersionKindsHasResetConfiguration returns whether the following gvk slice contains a ResetConfiguration object
func GroupVersionKindsHasResetConfiguration(gvks ...schema.GroupVersionKind) bool {
	return GroupVersionKindsHasKind(gvks, constants.ResetConfigurationKind)
}

// GroupVersionKindsHasUpgradeConfiguration returns whether the following gvk slice contains a UpgradeConfiguration object
func GroupVersionKindsHasUpgradeConfiguration(gvks ...schema.GroupVersionKind) bool {
	return GroupVersionKindsHasKind(gvks, constants.UpgradeConfigurationKind)
}
