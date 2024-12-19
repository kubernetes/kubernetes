/*
Copyright 2024 The Kubernetes Authors.

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

package kuberc

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/runtime/schema"
	yamlserializer "k8s.io/apimachinery/pkg/runtime/serializer/yaml"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"

	"k8s.io/kubectl/pkg/config"
)

// decodePreference iterates over the yamls in kuberc file to find the supported kuberc version.
// Once it finds, it returns the compatible kuberc object as well as accumulated errors during the iteration.
func decodePreference(kubercFile string, explicitly bool) (*config.Preference, error) {
	kubercBytes, err := os.ReadFile(kubercFile)
	if err != nil {
		if !explicitly && os.IsNotExist(err) {
			// We don't log if the kuberc file does not exist. Because user simply does not
			// specify neither default kuberc file nor explicitly pass it.
			// We'll continue to default behavior without raising any error.
			return nil, nil
		}
		return nil, err
	}

	var errs []error
	document, err := splitYAMLDocuments(kubercBytes)
	if err != nil {
		errs = append(errs, err)
	}
	for docGVK, doc := range document {
		pref, gvk, decodeErr := strictCodecs.UniversalDecoder().Decode(doc, nil, nil)
		if decodeErr != nil {
			errs = append(errs, decodeErr)

			// default kuberc is incompatible with this version, or it simply is invalid.
			// falling back to lenient decoding to do our best.
			pref, gvk, decodeErr = lenientCodecs.UniversalDecoder().Decode(doc, nil, nil)
			if decodeErr != nil {
				errs = append(errs, decodeErr)
				continue
			}
		}

		expectedGK := schema.GroupKind{
			Group: config.SchemeGroupVersion.Group,
			Kind:  "Preference",
		}
		if gvk.GroupKind() != expectedGK && docGVK.GroupKind() != expectedGK {
			errs = append(errs, fmt.Errorf("unsupported preference GVK %s", gvk.GroupKind().String()))
			continue
		}

		preferences, ok := pref.(*config.Preference)
		if !ok {
			errs = append(errs, fmt.Errorf("preferences in %s file is not a valid config.Preference type", kubercFile))
			continue
		}

		return preferences, errorsutil.NewAggregate(errs)
	}

	return nil, errorsutil.NewAggregate(errs)
}

// splitYAMLDocuments reads the YAML bytes per-document, unmarshals the TypeMeta information from each document
// and returns a map between the GroupVersionKind of the document and the document bytes
func splitYAMLDocuments(yamlBytes []byte) (map[schema.GroupVersionKind][]byte, error) {
	gvkMap := make(map[schema.GroupVersionKind][]byte)
	var errs []error
	buf := bytes.NewBuffer(yamlBytes)
	reader := utilyaml.NewYAMLReader(bufio.NewReader(buf))
	for {
		// Read one YAML document at a time, until io.EOF is returned
		b, err := reader.Read()
		if errors.Is(err, io.EOF) {
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
			errs = append(errs, errors.Errorf("invalid configuration for GroupVersionKind %+v: kind and apiVersion is mandatory information that must be specified", gvk))
			continue
		}

		// There must be only one kuberc file with the given group, version and kind, e.g. preference.kubectl.config.k8s.io/v1alpha1
		if _, ok := gvkMap[*gvk]; ok {
			errs = append(errs, errors.Errorf("invalid configuration: GroupVersionKind %+v is specified more than once in YAML file", gvk))
			continue
		}
		// Save the mapping between the gvk and the bytes that object consists of
		gvkMap[*gvk] = b
	}
	if err := errorsutil.NewAggregate(errs); err != nil {
		return nil, err
	}
	return gvkMap, nil
}
