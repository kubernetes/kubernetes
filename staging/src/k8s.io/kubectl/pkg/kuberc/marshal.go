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
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/klog/v2"
	"k8s.io/kubectl/pkg/config"
	"k8s.io/kubectl/pkg/config/scheme"
	"k8s.io/kubectl/pkg/config/v1beta1"
	"sigs.k8s.io/yaml"
)

// decodePreference iterates over the yamls in kuberc file to find the supported kuberc version.
// Once it finds, it returns the compatible kuberc object as well as accumulated errors during the iteration.
func decodePreference(kubercFile string) (*config.Preference, error) {
	kubercBytes, err := os.ReadFile(kubercFile)
	if err != nil {
		return nil, err
	}

	attemptedItems := 0
	reader := utilyaml.NewYAMLReader(bufio.NewReader(bytes.NewBuffer(kubercBytes)))
	for {
		doc, readErr := reader.Read()
		if errors.Is(readErr, io.EOF) {
			// no more entries, expected when we reach the end of the file
			break
		}
		if readErr != nil {
			// other errors are fatal
			return nil, readErr
		}
		if len(bytes.TrimSpace(doc)) == 0 {
			// empty item, ignore
			continue
		}
		// remember we attempted
		attemptedItems++
		pref, gvk, strictDecodeErr := scheme.StrictCodecs.UniversalDecoder().Decode(doc, nil, nil)
		if strictDecodeErr != nil {
			var lenientDecodeErr error
			pref, gvk, lenientDecodeErr = scheme.LenientCodecs.UniversalDecoder().Decode(doc, nil, nil)
			if lenientDecodeErr != nil {
				// both strict and lenient failed
				// verbose log the error with the most information about this item and continue
				klog.V(5).Infof("kuberc: strict decoding error for entry %d in %s: %v", attemptedItems, kubercFile, strictDecodeErr)
				continue
			}
		}

		// check expected GVK, if bad, verbose log and continue
		expectedGK := schema.GroupKind{
			Group: config.SchemeGroupVersion.Group,
			Kind:  "Preference",
		}
		if gvk.GroupKind() != expectedGK {
			klog.V(5).Infof("kuberc: unexpected GroupVersionKind for entry %d in %s: %v", attemptedItems, kubercFile, gvk)
			continue
		}

		// check expected go type, if bad, verbose log and continue
		preferences, ok := pref.(*config.Preference)
		if !ok {
			klog.V(5).Infof("kuberc: unexpected object type %T for entry %d in %s", pref, attemptedItems, kubercFile)
			continue
		}

		// we have a usable preferences to return
		klog.V(5).Infof("kuberc: using entry %d (%s) in %s", attemptedItems, gvk.GroupVersion(), kubercFile)
		return preferences, strictDecodeErr
	}

	if attemptedItems > 0 {
		return nil, fmt.Errorf("no valid preferences found in %s, use --v=5 to see details", kubercFile)
	}

	// empty doc
	klog.V(5).Infof("kuberc: no preferences found in %s", kubercFile)
	return nil, nil
}

// SavePreference saves the preference to the kuberc file
func SavePreference(pref *v1beta1.Preference, file string, out io.Writer) error {
	dir := filepath.Dir(file)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	data, err := yaml.Marshal(pref)
	if err != nil {
		return fmt.Errorf("failed to marshal preferences: %w", err)
	}

	if err := os.WriteFile(file, data, 0644); err != nil {
		return fmt.Errorf("failed to write kuberc file: %w", err)
	}

	fmt.Fprintf(out, "Updated %s\n", file) // nolint:errcheck
	return nil
}

// CreateDefaultPreference returns Preference object with
// some default configurations.
func CreateDefaultPreference() *v1beta1.Preference {
	return &v1beta1.Preference{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "kubectl.config.k8s.io/v1beta1",
			Kind:       "Preference",
		},
		Defaults: []v1beta1.CommandDefaults{
			{
				Command: "apply",
				Options: []v1beta1.CommandOptionDefault{
					{
						Name:    "server-side",
						Default: "true",
					},
				},
			},
			{
				Command: "delete",
				Options: []v1beta1.CommandOptionDefault{
					{
						Name:    "interactive",
						Default: "true",
					},
				},
			},
		},
	}
}
