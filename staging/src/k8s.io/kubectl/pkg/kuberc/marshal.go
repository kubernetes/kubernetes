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

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"

	"k8s.io/kubectl/pkg/config"
)

// decodePreference iterates over the yamls in kuberc file to find the supported kuberc version.
// Once it finds, it returns the compatible kuberc object as well as accumulated errors during the iteration.
func decodePreference(kubercFile string, verbosity uint64, errOut io.Writer) (*config.Preference, error) {
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
		pref, gvk, strictDecodeErr := strictCodecs.UniversalDecoder().Decode(doc, nil, nil)
		if strictDecodeErr != nil {
			var lenientDecodeErr error
			pref, gvk, lenientDecodeErr = lenientCodecs.UniversalDecoder().Decode(doc, nil, nil)
			if lenientDecodeErr != nil {
				// both strict and lenient failed
				// verbose log the error with the most information about this item and continue
				if verbosity >= 5 {
					_, _ = fmt.Fprintf(errOut, "kuberc: strict decoding error for entry %d in %s: %v\n", attemptedItems, kubercFile, strictDecodeErr)
				}
				continue
			}
		}

		// check expected GVK, if bad, verbose log and continue
		expectedGK := schema.GroupKind{
			Group: config.SchemeGroupVersion.Group,
			Kind:  "Preference",
		}
		if gvk.GroupKind() != expectedGK {
			if verbosity >= 5 {
				_, _ = fmt.Fprintf(errOut, "kuberc: unexpected GroupVersionKind for entry %d in %s: %v\n", attemptedItems, kubercFile, gvk)
			}
			continue
		}

		// check expected go type, if bad, verbose log and continue
		preferences, ok := pref.(*config.Preference)
		if !ok {
			if verbosity >= 5 {
				_, _ = fmt.Fprintf(errOut, "kuberc: unexpected object type %T for entry %d in %s\n", pref, attemptedItems, kubercFile)
			}
			continue
		}

		// we have a usable preferences to return
		if verbosity >= 5 {
			_, _ = fmt.Fprintf(errOut, "kuberc: successfully decoded entry %d in %s\n", attemptedItems, kubercFile)
		}
		return preferences, strictDecodeErr

	}
	if attemptedItems > 0 {
		return nil, fmt.Errorf("no valid preferences found in %s, use --v=5 to see details", kubercFile)
	}
	// empty doc
	if verbosity >= 5 {
		_, _ = fmt.Fprintf(errOut, "kuberc: no preferences found in %s\n", kubercFile)
	}
	return nil, nil
}
