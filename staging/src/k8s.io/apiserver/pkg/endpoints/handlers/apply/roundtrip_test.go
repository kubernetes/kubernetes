/*
Copyright 2018 The Kubernetes Authors.

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

package apply

import (
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/ghodss/yaml"
)

// TestRoundTripManagedFields will roundtrip ManagedFields from the format used by
// sigs.k8s.io/structured-merge-diff to the wire format (api format) and back
func TestRoundTripManagedFields(t *testing.T) {
	tests := []struct {
		yaml      []byte
		errString string
	}{
		{
			yaml: []byte(`foo:
  apiVersion: v1
  fields:
    children:
    - pathElement:
        index: 5
      set:
        members:
        - fieldName: i
    - pathElement:
        value:
          floatValue: 3.1415
      set:
        members:
        - fieldName: pi
    - pathElement:
        value:
          intValue: 3
      set:
        members:
        - fieldName: alsoPi
    - pathElement:
        value:
          booleanValue: false
      set:
        members:
        - fieldName: notTrue`),
		}, {
			yaml: []byte(`foo:
  apiVersion: v1
  fields:
    children:
    - pathElement:
        fieldName: spec
      set:
        children:
        - pathElement:
            fieldName: containers
          set:
            children:
            - pathElement:
                key:
                - name: name
                  value:
                    stringValue: c
                    'null': false
              set:
                members:
                - fieldName: image
                - fieldName: name`),
		}, {
			yaml: []byte(`foo:
  apiVersion: v1
  fields:
    members:
    - fieldName: apiVersion
    - fieldName: kind
    children:
    - pathElement:
        fieldName: metadata
      set:
        members:
        - fieldName: name
        children:
        - pathElement:
            fieldName: labels
          set:
            members:
            - fieldName: app
    - pathElement:
        fieldName: spec
      set:
        members:
        - fieldName: replicas
        children:
        - pathElement:
            fieldName: selector
          set:
            children:
            - pathElement:
                fieldName: matchLabels
              set:
                members:
                - fieldName: app
        - pathElement:
            fieldName: template
          set:
            children:
            - pathElement:
                fieldName: metadata
              set:
                children:
                - pathElement:
                    fieldName: labels
                  set:
                    members:
                    - fieldName: app
            - pathElement:
                fieldName: spec
              set:
                children:
                - pathElement:
                    fieldName: containers
                  set:
                    children:
                    - pathElement:
                        key:
                        - name: name
                          value:
                            stringValue: nginx
                            'null': false
                      set:
                        members:
                        - fieldName: image
                        - fieldName: name
                        children:
                        - pathElement:
                            fieldName: ports
                          set:
                            children:
                            - pathElement:
                                index: 0
                              set:
                                members:
                                - fieldName: containerPort`),
		}, {
			yaml: []byte(`foo:
  apiVersion: v1
  fields:
    members:
    - fieldName: allowVolumeExpansion
    - fieldName: apiVersion
    - fieldName: kind
    - fieldName: provisioner
    children:
    - pathElement:
        fieldName: metadata
      set:
        members:
        - fieldName: name
    - pathElement:
        fieldName: parameters
      set:
        members:
        - fieldName: resturl
        - fieldName: restuser
        - fieldName: secretName
        - fieldName: secretNamespace`),
		}, {
			yaml: []byte(`foo:
  apiVersion: v1
  fields:
    members:
    - fieldName: apiVersion
    - fieldName: kind
    children:
    - pathElement:
        fieldName: metadata
      set:
        members:
        - fieldName: name
    - pathElement:
        fieldName: spec
      set:
        members:
        - fieldName: group
        - fieldName: scope
        children:
        - pathElement:
            fieldName: names
          set:
            members:
            - fieldName: kind
            - fieldName: plural
            - fieldName: singular
            children:
            - pathElement:
                fieldName: shortNames
              set:
                members:
                - index: 0
        - pathElement:
            fieldName: versions
          set:
            children:
            - pathElement:
                key:
                - name: name
                  value:
                    stringValue: v1
                    'null': false
              set:
                members:
                - fieldName: name
                - fieldName: served
                - fieldName: storage`),
		},
	}

	for i, tc := range tests {
		var original map[string]metav1.VersionedFieldSet
		if err := yaml.Unmarshal(tc.yaml, &original); err != nil {
			t.Errorf("[%v]did not expect yaml unmarshalling error but got: %v", i, err)
			continue
		}

		decoded, err := DecodeManagedFields(original)
		if err == nil && len(tc.errString) > 0 {
			t.Errorf("[%v]expected error but got none.", i)
			continue
		}
		if err != nil && len(tc.errString) == 0 {
			t.Errorf("[%v]did not expect error but got: %v", i, err)
			continue
		}
		if err != nil && len(tc.errString) > 0 && !strings.Contains(err.Error(), tc.errString) {
			t.Errorf("[%v]expected error with %q but got: %v", i, tc.errString, err)
			continue
		}
		encoded, err := EncodeManagedFields(decoded)
		if err != nil {
			t.Errorf("[%v]did not expect round trip error but got: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(encoded, original) {
			t.Errorf("[%v]expected:\n\t%+v\nbut got:\n\t%+v", i, original, encoded)
		}
	}
}
