/*
Copyright 2022 The Kubernetes Authors.

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

// The code below was copied from
// https://github.com/container-orchestrated-devices/container-device-interface/blob/v0.5.3/pkg/cdi/annotations.go
// https://github.com/container-orchestrated-devices/container-device-interface/blob/v0.5.3/pkg/cdi/qualified-device.go
// to avoid a dependency on that package and the indirect dependencies that
// this would have implied.
//
// Long term it would be good to avoid this duplication:
// https://github.com/container-orchestrated-devices/container-device-interface/issues/97

package cdi

import (
	"errors"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const (
	// annotationPrefix is the prefix for CDI container annotation keys.
	annotationPrefix = "cdi.k8s.io/"
)

// GenerateAnnotations generate container annotations using CDI UpdateAnnotations API.
func GenerateAnnotations(
	claimUID types.UID,
	driverName string,
	cdiDevices []string,
) ([]kubecontainer.Annotation, error) {
	if len(cdiDevices) == 0 {
		return nil, nil
	}
	annotations, err := updateAnnotations(map[string]string{}, driverName, string(claimUID), cdiDevices)
	if err != nil {
		return nil, fmt.Errorf("can't generate CDI annotations: %+v", err)
	}

	kubeAnnotations := []kubecontainer.Annotation{}
	for key, value := range annotations {
		kubeAnnotations = append(kubeAnnotations, kubecontainer.Annotation{Name: key, Value: value})
	}

	return kubeAnnotations, nil
}

// updateAnnotations updates annotations with a plugin-specific CDI device
// injection request for the given devices. Upon any error a non-nil error
// is returned and annotations are left intact. By convention plugin should
// be in the format of "vendor.device-type".
func updateAnnotations(annotations map[string]string, plugin string, deviceID string, devices []string) (map[string]string, error) {
	key, err := annotationKey(plugin, deviceID)
	if err != nil {
		return annotations, fmt.Errorf("CDI annotation failed: %v", err)
	}
	if _, ok := annotations[key]; ok {
		return annotations, fmt.Errorf("CDI annotation failed, key %q used", key)
	}
	value, err := annotationValue(devices)
	if err != nil {
		return annotations, fmt.Errorf("CDI annotation failed: %v", err)
	}

	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[key] = value

	return annotations, nil
}

// annotationKey returns a unique annotation key for an device allocation
// by a K8s device plugin. pluginName should be in the format of
// "vendor.device-type". deviceID is the ID of the device the plugin is
// allocating. It is used to make sure that the generated key is unique
// even if multiple allocations by a single plugin needs to be annotated.
func annotationKey(pluginName, deviceID string) (string, error) {
	const maxNameLen = 63

	if pluginName == "" {
		return "", errors.New("invalid plugin name, empty")
	}
	if deviceID == "" {
		return "", errors.New("invalid deviceID, empty")
	}

	name := pluginName + "_" + strings.ReplaceAll(deviceID, "/", "_")

	if len(name) > maxNameLen {
		return "", fmt.Errorf("invalid plugin+deviceID %q, too long", name)
	}

	if c := rune(name[0]); !isAlphaNumeric(c) {
		return "", fmt.Errorf("invalid name %q, first '%c' should be alphanumeric", name, c)
	}
	if len(name) > 2 {
		for _, c := range name[1 : len(name)-1] {
			switch {
			case isAlphaNumeric(c):
			case c == '_' || c == '-' || c == '.':
			default:
				return "", fmt.Errorf("invalid name %q, invalid charcter '%c'", name, c)
			}
		}
	}
	if c := rune(name[len(name)-1]); !isAlphaNumeric(c) {
		return "", fmt.Errorf("invalid name %q, last '%c' should be alphanumeric", name, c)
	}

	return annotationPrefix + name, nil
}

// annotationValue returns an annotation value for the given devices.
func annotationValue(devices []string) (string, error) {
	value, sep := "", ""
	for _, d := range devices {
		if _, _, _, err := parseQualifiedName(d); err != nil {
			return "", err
		}
		value += sep + d
		sep = ","
	}

	return value, nil
}

// parseQualifiedName splits a qualified name into device vendor, class,
// and name. If the device fails to parse as a qualified name, or if any
// of the split components fail to pass syntax validation, vendor and
// class are returned as empty, together with the verbatim input as the
// name and an error describing the reason for failure.
func parseQualifiedName(device string) (string, string, string, error) {
	vendor, class, name := parseDevice(device)

	if vendor == "" {
		return "", "", device, fmt.Errorf("unqualified device %q, missing vendor", device)
	}
	if class == "" {
		return "", "", device, fmt.Errorf("unqualified device %q, missing class", device)
	}
	if name == "" {
		return "", "", device, fmt.Errorf("unqualified device %q, missing device name", device)
	}

	if err := validateVendorName(vendor); err != nil {
		return "", "", device, fmt.Errorf("invalid device %q: %v", device, err)
	}
	if err := validateClassName(class); err != nil {
		return "", "", device, fmt.Errorf("invalid device %q: %v", device, err)
	}
	if err := validateDeviceName(name); err != nil {
		return "", "", device, fmt.Errorf("invalid device %q: %v", device, err)
	}

	return vendor, class, name, nil
}

// parseDevice tries to split a device name into vendor, class, and name.
// If this fails, for instance in the case of unqualified device names,
// parseDevice returns an empty vendor and class together with name set
// to the verbatim input.
func parseDevice(device string) (string, string, string) {
	if device == "" || device[0] == '/' {
		return "", "", device
	}

	parts := strings.SplitN(device, "=", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", device
	}

	name := parts[1]
	vendor, class := parseQualifier(parts[0])
	if vendor == "" {
		return "", "", device
	}

	return vendor, class, name
}

// parseQualifier splits a device qualifier into vendor and class.
// The syntax for a device qualifier is
//
//	"<vendor>/<class>"
//
// If parsing fails, an empty vendor and the class set to the
// verbatim input is returned.
func parseQualifier(kind string) (string, string) {
	parts := strings.SplitN(kind, "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", kind
	}
	return parts[0], parts[1]
}

// validateVendorName checks the validity of a vendor name.
// A vendor name may contain the following ASCII characters:
//   - upper- and lowercase letters ('A'-'Z', 'a'-'z')
//   - digits ('0'-'9')
//   - underscore, dash, and dot ('_', '-', and '.')
func validateVendorName(vendor string) error {
	if vendor == "" {
		return fmt.Errorf("invalid (empty) vendor name")
	}
	if !isLetter(rune(vendor[0])) {
		return fmt.Errorf("invalid vendor %q, should start with letter", vendor)
	}
	for _, c := range string(vendor[1 : len(vendor)-1]) {
		switch {
		case isAlphaNumeric(c):
		case c == '_' || c == '-' || c == '.':
		default:
			return fmt.Errorf("invalid character '%c' in vendor name %q",
				c, vendor)
		}
	}
	if !isAlphaNumeric(rune(vendor[len(vendor)-1])) {
		return fmt.Errorf("invalid vendor %q, should end with a letter or digit", vendor)
	}

	return nil
}

// validateClassName checks the validity of class name.
// A class name may contain the following ASCII characters:
//   - upper- and lowercase letters ('A'-'Z', 'a'-'z')
//   - digits ('0'-'9')
//   - underscore and dash ('_', '-')
func validateClassName(class string) error {
	if class == "" {
		return fmt.Errorf("invalid (empty) device class")
	}
	if !isLetter(rune(class[0])) {
		return fmt.Errorf("invalid class %q, should start with letter", class)
	}
	for _, c := range string(class[1 : len(class)-1]) {
		switch {
		case isAlphaNumeric(c):
		case c == '_' || c == '-':
		default:
			return fmt.Errorf("invalid character '%c' in device class %q",
				c, class)
		}
	}
	if !isAlphaNumeric(rune(class[len(class)-1])) {
		return fmt.Errorf("invalid class %q, should end with a letter or digit", class)
	}
	return nil
}

// validateDeviceName checks the validity of a device name.
// A device name may contain the following ASCII characters:
//   - upper- and lowercase letters ('A'-'Z', 'a'-'z')
//   - digits ('0'-'9')
//   - underscore, dash, dot, colon ('_', '-', '.', ':')
func validateDeviceName(name string) error {
	if name == "" {
		return fmt.Errorf("invalid (empty) device name")
	}
	if !isAlphaNumeric(rune(name[0])) {
		return fmt.Errorf("invalid class %q, should start with a letter or digit", name)
	}
	if len(name) == 1 {
		return nil
	}
	for _, c := range string(name[1 : len(name)-1]) {
		switch {
		case isAlphaNumeric(c):
		case c == '_' || c == '-' || c == '.' || c == ':':
		default:
			return fmt.Errorf("invalid character '%c' in device name %q",
				c, name)
		}
	}
	if !isAlphaNumeric(rune(name[len(name)-1])) {
		return fmt.Errorf("invalid name %q, should end with a letter or digit", name)
	}
	return nil
}

func isLetter(c rune) bool {
	return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z')
}

func isDigit(c rune) bool {
	return '0' <= c && c <= '9'
}

func isAlphaNumeric(c rune) bool {
	return isLetter(c) || isDigit(c)
}
