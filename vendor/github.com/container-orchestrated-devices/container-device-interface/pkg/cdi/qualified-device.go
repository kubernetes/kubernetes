/*
   Copyright © 2021 The CDI Authors

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

package cdi

import (
	"strings"

	"github.com/pkg/errors"
)

// QualifiedName returns the qualified name for a device.
// The syntax for a qualified device names is
//   "<vendor>/<class>=<name>".
// A valid vendor name may contain the following runes:
//   'A'-'Z', 'a'-'z', '0'-'9', '.', '-', '_'.
// A valid class name may contain the following runes:
//   'A'-'Z', 'a'-'z', '0'-'9', '-', '_'.
// A valid device name may containe the following runes:
//   'A'-'Z', 'a'-'z', '0'-'9', '-', '_', '.', ':'
func QualifiedName(vendor, class, name string) string {
	return vendor + "/" + class + "=" + name
}

// IsQualifiedName tests if a device name is qualified.
func IsQualifiedName(device string) bool {
	_, _, _, err := ParseQualifiedName(device)
	return err == nil
}

// ParseQualifiedName splits a qualified name into device vendor, class,
// and name. If the device fails to parse as a qualified name, or if any
// of the split components fail to pass syntax validation, vendor and
// class are returned as empty, together with the verbatim input as the
// name and an error describing the reason for failure.
func ParseQualifiedName(device string) (string, string, string, error) {
	vendor, class, name := ParseDevice(device)

	if vendor == "" {
		return "", "", device, errors.Errorf("unqualified device %q, missing vendor", device)
	}
	if class == "" {
		return "", "", device, errors.Errorf("unqualified device %q, missing class", device)
	}
	if name == "" {
		return "", "", device, errors.Errorf("unqualified device %q, missing device name", device)
	}

	if err := ValidateVendorName(vendor); err != nil {
		return "", "", device, errors.Wrapf(err, "invalid device %q", device)
	}
	if err := ValidateClassName(class); err != nil {
		return "", "", device, errors.Wrapf(err, "invalid device %q", device)
	}
	if err := ValidateDeviceName(name); err != nil {
		return "", "", device, errors.Wrapf(err, "invalid device %q", device)
	}

	return vendor, class, name, nil
}

// ParseDevice tries to split a device name into vendor, class, and name.
// If this fails, for instance in the case of unqualified device names,
// ParseDevice returns an empty vendor and class together with name set
// to the verbatim input.
func ParseDevice(device string) (string, string, string) {
	if device == "" || device[0] == '/' {
		return "", "", device
	}

	parts := strings.SplitN(device, "=", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", device
	}

	name := parts[1]
	vendor, class := ParseQualifier(parts[0])
	if vendor == "" {
		return "", "", device
	}

	return vendor, class, name
}

// ParseQualifier splits a device qualifier into vendor and class.
// The syntax for a device qualifier is
//     "<vendor>/<class>"
// If parsing fails, an empty vendor and the class set to the
// verbatim input is returned.
func ParseQualifier(kind string) (string, string) {
	parts := strings.SplitN(kind, "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", kind
	}
	return parts[0], parts[1]
}

// ValidateVendorName checks the validity of a vendor name.
// A vendor name may contain the following ASCII characters:
//   - upper- and lowercase letters ('A'-'Z', 'a'-'z')
//   - digits ('0'-'9')
//   - underscore, dash, and dot ('_', '-', and '.')
func ValidateVendorName(vendor string) error {
	if vendor == "" {
		return errors.Errorf("invalid (empty) vendor name")
	}
	if !isLetter(rune(vendor[0])) {
		return errors.Errorf("invalid vendor %q, should start with letter", vendor)
	}
	for _, c := range string(vendor[1 : len(vendor)-1]) {
		switch {
		case isAlphaNumeric(c):
		case c == '_' || c == '-' || c == '.':
		default:
			return errors.Errorf("invalid character '%c' in vendor name %q",
				c, vendor)
		}
	}
	if !isAlphaNumeric(rune(vendor[len(vendor)-1])) {
		return errors.Errorf("invalid vendor %q, should end with letter", vendor)
	}

	return nil
}

// ValidateClassName checks the validity of class name.
// A class name may contain the following ASCII characters:
//   - upper- and lowercase letters ('A'-'Z', 'a'-'z')
//   - digits ('0'-'9')
//   - underscore and dash ('_', '-')
func ValidateClassName(class string) error {
	if class == "" {
		return errors.Errorf("invalid (empty) device class")
	}
	if !isLetter(rune(class[0])) {
		return errors.Errorf("invalid class %q, should start with letter", class)
	}
	for _, c := range string(class[1 : len(class)-1]) {
		switch {
		case isAlphaNumeric(c):
		case c == '_' || c == '-':
		default:
			return errors.Errorf("invalid character '%c' in device class %q",
				c, class)
		}
	}
	if !isAlphaNumeric(rune(class[len(class)-1])) {
		return errors.Errorf("invalid class %q, should end with letter", class)
	}
	return nil
}

// ValidateDeviceName checks the validity of a device name.
// A device name may contain the following ASCII characters:
//   - upper- and lowercase letters ('A'-'Z', 'a'-'z')
//   - digits ('0'-'9')
//   - underscore, dash, dot, colon ('_', '-', '.', ':')
func ValidateDeviceName(name string) error {
	if name == "" {
		return errors.Errorf("invalid (empty) device name")
	}
	if !isLetter(rune(name[0])) {
		return errors.Errorf("invalid name %q, should start with letter", name)
	}
	for _, c := range string(name[1 : len(name)-1]) {
		switch {
		case isAlphaNumeric(c):
		case c == '_' || c == '-' || c == '.' || c == ':':
		default:
			return errors.Errorf("invalid character '%c' in device name %q",
				c, name)
		}
	}
	if !isAlphaNumeric(rune(name[len(name)-1])) {
		return errors.Errorf("invalid name %q, should start with letter", name)
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
