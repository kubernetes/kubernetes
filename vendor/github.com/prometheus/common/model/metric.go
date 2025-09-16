// Copyright 2013 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	dto "github.com/prometheus/client_model/go"
	"go.yaml.in/yaml/v2"
	"google.golang.org/protobuf/proto"
)

var (
	// NameValidationScheme determines the global default method of the name
	// validation to be used by all calls to IsValidMetricName() and LabelName
	// IsValid().
	//
	// Deprecated: This variable should not be used and might be removed in the
	// far future. If you wish to stick to the legacy name validation use
	// `IsValidLegacyMetricName()` and `LabelName.IsValidLegacy()` methods
	// instead. This variable is here as an escape hatch for emergency cases,
	// given the recent change from `LegacyValidation` to `UTF8Validation`, e.g.,
	// to delay UTF-8 migrations in time or aid in debugging unforeseen results of
	// the change. In such a case, a temporary assignment to `LegacyValidation`
	// value in the `init()` function in your main.go or so, could be considered.
	//
	// Historically we opted for a global variable for feature gating different
	// validation schemes in operations that were not otherwise easily adjustable
	// (e.g. Labels yaml unmarshaling). That could have been a mistake, a separate
	// Labels structure or package might have been a better choice. Given the
	// change was made and many upgraded the common already, we live this as-is
	// with this warning and learning for the future.
	NameValidationScheme = UTF8Validation

	// NameEscapingScheme defines the default way that names will be escaped when
	// presented to systems that do not support UTF-8 names. If the Content-Type
	// "escaping" term is specified, that will override this value.
	// NameEscapingScheme should not be set to the NoEscaping value. That string
	// is used in content negotiation to indicate that a system supports UTF-8 and
	// has that feature enabled.
	NameEscapingScheme = UnderscoreEscaping
)

// ValidationScheme is a Go enum for determining how metric and label names will
// be validated by this library.
type ValidationScheme int

const (
	// UnsetValidation represents an undefined ValidationScheme.
	// Should not be used in practice.
	UnsetValidation ValidationScheme = iota

	// LegacyValidation is a setting that requires that all metric and label names
	// conform to the original Prometheus character requirements described by
	// MetricNameRE and LabelNameRE.
	LegacyValidation

	// UTF8Validation only requires that metric and label names be valid UTF-8
	// strings.
	UTF8Validation
)

var _ interface {
	yaml.Marshaler
	yaml.Unmarshaler
	json.Marshaler
	json.Unmarshaler
	fmt.Stringer
} = new(ValidationScheme)

// String returns the string representation of s.
func (s ValidationScheme) String() string {
	switch s {
	case UnsetValidation:
		return "unset"
	case LegacyValidation:
		return "legacy"
	case UTF8Validation:
		return "utf8"
	default:
		panic(fmt.Errorf("unhandled ValidationScheme: %d", s))
	}
}

// MarshalYAML implements the yaml.Marshaler interface.
func (s ValidationScheme) MarshalYAML() (any, error) {
	switch s {
	case UnsetValidation:
		return "", nil
	case LegacyValidation, UTF8Validation:
		return s.String(), nil
	default:
		panic(fmt.Errorf("unhandled ValidationScheme: %d", s))
	}
}

// UnmarshalYAML implements the yaml.Unmarshaler interface.
func (s *ValidationScheme) UnmarshalYAML(unmarshal func(any) error) error {
	var scheme string
	if err := unmarshal(&scheme); err != nil {
		return err
	}
	return s.Set(scheme)
}

// MarshalJSON implements the json.Marshaler interface.
func (s ValidationScheme) MarshalJSON() ([]byte, error) {
	switch s {
	case UnsetValidation:
		return json.Marshal("")
	case UTF8Validation, LegacyValidation:
		return json.Marshal(s.String())
	default:
		return nil, fmt.Errorf("unhandled ValidationScheme: %d", s)
	}
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (s *ValidationScheme) UnmarshalJSON(bytes []byte) error {
	var repr string
	if err := json.Unmarshal(bytes, &repr); err != nil {
		return err
	}
	return s.Set(repr)
}

// Set implements the pflag.Value interface.
func (s *ValidationScheme) Set(text string) error {
	switch text {
	case "":
		// Don't change the value.
	case LegacyValidation.String():
		*s = LegacyValidation
	case UTF8Validation.String():
		*s = UTF8Validation
	default:
		return fmt.Errorf("unrecognized ValidationScheme: %q", text)
	}
	return nil
}

// IsValidMetricName returns whether metricName is valid according to s.
func (s ValidationScheme) IsValidMetricName(metricName string) bool {
	switch s {
	case LegacyValidation:
		if len(metricName) == 0 {
			return false
		}
		for i, b := range metricName {
			if !isValidLegacyRune(b, i) {
				return false
			}
		}
		return true
	case UTF8Validation:
		if len(metricName) == 0 {
			return false
		}
		return utf8.ValidString(metricName)
	default:
		panic(fmt.Sprintf("Invalid name validation scheme requested: %s", s.String()))
	}
}

// IsValidLabelName returns whether labelName is valid according to s.
func (s ValidationScheme) IsValidLabelName(labelName string) bool {
	switch s {
	case LegacyValidation:
		if len(labelName) == 0 {
			return false
		}
		for i, b := range labelName {
			// TODO: Apply De Morgan's law. Make sure there are tests for this.
			if !((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b == '_' || (b >= '0' && b <= '9' && i > 0)) { //nolint:staticcheck
				return false
			}
		}
		return true
	case UTF8Validation:
		if len(labelName) == 0 {
			return false
		}
		return utf8.ValidString(labelName)
	default:
		panic(fmt.Sprintf("Invalid name validation scheme requested: %s", s))
	}
}

// Type implements the pflag.Value interface.
func (ValidationScheme) Type() string {
	return "validationScheme"
}

type EscapingScheme int

const (
	// NoEscaping indicates that a name will not be escaped. Unescaped names that
	// do not conform to the legacy validity check will use a new exposition
	// format syntax that will be officially standardized in future versions.
	NoEscaping EscapingScheme = iota

	// UnderscoreEscaping replaces all legacy-invalid characters with underscores.
	UnderscoreEscaping

	// DotsEscaping is similar to UnderscoreEscaping, except that dots are
	// converted to `_dot_` and pre-existing underscores are converted to `__`.
	DotsEscaping

	// ValueEncodingEscaping prepends the name with `U__` and replaces all invalid
	// characters with the unicode value, surrounded by underscores. Single
	// underscores are replaced with double underscores.
	ValueEncodingEscaping
)

const (
	// EscapingKey is the key in an Accept or Content-Type header that defines how
	// metric and label names that do not conform to the legacy character
	// requirements should be escaped when being scraped by a legacy prometheus
	// system. If a system does not explicitly pass an escaping parameter in the
	// Accept header, the default NameEscapingScheme will be used.
	EscapingKey = "escaping"

	// Possible values for Escaping Key.
	AllowUTF8         = "allow-utf-8" // No escaping required.
	EscapeUnderscores = "underscores"
	EscapeDots        = "dots"
	EscapeValues      = "values"
)

// MetricNameRE is a regular expression matching valid metric
// names. Note that the IsValidMetricName function performs the same
// check but faster than a match with this regular expression.
var MetricNameRE = regexp.MustCompile(`^[a-zA-Z_:][a-zA-Z0-9_:]*$`)

// A Metric is similar to a LabelSet, but the key difference is that a Metric is
// a singleton and refers to one and only one stream of samples.
type Metric LabelSet

// Equal compares the metrics.
func (m Metric) Equal(o Metric) bool {
	return LabelSet(m).Equal(LabelSet(o))
}

// Before compares the metrics' underlying label sets.
func (m Metric) Before(o Metric) bool {
	return LabelSet(m).Before(LabelSet(o))
}

// Clone returns a copy of the Metric.
func (m Metric) Clone() Metric {
	clone := make(Metric, len(m))
	for k, v := range m {
		clone[k] = v
	}
	return clone
}

func (m Metric) String() string {
	metricName, hasName := m[MetricNameLabel]
	numLabels := len(m) - 1
	if !hasName {
		numLabels = len(m)
	}
	labelStrings := make([]string, 0, numLabels)
	for label, value := range m {
		if label != MetricNameLabel {
			labelStrings = append(labelStrings, fmt.Sprintf("%s=%q", label, value))
		}
	}

	switch numLabels {
	case 0:
		if hasName {
			return string(metricName)
		}
		return "{}"
	default:
		sort.Strings(labelStrings)
		return fmt.Sprintf("%s{%s}", metricName, strings.Join(labelStrings, ", "))
	}
}

// Fingerprint returns a Metric's Fingerprint.
func (m Metric) Fingerprint() Fingerprint {
	return LabelSet(m).Fingerprint()
}

// FastFingerprint returns a Metric's Fingerprint calculated by a faster hashing
// algorithm, which is, however, more susceptible to hash collisions.
func (m Metric) FastFingerprint() Fingerprint {
	return LabelSet(m).FastFingerprint()
}

// IsValidMetricName returns true iff name matches the pattern of MetricNameRE
// for legacy names, and iff it's valid UTF-8 if the UTF8Validation scheme is
// selected.
//
// Deprecated: This function should not be used and might be removed in the future.
// Use [ValidationScheme.IsValidMetricName] instead.
func IsValidMetricName(n LabelValue) bool {
	return NameValidationScheme.IsValidMetricName(string(n))
}

// IsValidLegacyMetricName is similar to IsValidMetricName but always uses the
// legacy validation scheme regardless of the value of NameValidationScheme.
// This function, however, does not use MetricNameRE for the check but a much
// faster hardcoded implementation.
//
// Deprecated: This function should not be used and might be removed in the future.
// Use [LegacyValidation.IsValidMetricName] instead.
func IsValidLegacyMetricName(n string) bool {
	return LegacyValidation.IsValidMetricName(n)
}

// EscapeMetricFamily escapes the given metric names and labels with the given
// escaping scheme. Returns a new object that uses the same pointers to fields
// when possible and creates new escaped versions so as not to mutate the
// input.
func EscapeMetricFamily(v *dto.MetricFamily, scheme EscapingScheme) *dto.MetricFamily {
	if v == nil {
		return nil
	}

	if scheme == NoEscaping {
		return v
	}

	out := &dto.MetricFamily{
		Help: v.Help,
		Type: v.Type,
		Unit: v.Unit,
	}

	// If the name is nil, copy as-is, don't try to escape.
	if v.Name == nil || IsValidLegacyMetricName(v.GetName()) {
		out.Name = v.Name
	} else {
		out.Name = proto.String(EscapeName(v.GetName(), scheme))
	}
	for _, m := range v.Metric {
		if !metricNeedsEscaping(m) {
			out.Metric = append(out.Metric, m)
			continue
		}

		escaped := &dto.Metric{
			Gauge:       m.Gauge,
			Counter:     m.Counter,
			Summary:     m.Summary,
			Untyped:     m.Untyped,
			Histogram:   m.Histogram,
			TimestampMs: m.TimestampMs,
		}

		for _, l := range m.Label {
			if l.GetName() == MetricNameLabel {
				if l.Value == nil || IsValidLegacyMetricName(l.GetValue()) {
					escaped.Label = append(escaped.Label, l)
					continue
				}
				escaped.Label = append(escaped.Label, &dto.LabelPair{
					Name:  proto.String(MetricNameLabel),
					Value: proto.String(EscapeName(l.GetValue(), scheme)),
				})
				continue
			}
			if l.Name == nil || IsValidLegacyMetricName(l.GetName()) {
				escaped.Label = append(escaped.Label, l)
				continue
			}
			escaped.Label = append(escaped.Label, &dto.LabelPair{
				Name:  proto.String(EscapeName(l.GetName(), scheme)),
				Value: l.Value,
			})
		}
		out.Metric = append(out.Metric, escaped)
	}
	return out
}

func metricNeedsEscaping(m *dto.Metric) bool {
	for _, l := range m.Label {
		if l.GetName() == MetricNameLabel && !IsValidLegacyMetricName(l.GetValue()) {
			return true
		}
		if !IsValidLegacyMetricName(l.GetName()) {
			return true
		}
	}
	return false
}

// EscapeName escapes the incoming name according to the provided escaping
// scheme. Depending on the rules of escaping, this may cause no change in the
// string that is returned. (Especially NoEscaping, which by definition is a
// noop). This function does not do any validation of the name.
func EscapeName(name string, scheme EscapingScheme) string {
	if len(name) == 0 {
		return name
	}
	var escaped strings.Builder
	switch scheme {
	case NoEscaping:
		return name
	case UnderscoreEscaping:
		if IsValidLegacyMetricName(name) {
			return name
		}
		for i, b := range name {
			if isValidLegacyRune(b, i) {
				escaped.WriteRune(b)
			} else {
				escaped.WriteRune('_')
			}
		}
		return escaped.String()
	case DotsEscaping:
		// Do not early return for legacy valid names, we still escape underscores.
		for i, b := range name {
			switch {
			case b == '_':
				escaped.WriteString("__")
			case b == '.':
				escaped.WriteString("_dot_")
			case isValidLegacyRune(b, i):
				escaped.WriteRune(b)
			default:
				escaped.WriteString("__")
			}
		}
		return escaped.String()
	case ValueEncodingEscaping:
		if IsValidLegacyMetricName(name) {
			return name
		}
		escaped.WriteString("U__")
		for i, b := range name {
			switch {
			case b == '_':
				escaped.WriteString("__")
			case isValidLegacyRune(b, i):
				escaped.WriteRune(b)
			case !utf8.ValidRune(b):
				escaped.WriteString("_FFFD_")
			default:
				escaped.WriteRune('_')
				escaped.WriteString(strconv.FormatInt(int64(b), 16))
				escaped.WriteRune('_')
			}
		}
		return escaped.String()
	default:
		panic(fmt.Sprintf("invalid escaping scheme %d", scheme))
	}
}

// lower function taken from strconv.atoi.
func lower(c byte) byte {
	return c | ('x' - 'X')
}

// UnescapeName unescapes the incoming name according to the provided escaping
// scheme if possible. Some schemes are partially or totally non-roundtripable.
// If any error is enountered, returns the original input.
func UnescapeName(name string, scheme EscapingScheme) string {
	if len(name) == 0 {
		return name
	}
	switch scheme {
	case NoEscaping:
		return name
	case UnderscoreEscaping:
		// It is not possible to unescape from underscore replacement.
		return name
	case DotsEscaping:
		name = strings.ReplaceAll(name, "_dot_", ".")
		name = strings.ReplaceAll(name, "__", "_")
		return name
	case ValueEncodingEscaping:
		escapedName, found := strings.CutPrefix(name, "U__")
		if !found {
			return name
		}

		var unescaped strings.Builder
	TOP:
		for i := 0; i < len(escapedName); i++ {
			// All non-underscores are treated normally.
			if escapedName[i] != '_' {
				unescaped.WriteByte(escapedName[i])
				continue
			}
			i++
			if i >= len(escapedName) {
				return name
			}
			// A double underscore is a single underscore.
			if escapedName[i] == '_' {
				unescaped.WriteByte('_')
				continue
			}
			// We think we are in a UTF-8 code, process it.
			var utf8Val uint
			for j := 0; i < len(escapedName); j++ {
				// This is too many characters for a utf8 value based on the MaxRune
				// value of '\U0010FFFF'.
				if j >= 6 {
					return name
				}
				// Found a closing underscore, convert to a rune, check validity, and append.
				if escapedName[i] == '_' {
					utf8Rune := rune(utf8Val)
					if !utf8.ValidRune(utf8Rune) {
						return name
					}
					unescaped.WriteRune(utf8Rune)
					continue TOP
				}
				r := lower(escapedName[i])
				utf8Val *= 16
				switch {
				case r >= '0' && r <= '9':
					utf8Val += uint(r) - '0'
				case r >= 'a' && r <= 'f':
					utf8Val += uint(r) - 'a' + 10
				default:
					return name
				}
				i++
			}
			// Didn't find closing underscore, invalid.
			return name
		}
		return unescaped.String()
	default:
		panic(fmt.Sprintf("invalid escaping scheme %d", scheme))
	}
}

func isValidLegacyRune(b rune, i int) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b == '_' || b == ':' || (b >= '0' && b <= '9' && i > 0)
}

func (e EscapingScheme) String() string {
	switch e {
	case NoEscaping:
		return AllowUTF8
	case UnderscoreEscaping:
		return EscapeUnderscores
	case DotsEscaping:
		return EscapeDots
	case ValueEncodingEscaping:
		return EscapeValues
	default:
		panic(fmt.Sprintf("unknown format scheme %d", e))
	}
}

func ToEscapingScheme(s string) (EscapingScheme, error) {
	if s == "" {
		return NoEscaping, errors.New("got empty string instead of escaping scheme")
	}
	switch s {
	case AllowUTF8:
		return NoEscaping, nil
	case EscapeUnderscores:
		return UnderscoreEscaping, nil
	case EscapeDots:
		return DotsEscaping, nil
	case EscapeValues:
		return ValueEncodingEscaping, nil
	default:
		return NoEscaping, fmt.Errorf("unknown format scheme %s", s)
	}
}
