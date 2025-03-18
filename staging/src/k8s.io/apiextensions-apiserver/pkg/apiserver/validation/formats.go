/*
Copyright 2019 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"strings"
	"sync"

	"golang.org/x/sync/singleflight"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// supportedVersionedFormats tracks the formats supported by CRD schemas, and the version at which support was introduced.
// Formats in CRD schemas are ignored when used in versions where they are not supported.
var supportedVersionedFormats = []versionedFormats{
	{
		introducedVersion: version.MajorMinor(1, 0),
		formats: sets.New(
			"bsonobjectid", // bson object ID
			"uri",          // an URI as parsed by Golang net/url.ParseRequestURI
			"email",        // an email address as parsed by Golang net/mail.ParseAddress
			"hostname",     // a valid representation for an Internet host name, as defined by RFC 1034, section 3.1 [RFC1034].
			"ipv4",         // an IPv4 IP as parsed by Golang net.ParseIP
			"ipv6",         // an IPv6 IP as parsed by Golang net.ParseIP
			"cidr",         // a CIDR as parsed by Golang net.ParseCIDR
			"mac",          // a MAC address as parsed by Golang net.ParseMAC
			"uuid",         // an UUID that allows uppercase defined by the regex (?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$
			"uuid3",        // an UUID3 that allows uppercase defined by the regex (?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?3[0-9a-f]{3}-?[0-9a-f]{4}-?[0-9a-f]{12}$
			"uuid4",        // an UUID4 that allows uppercase defined by the regex (?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?4[0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}$
			"uuid5",        // an UUID6 that allows uppercase defined by the regex (?i)^[0-9a-f]{8}-?[0-9a-f]{4}-?5[0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}$
			"isbn",         // an ISBN10 or ISBN13 number string like "0321751043" or "978-0321751041"
			"isbn10",       // an ISBN10 number string like "0321751043"
			"isbn13",       // an ISBN13 number string like "978-0321751041"
			"creditcard",   // a credit card number defined by the regex ^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\\d{3})\\d{11})$ with any non digit characters mixed in
			"ssn",          // a U.S. social security number following the regex ^\\d{3}[- ]?\\d{2}[- ]?\\d{4}$
			"hexcolor",     // an hexadecimal color code like "#FFFFFF", following the regex ^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$
			"rgbcolor",     // an RGB color code like rgb like "rgb(255,255,2559"
			"byte",         // base64 encoded binary data
			"password",     // any kind of string
			"date",         // a date string like "2006-01-02" as defined by full-date in RFC3339
			"duration",     // a duration string like "22 ns" as parsed by Golang time.ParseDuration or compatible with Scala duration format
			"datetime",     // a date time string like "2014-12-15T19:30:20.000Z" as defined by date-time in RFC3339
		),
	},
}

// StripUnsupportedFormatsPostProcess sets unsupported formats to empty string.
// Only supports formats supported by all known version of Kubernetes.
// Deprecated: Use StripUnsupportedFormatsPostProcessorForVersion instead.
func StripUnsupportedFormatsPostProcess(s *spec.Schema) error {
	return legacyPostProcessor(s)
}

// StripUnsupportedFormatsPostProcessorForVersion determines the supported formats at the given compatibility version and
// sets unsupported formats to empty string.
func StripUnsupportedFormatsPostProcessorForVersion(compatibilityVersion *version.Version) func(s *spec.Schema) error {
	return func(s *spec.Schema) error {
		if len(s.Format) == 0 {
			return nil
		}

		normalized := strings.ReplaceAll(s.Format, "-", "") // go-openapi default format name normalization
		if !supportedFormatsAtVersion(compatibilityVersion).supported.Has(normalized) {
			s.Format = ""
		}

		return nil
	}
}

type versionedFormats struct {
	introducedVersion *version.Version
	formats           sets.Set[string]
}
type supportedFormats struct {
	compatibilityVersion *version.Version
	// supported is a set of formats validated at compatibilityVersion of Kubernetes.
	supported sets.Set[string]
}

var cacheFormatSets = true

func supportedFormatsAtVersion(ver *version.Version) *supportedFormats {
	key := fmt.Sprintf("%d.%d", ver.Major(), ver.Minor())
	var entry interface{}
	if entry, ok := baseEnvs.Load(key); ok {
		return entry.(*supportedFormats)
	}
	entry, _, _ = baseEnvsSingleflight.Do(key, func() (interface{}, error) {
		entry := newFormatsAtVersion(ver, supportedVersionedFormats)
		if cacheFormatSets {
			baseEnvs.Store(key, entry)
		}
		return entry, nil
	})
	return entry.(*supportedFormats)
}

func newFormatsAtVersion(ver *version.Version, versionedFormats []versionedFormats) *supportedFormats {
	result := &supportedFormats{
		compatibilityVersion: ver,
		supported:            sets.New[string](),
	}
	for _, vf := range versionedFormats {
		if ver.AtLeast(vf.introducedVersion) {
			result.supported = result.supported.Union(vf.formats)

		}
	}
	return result
}

var (
	baseEnvs             = sync.Map{}
	baseEnvsSingleflight = &singleflight.Group{}
)

var legacyPostProcessor = StripUnsupportedFormatsPostProcessorForVersion(version.MajorMinor(1, 0))
