package ldaputil

import (
	"encoding/base64"
	"strings"

	"gopkg.in/ldap.v2"
)

// GetAttributeValue finds the first attribute of those given that the LDAP entry has, and
// returns it. GetAttributeValue is able to query the DN as well as Attributes of the LDAP entry.
// If no value is found, the empty string is returned.
func GetAttributeValue(entry *ldap.Entry, attributes []string) string {
	for _, k := range attributes {
		// Ignore empty attributes
		if len(k) == 0 {
			continue
		}
		// Special-case DN, since it's not an attribute
		if strings.ToLower(k) == "dn" {
			return entry.DN
		}
		// Otherwise get an attribute and return it if present
		if v := entry.GetAttributeValue(k); len(v) > 0 {
			return v
		}
	}
	return ""
}

func GetRawAttributeValue(entry *ldap.Entry, attributes []string) string {
	for _, k := range attributes {
		// Ignore empty attributes
		if len(k) == 0 {
			continue
		}
		// Special-case DN, since it's not an attribute
		if strings.ToLower(k) == "dn" {
			return base64.RawURLEncoding.EncodeToString([]byte(entry.DN))
		}
		// Otherwise get an attribute and return it if present
		if v := entry.GetRawAttributeValue(k); len(v) > 0 {
			return base64.RawURLEncoding.EncodeToString(v)
		}
	}
	return ""
}
