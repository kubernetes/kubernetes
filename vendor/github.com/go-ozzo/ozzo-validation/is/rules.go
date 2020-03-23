// Copyright 2016 Qiang Xue. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

// Package is provides a list of commonly used string validation rules.
package is

import (
	"regexp"
	"unicode"

	"github.com/asaskevich/govalidator"
	"github.com/go-ozzo/ozzo-validation"
)

var (
	// Email validates if a string is an email or not.
	Email = validation.NewStringRule(govalidator.IsEmail, "must be a valid email address")
	// URL validates if a string is a valid URL
	URL = validation.NewStringRule(govalidator.IsURL, "must be a valid URL")
	// RequestURL validates if a string is a valid request URL
	RequestURL = validation.NewStringRule(govalidator.IsRequestURL, "must be a valid request URL")
	// RequestURI validates if a string is a valid request URI
	RequestURI = validation.NewStringRule(govalidator.IsRequestURI, "must be a valid request URI")
	// Alpha validates if a string contains English letters only (a-zA-Z)
	Alpha = validation.NewStringRule(govalidator.IsAlpha, "must contain English letters only")
	// Digit validates if a string contains digits only (0-9)
	Digit = validation.NewStringRule(isDigit, "must contain digits only")
	// Alphanumeric validates if a string contains English letters and digits only (a-zA-Z0-9)
	Alphanumeric = validation.NewStringRule(govalidator.IsAlphanumeric, "must contain English letters and digits only")
	// UTFLetter validates if a string contains unicode letters only
	UTFLetter = validation.NewStringRule(govalidator.IsUTFLetter, "must contain unicode letter characters only")
	// UTFDigit validates if a string contains unicode decimal digits only
	UTFDigit = validation.NewStringRule(govalidator.IsUTFDigit, "must contain unicode decimal digits only")
	// UTFLetterNumeric validates if a string contains unicode letters and numbers only
	UTFLetterNumeric = validation.NewStringRule(govalidator.IsUTFLetterNumeric, "must contain unicode letters and numbers only")
	// UTFNumeric validates if a string contains unicode number characters (category N) only
	UTFNumeric = validation.NewStringRule(isUTFNumeric, "must contain unicode number characters only")
	// LowerCase validates if a string contains lower case unicode letters only
	LowerCase = validation.NewStringRule(govalidator.IsLowerCase, "must be in lower case")
	// UpperCase validates if a string contains upper case unicode letters only
	UpperCase = validation.NewStringRule(govalidator.IsUpperCase, "must be in upper case")
	// Hexadecimal validates if a string is a valid hexadecimal number
	Hexadecimal = validation.NewStringRule(govalidator.IsHexadecimal, "must be a valid hexadecimal number")
	// HexColor validates if a string is a valid hexadecimal color code
	HexColor = validation.NewStringRule(govalidator.IsHexcolor, "must be a valid hexadecimal color code")
	// RGBColor validates if a string is a valid RGB color in the form of rgb(R, G, B)
	RGBColor = validation.NewStringRule(govalidator.IsRGBcolor, "must be a valid RGB color code")
	// Int validates if a string is a valid integer number
	Int = validation.NewStringRule(govalidator.IsInt, "must be an integer number")
	// Float validates if a string is a floating point number
	Float = validation.NewStringRule(govalidator.IsFloat, "must be a floating point number")
	// UUIDv3 validates if a string is a valid version 3 UUID
	UUIDv3 = validation.NewStringRule(govalidator.IsUUIDv3, "must be a valid UUID v3")
	// UUIDv4 validates if a string is a valid version 4 UUID
	UUIDv4 = validation.NewStringRule(govalidator.IsUUIDv4, "must be a valid UUID v4")
	// UUIDv5 validates if a string is a valid version 5 UUID
	UUIDv5 = validation.NewStringRule(govalidator.IsUUIDv5, "must be a valid UUID v5")
	// UUID validates if a string is a valid UUID
	UUID = validation.NewStringRule(govalidator.IsUUID, "must be a valid UUID")
	// CreditCard validates if a string is a valid credit card number
	CreditCard = validation.NewStringRule(govalidator.IsCreditCard, "must be a valid credit card number")
	// ISBN10 validates if a string is an ISBN version 10
	ISBN10 = validation.NewStringRule(govalidator.IsISBN10, "must be a valid ISBN-10")
	// ISBN13 validates if a string is an ISBN version 13
	ISBN13 = validation.NewStringRule(govalidator.IsISBN13, "must be a valid ISBN-13")
	// ISBN validates if a string is an ISBN (either version 10 or 13)
	ISBN = validation.NewStringRule(isISBN, "must be a valid ISBN")
	// JSON validates if a string is in valid JSON format
	JSON = validation.NewStringRule(govalidator.IsJSON, "must be in valid JSON format")
	// ASCII validates if a string contains ASCII characters only
	ASCII = validation.NewStringRule(govalidator.IsASCII, "must contain ASCII characters only")
	// PrintableASCII validates if a string contains printable ASCII characters only
	PrintableASCII = validation.NewStringRule(govalidator.IsPrintableASCII, "must contain printable ASCII characters only")
	// Multibyte validates if a string contains multibyte characters
	Multibyte = validation.NewStringRule(govalidator.IsMultibyte, "must contain multibyte characters")
	// FullWidth validates if a string contains full-width characters
	FullWidth = validation.NewStringRule(govalidator.IsFullWidth, "must contain full-width characters")
	// HalfWidth validates if a string contains half-width characters
	HalfWidth = validation.NewStringRule(govalidator.IsHalfWidth, "must contain half-width characters")
	// VariableWidth validates if a string contains both full-width and half-width characters
	VariableWidth = validation.NewStringRule(govalidator.IsVariableWidth, "must contain both full-width and half-width characters")
	// Base64 validates if a string is encoded in Base64
	Base64 = validation.NewStringRule(govalidator.IsBase64, "must be encoded in Base64")
	// DataURI validates if a string is a valid base64-encoded data URI
	DataURI = validation.NewStringRule(govalidator.IsDataURI, "must be a Base64-encoded data URI")
	// E164 validates if a string is a valid ISO3166 Alpha 2 country code
	E164 = validation.NewStringRule(isE164Number, "must be a valid E164 number")
	// CountryCode2 validates if a string is a valid ISO3166 Alpha 2 country code
	CountryCode2 = validation.NewStringRule(govalidator.IsISO3166Alpha2, "must be a valid two-letter country code")
	// CountryCode3 validates if a string is a valid ISO3166 Alpha 3 country code
	CountryCode3 = validation.NewStringRule(govalidator.IsISO3166Alpha3, "must be a valid three-letter country code")
	// DialString validates if a string is a valid dial string that can be passed to Dial()
	DialString = validation.NewStringRule(govalidator.IsDialString, "must be a valid dial string")
	// MAC validates if a string is a MAC address
	MAC = validation.NewStringRule(govalidator.IsMAC, "must be a valid MAC address")
	// IP validates if a string is a valid IP address (either version 4 or 6)
	IP = validation.NewStringRule(govalidator.IsIP, "must be a valid IP address")
	// IPv4 validates if a string is a valid version 4 IP address
	IPv4 = validation.NewStringRule(govalidator.IsIPv4, "must be a valid IPv4 address")
	// IPv6 validates if a string is a valid version 6 IP address
	IPv6 = validation.NewStringRule(govalidator.IsIPv6, "must be a valid IPv6 address")
	// Subdomain validates if a string is valid subdomain
	Subdomain = validation.NewStringRule(isSubdomain, "must be a valid subdomain")
	// Domain validates if a string is valid domain
	Domain = validation.NewStringRule(isDomain, "must be a valid domain")
	// DNSName validates if a string is valid DNS name
	DNSName = validation.NewStringRule(govalidator.IsDNSName, "must be a valid DNS name")
	// Host validates if a string is a valid IP (both v4 and v6) or a valid DNS name
	Host = validation.NewStringRule(govalidator.IsHost, "must be a valid IP address or DNS name")
	// Port validates if a string is a valid port number
	Port = validation.NewStringRule(govalidator.IsPort, "must be a valid port number")
	// MongoID validates if a string is a valid Mongo ID
	MongoID = validation.NewStringRule(govalidator.IsMongoID, "must be a valid hex-encoded MongoDB ObjectId")
	// Latitude validates if a string is a valid latitude
	Latitude = validation.NewStringRule(govalidator.IsLatitude, "must be a valid latitude")
	// Longitude validates if a string is a valid longitude
	Longitude = validation.NewStringRule(govalidator.IsLongitude, "must be a valid longitude")
	// SSN validates if a string is a social security number (SSN)
	SSN = validation.NewStringRule(govalidator.IsSSN, "must be a valid social security number")
	// Semver validates if a string is a valid semantic version
	Semver = validation.NewStringRule(govalidator.IsSemver, "must be a valid semantic version")
)

var (
	reDigit = regexp.MustCompile("^[0-9]+$")
	// Subdomain regex source: https://stackoverflow.com/a/7933253
	reSubdomain = regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?$`)
)

func isISBN(value string) bool {
	return govalidator.IsISBN(value, 10) || govalidator.IsISBN(value, 13)
}

func isDigit(value string) bool {
	return reDigit.MatchString(value)
}

func isE164Number(value string) bool {
	// E164 regex source: https://stackoverflow.com/a/23299989
	reE164 := regexp.MustCompile(`^\+?[1-9]\d{1,14}$`)
	return reE164.MatchString(value)
}

func isSubdomain(value string) bool {
	// Subdomain regex source: https://stackoverflow.com/a/7933253
	reSubdomain := regexp.MustCompile(`^[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?$`)
	return reSubdomain.MatchString(value)
}

func isDomain(value string) bool {
	if len(value) > 255 {
		return false
	}

	// Domain regex source: https://stackoverflow.com/a/7933253
	// Slightly modified: Removed 255 max length validation since Go regex does not
	// support lookarounds. More info: https://stackoverflow.com/a/38935027
	reDomain := regexp.MustCompile(`^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{1,63}| xn--[a-z0-9]{1,59})$`)

	return reDomain.MatchString(value)
}

func isUTFNumeric(value string) bool {
	for _, c := range value {
		if unicode.IsNumber(c) == false {
			return false
		}
	}
	return true
}
