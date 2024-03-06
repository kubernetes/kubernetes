// Package govalidator is package of validators and sanitizers for strings, structs and collections.
package govalidator

import (
	"bytes"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"net"
	"net/url"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

var (
	fieldsRequiredByDefault bool
	nilPtrAllowedByRequired = false
	notNumberRegexp         = regexp.MustCompile("[^0-9]+")
	whiteSpacesAndMinus     = regexp.MustCompile(`[\s-]+`)
	paramsRegexp            = regexp.MustCompile(`\(.*\)$`)
)

const maxURLRuneCount = 2083
const minURLRuneCount = 3
const rfc3339WithoutZone = "2006-01-02T15:04:05"

// SetFieldsRequiredByDefault causes validation to fail when struct fields
// do not include validations or are not explicitly marked as exempt (using `valid:"-"` or `valid:"email,optional"`).
// This struct definition will fail govalidator.ValidateStruct() (and the field values do not matter):
//     type exampleStruct struct {
//         Name  string ``
//         Email string `valid:"email"`
// This, however, will only fail when Email is empty or an invalid email address:
//     type exampleStruct2 struct {
//         Name  string `valid:"-"`
//         Email string `valid:"email"`
// Lastly, this will only fail when Email is an invalid email address but not when it's empty:
//     type exampleStruct2 struct {
//         Name  string `valid:"-"`
//         Email string `valid:"email,optional"`
func SetFieldsRequiredByDefault(value bool) {
	fieldsRequiredByDefault = value
}

// SetNilPtrAllowedByRequired causes validation to pass for nil ptrs when a field is set to required.
// The validation will still reject ptr fields in their zero value state. Example with this enabled:
//     type exampleStruct struct {
//         Name  *string `valid:"required"`
// With `Name` set to "", this will be considered invalid input and will cause a validation error.
// With `Name` set to nil, this will be considered valid by validation.
// By default this is disabled.
func SetNilPtrAllowedByRequired(value bool) {
	nilPtrAllowedByRequired = value
}

// IsEmail checks if the string is an email.
func IsEmail(str string) bool {
	// TODO uppercase letters are not supported
	return rxEmail.MatchString(str)
}

// IsExistingEmail checks if the string is an email of existing domain
func IsExistingEmail(email string) bool {

	if len(email) < 6 || len(email) > 254 {
		return false
	}
	at := strings.LastIndex(email, "@")
	if at <= 0 || at > len(email)-3 {
		return false
	}
	user := email[:at]
	host := email[at+1:]
	if len(user) > 64 {
		return false
	}
	switch host {
	case "localhost", "example.com":
		return true
	}
	if userDotRegexp.MatchString(user) || !userRegexp.MatchString(user) || !hostRegexp.MatchString(host) {
		return false
	}
	if _, err := net.LookupMX(host); err != nil {
		if _, err := net.LookupIP(host); err != nil {
			return false
		}
	}

	return true
}

// IsURL checks if the string is an URL.
func IsURL(str string) bool {
	if str == "" || utf8.RuneCountInString(str) >= maxURLRuneCount || len(str) <= minURLRuneCount || strings.HasPrefix(str, ".") {
		return false
	}
	strTemp := str
	if strings.Contains(str, ":") && !strings.Contains(str, "://") {
		// support no indicated urlscheme but with colon for port number
		// http:// is appended so url.Parse will succeed, strTemp used so it does not impact rxURL.MatchString
		strTemp = "http://" + str
	}
	u, err := url.Parse(strTemp)
	if err != nil {
		return false
	}
	if strings.HasPrefix(u.Host, ".") {
		return false
	}
	if u.Host == "" && (u.Path != "" && !strings.Contains(u.Path, ".")) {
		return false
	}
	return rxURL.MatchString(str)
}

// IsRequestURL checks if the string rawurl, assuming
// it was received in an HTTP request, is a valid
// URL confirm to RFC 3986
func IsRequestURL(rawurl string) bool {
	url, err := url.ParseRequestURI(rawurl)
	if err != nil {
		return false //Couldn't even parse the rawurl
	}
	if len(url.Scheme) == 0 {
		return false //No Scheme found
	}
	return true
}

// IsRequestURI checks if the string rawurl, assuming
// it was received in an HTTP request, is an
// absolute URI or an absolute path.
func IsRequestURI(rawurl string) bool {
	_, err := url.ParseRequestURI(rawurl)
	return err == nil
}

// IsAlpha checks if the string contains only letters (a-zA-Z). Empty string is valid.
func IsAlpha(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxAlpha.MatchString(str)
}

//IsUTFLetter checks if the string contains only unicode letter characters.
//Similar to IsAlpha but for all languages. Empty string is valid.
func IsUTFLetter(str string) bool {
	if IsNull(str) {
		return true
	}

	for _, c := range str {
		if !unicode.IsLetter(c) {
			return false
		}
	}
	return true

}

// IsAlphanumeric checks if the string contains only letters and numbers. Empty string is valid.
func IsAlphanumeric(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxAlphanumeric.MatchString(str)
}

// IsUTFLetterNumeric checks if the string contains only unicode letters and numbers. Empty string is valid.
func IsUTFLetterNumeric(str string) bool {
	if IsNull(str) {
		return true
	}
	for _, c := range str {
		if !unicode.IsLetter(c) && !unicode.IsNumber(c) { //letters && numbers are ok
			return false
		}
	}
	return true

}

// IsNumeric checks if the string contains only numbers. Empty string is valid.
func IsNumeric(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxNumeric.MatchString(str)
}

// IsUTFNumeric checks if the string contains only unicode numbers of any kind.
// Numbers can be 0-9 but also Fractions ¾,Roman Ⅸ and Hangzhou 〩. Empty string is valid.
func IsUTFNumeric(str string) bool {
	if IsNull(str) {
		return true
	}
	if strings.IndexAny(str, "+-") > 0 {
		return false
	}
	if len(str) > 1 {
		str = strings.TrimPrefix(str, "-")
		str = strings.TrimPrefix(str, "+")
	}
	for _, c := range str {
		if !unicode.IsNumber(c) { //numbers && minus sign are ok
			return false
		}
	}
	return true

}

// IsUTFDigit checks if the string contains only unicode radix-10 decimal digits. Empty string is valid.
func IsUTFDigit(str string) bool {
	if IsNull(str) {
		return true
	}
	if strings.IndexAny(str, "+-") > 0 {
		return false
	}
	if len(str) > 1 {
		str = strings.TrimPrefix(str, "-")
		str = strings.TrimPrefix(str, "+")
	}
	for _, c := range str {
		if !unicode.IsDigit(c) { //digits && minus sign are ok
			return false
		}
	}
	return true

}

// IsHexadecimal checks if the string is a hexadecimal number.
func IsHexadecimal(str string) bool {
	return rxHexadecimal.MatchString(str)
}

// IsHexcolor checks if the string is a hexadecimal color.
func IsHexcolor(str string) bool {
	return rxHexcolor.MatchString(str)
}

// IsRGBcolor checks if the string is a valid RGB color in form rgb(RRR, GGG, BBB).
func IsRGBcolor(str string) bool {
	return rxRGBcolor.MatchString(str)
}

// IsLowerCase checks if the string is lowercase. Empty string is valid.
func IsLowerCase(str string) bool {
	if IsNull(str) {
		return true
	}
	return str == strings.ToLower(str)
}

// IsUpperCase checks if the string is uppercase. Empty string is valid.
func IsUpperCase(str string) bool {
	if IsNull(str) {
		return true
	}
	return str == strings.ToUpper(str)
}

// HasLowerCase checks if the string contains at least 1 lowercase. Empty string is valid.
func HasLowerCase(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxHasLowerCase.MatchString(str)
}

// HasUpperCase checks if the string contains as least 1 uppercase. Empty string is valid.
func HasUpperCase(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxHasUpperCase.MatchString(str)
}

// IsInt checks if the string is an integer. Empty string is valid.
func IsInt(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxInt.MatchString(str)
}

// IsFloat checks if the string is a float.
func IsFloat(str string) bool {
	return str != "" && rxFloat.MatchString(str)
}

// IsDivisibleBy checks if the string is a number that's divisible by another.
// If second argument is not valid integer or zero, it's return false.
// Otherwise, if first argument is not valid integer or zero, it's return true (Invalid string converts to zero).
func IsDivisibleBy(str, num string) bool {
	f, _ := ToFloat(str)
	p := int64(f)
	q, _ := ToInt(num)
	if q == 0 {
		return false
	}
	return (p == 0) || (p%q == 0)
}

// IsNull checks if the string is null.
func IsNull(str string) bool {
	return len(str) == 0
}

// IsNotNull checks if the string is not null.
func IsNotNull(str string) bool {
	return !IsNull(str)
}

// HasWhitespaceOnly checks the string only contains whitespace
func HasWhitespaceOnly(str string) bool {
	return len(str) > 0 && rxHasWhitespaceOnly.MatchString(str)
}

// HasWhitespace checks if the string contains any whitespace
func HasWhitespace(str string) bool {
	return len(str) > 0 && rxHasWhitespace.MatchString(str)
}

// IsByteLength checks if the string's length (in bytes) falls in a range.
func IsByteLength(str string, min, max int) bool {
	return len(str) >= min && len(str) <= max
}

// IsUUIDv3 checks if the string is a UUID version 3.
func IsUUIDv3(str string) bool {
	return rxUUID3.MatchString(str)
}

// IsUUIDv4 checks if the string is a UUID version 4.
func IsUUIDv4(str string) bool {
	return rxUUID4.MatchString(str)
}

// IsUUIDv5 checks if the string is a UUID version 5.
func IsUUIDv5(str string) bool {
	return rxUUID5.MatchString(str)
}

// IsUUID checks if the string is a UUID (version 3, 4 or 5).
func IsUUID(str string) bool {
	return rxUUID.MatchString(str)
}

// Byte to index table for O(1) lookups when unmarshaling.
// We use 0xFF as sentinel value for invalid indexes.
var ulidDec = [...]byte{
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x01,
	0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
	0x0F, 0x10, 0x11, 0xFF, 0x12, 0x13, 0xFF, 0x14, 0x15, 0xFF,
	0x16, 0x17, 0x18, 0x19, 0x1A, 0xFF, 0x1B, 0x1C, 0x1D, 0x1E,
	0x1F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x0A, 0x0B, 0x0C,
	0x0D, 0x0E, 0x0F, 0x10, 0x11, 0xFF, 0x12, 0x13, 0xFF, 0x14,
	0x15, 0xFF, 0x16, 0x17, 0x18, 0x19, 0x1A, 0xFF, 0x1B, 0x1C,
	0x1D, 0x1E, 0x1F, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
	0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
}

// EncodedSize is the length of a text encoded ULID.
const ulidEncodedSize = 26

// IsULID checks if the string is a ULID.
//
// Implementation got from:
//   https://github.com/oklog/ulid (Apache-2.0 License)
//
func IsULID(str string) bool {
	// Check if a base32 encoded ULID is the right length.
	if len(str) != ulidEncodedSize {
		return false
	}

	// Check if all the characters in a base32 encoded ULID are part of the
	// expected base32 character set.
	if ulidDec[str[0]] == 0xFF ||
		ulidDec[str[1]] == 0xFF ||
		ulidDec[str[2]] == 0xFF ||
		ulidDec[str[3]] == 0xFF ||
		ulidDec[str[4]] == 0xFF ||
		ulidDec[str[5]] == 0xFF ||
		ulidDec[str[6]] == 0xFF ||
		ulidDec[str[7]] == 0xFF ||
		ulidDec[str[8]] == 0xFF ||
		ulidDec[str[9]] == 0xFF ||
		ulidDec[str[10]] == 0xFF ||
		ulidDec[str[11]] == 0xFF ||
		ulidDec[str[12]] == 0xFF ||
		ulidDec[str[13]] == 0xFF ||
		ulidDec[str[14]] == 0xFF ||
		ulidDec[str[15]] == 0xFF ||
		ulidDec[str[16]] == 0xFF ||
		ulidDec[str[17]] == 0xFF ||
		ulidDec[str[18]] == 0xFF ||
		ulidDec[str[19]] == 0xFF ||
		ulidDec[str[20]] == 0xFF ||
		ulidDec[str[21]] == 0xFF ||
		ulidDec[str[22]] == 0xFF ||
		ulidDec[str[23]] == 0xFF ||
		ulidDec[str[24]] == 0xFF ||
		ulidDec[str[25]] == 0xFF {
		return false
	}

	// Check if the first character in a base32 encoded ULID will overflow. This
	// happens because the base32 representation encodes 130 bits, while the
	// ULID is only 128 bits.
	//
	// See https://github.com/oklog/ulid/issues/9 for details.
	if str[0] > '7' {
		return false
	}
	return true
}

// IsCreditCard checks if the string is a credit card.
func IsCreditCard(str string) bool {
	sanitized := whiteSpacesAndMinus.ReplaceAllString(str, "")
	if !rxCreditCard.MatchString(sanitized) {
		return false
	}
	
	number, _ := ToInt(sanitized)
	number, lastDigit := number / 10, number % 10	

	var sum int64
	for i:=0; number > 0; i++ {
		digit := number % 10
		
		if i % 2 == 0 {
			digit *= 2
			if digit > 9 {
				digit -= 9
			}
		}
		
		sum += digit
		number = number / 10
	}
	
	return (sum + lastDigit) % 10 == 0
}

// IsISBN10 checks if the string is an ISBN version 10.
func IsISBN10(str string) bool {
	return IsISBN(str, 10)
}

// IsISBN13 checks if the string is an ISBN version 13.
func IsISBN13(str string) bool {
	return IsISBN(str, 13)
}

// IsISBN checks if the string is an ISBN (version 10 or 13).
// If version value is not equal to 10 or 13, it will be checks both variants.
func IsISBN(str string, version int) bool {
	sanitized := whiteSpacesAndMinus.ReplaceAllString(str, "")
	var checksum int32
	var i int32
	if version == 10 {
		if !rxISBN10.MatchString(sanitized) {
			return false
		}
		for i = 0; i < 9; i++ {
			checksum += (i + 1) * int32(sanitized[i]-'0')
		}
		if sanitized[9] == 'X' {
			checksum += 10 * 10
		} else {
			checksum += 10 * int32(sanitized[9]-'0')
		}
		if checksum%11 == 0 {
			return true
		}
		return false
	} else if version == 13 {
		if !rxISBN13.MatchString(sanitized) {
			return false
		}
		factor := []int32{1, 3}
		for i = 0; i < 12; i++ {
			checksum += factor[i%2] * int32(sanitized[i]-'0')
		}
		return (int32(sanitized[12]-'0'))-((10-(checksum%10))%10) == 0
	}
	return IsISBN(str, 10) || IsISBN(str, 13)
}

// IsJSON checks if the string is valid JSON (note: uses json.Unmarshal).
func IsJSON(str string) bool {
	var js json.RawMessage
	return json.Unmarshal([]byte(str), &js) == nil
}

// IsMultibyte checks if the string contains one or more multibyte chars. Empty string is valid.
func IsMultibyte(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxMultibyte.MatchString(str)
}

// IsASCII checks if the string contains ASCII chars only. Empty string is valid.
func IsASCII(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxASCII.MatchString(str)
}

// IsPrintableASCII checks if the string contains printable ASCII chars only. Empty string is valid.
func IsPrintableASCII(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxPrintableASCII.MatchString(str)
}

// IsFullWidth checks if the string contains any full-width chars. Empty string is valid.
func IsFullWidth(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxFullWidth.MatchString(str)
}

// IsHalfWidth checks if the string contains any half-width chars. Empty string is valid.
func IsHalfWidth(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxHalfWidth.MatchString(str)
}

// IsVariableWidth checks if the string contains a mixture of full and half-width chars. Empty string is valid.
func IsVariableWidth(str string) bool {
	if IsNull(str) {
		return true
	}
	return rxHalfWidth.MatchString(str) && rxFullWidth.MatchString(str)
}

// IsBase64 checks if a string is base64 encoded.
func IsBase64(str string) bool {
	return rxBase64.MatchString(str)
}

// IsFilePath checks is a string is Win or Unix file path and returns it's type.
func IsFilePath(str string) (bool, int) {
	if rxWinPath.MatchString(str) {
		//check windows path limit see:
		//  http://msdn.microsoft.com/en-us/library/aa365247(VS.85).aspx#maxpath
		if len(str[3:]) > 32767 {
			return false, Win
		}
		return true, Win
	} else if rxUnixPath.MatchString(str) {
		return true, Unix
	}
	return false, Unknown
}

//IsWinFilePath checks both relative & absolute paths in Windows
func IsWinFilePath(str string) bool {
	if rxARWinPath.MatchString(str) {
		//check windows path limit see:
		//  http://msdn.microsoft.com/en-us/library/aa365247(VS.85).aspx#maxpath
		if len(str[3:]) > 32767 {
			return false
		}
		return true
	}
	return false
}

//IsUnixFilePath checks both relative & absolute paths in Unix
func IsUnixFilePath(str string) bool {
	if rxARUnixPath.MatchString(str) {
		return true
	}
	return false
}

// IsDataURI checks if a string is base64 encoded data URI such as an image
func IsDataURI(str string) bool {
	dataURI := strings.Split(str, ",")
	if !rxDataURI.MatchString(dataURI[0]) {
		return false
	}
	return IsBase64(dataURI[1])
}

// IsMagnetURI checks if a string is valid magnet URI
func IsMagnetURI(str string) bool {
	return rxMagnetURI.MatchString(str)
}

// IsISO3166Alpha2 checks if a string is valid two-letter country code
func IsISO3166Alpha2(str string) bool {
	for _, entry := range ISO3166List {
		if str == entry.Alpha2Code {
			return true
		}
	}
	return false
}

// IsISO3166Alpha3 checks if a string is valid three-letter country code
func IsISO3166Alpha3(str string) bool {
	for _, entry := range ISO3166List {
		if str == entry.Alpha3Code {
			return true
		}
	}
	return false
}

// IsISO693Alpha2 checks if a string is valid two-letter language code
func IsISO693Alpha2(str string) bool {
	for _, entry := range ISO693List {
		if str == entry.Alpha2Code {
			return true
		}
	}
	return false
}

// IsISO693Alpha3b checks if a string is valid three-letter language code
func IsISO693Alpha3b(str string) bool {
	for _, entry := range ISO693List {
		if str == entry.Alpha3bCode {
			return true
		}
	}
	return false
}

// IsDNSName will validate the given string as a DNS name
func IsDNSName(str string) bool {
	if str == "" || len(strings.Replace(str, ".", "", -1)) > 255 {
		// constraints already violated
		return false
	}
	return !IsIP(str) && rxDNSName.MatchString(str)
}

// IsHash checks if a string is a hash of type algorithm.
// Algorithm is one of ['md4', 'md5', 'sha1', 'sha256', 'sha384', 'sha512', 'ripemd128', 'ripemd160', 'tiger128', 'tiger160', 'tiger192', 'crc32', 'crc32b']
func IsHash(str string, algorithm string) bool {
	var len string
	algo := strings.ToLower(algorithm)

	if algo == "crc32" || algo == "crc32b" {
		len = "8"
	} else if algo == "md5" || algo == "md4" || algo == "ripemd128" || algo == "tiger128" {
		len = "32"
	} else if algo == "sha1" || algo == "ripemd160" || algo == "tiger160" {
		len = "40"
	} else if algo == "tiger192" {
		len = "48"
	} else if algo == "sha3-224" {
		len = "56"
	} else if algo == "sha256" || algo == "sha3-256" {
		len = "64"
	} else if algo == "sha384" || algo == "sha3-384" {
		len = "96"
	} else if algo == "sha512" || algo == "sha3-512" {
		len = "128"
	} else {
		return false
	}

	return Matches(str, "^[a-f0-9]{"+len+"}$")
}

// IsSHA3224 checks is a string is a SHA3-224 hash. Alias for `IsHash(str, "sha3-224")`
func IsSHA3224(str string) bool {
	return IsHash(str, "sha3-224")
}

// IsSHA3256 checks is a string is a SHA3-256 hash. Alias for `IsHash(str, "sha3-256")`
func IsSHA3256(str string) bool {
	return IsHash(str, "sha3-256")
}

// IsSHA3384 checks is a string is a SHA3-384 hash. Alias for `IsHash(str, "sha3-384")`
func IsSHA3384(str string) bool {
	return IsHash(str, "sha3-384")
}

// IsSHA3512 checks is a string is a SHA3-512 hash. Alias for `IsHash(str, "sha3-512")`
func IsSHA3512(str string) bool {
	return IsHash(str, "sha3-512")
}

// IsSHA512 checks is a string is a SHA512 hash. Alias for `IsHash(str, "sha512")`
func IsSHA512(str string) bool {
	return IsHash(str, "sha512")
}

// IsSHA384 checks is a string is a SHA384 hash. Alias for `IsHash(str, "sha384")`
func IsSHA384(str string) bool {
	return IsHash(str, "sha384")
}

// IsSHA256 checks is a string is a SHA256 hash. Alias for `IsHash(str, "sha256")`
func IsSHA256(str string) bool {
	return IsHash(str, "sha256")
}

// IsTiger192 checks is a string is a Tiger192 hash. Alias for `IsHash(str, "tiger192")`
func IsTiger192(str string) bool {
	return IsHash(str, "tiger192")
}

// IsTiger160 checks is a string is a Tiger160 hash. Alias for `IsHash(str, "tiger160")`
func IsTiger160(str string) bool {
	return IsHash(str, "tiger160")
}

// IsRipeMD160 checks is a string is a RipeMD160 hash. Alias for `IsHash(str, "ripemd160")`
func IsRipeMD160(str string) bool {
	return IsHash(str, "ripemd160")
}

// IsSHA1 checks is a string is a SHA-1 hash. Alias for `IsHash(str, "sha1")`
func IsSHA1(str string) bool {
	return IsHash(str, "sha1")
}

// IsTiger128 checks is a string is a Tiger128 hash. Alias for `IsHash(str, "tiger128")`
func IsTiger128(str string) bool {
	return IsHash(str, "tiger128")
}

// IsRipeMD128 checks is a string is a RipeMD128 hash. Alias for `IsHash(str, "ripemd128")`
func IsRipeMD128(str string) bool {
	return IsHash(str, "ripemd128")
}

// IsCRC32 checks is a string is a CRC32 hash. Alias for `IsHash(str, "crc32")`
func IsCRC32(str string) bool {
	return IsHash(str, "crc32")
}

// IsCRC32b checks is a string is a CRC32b hash. Alias for `IsHash(str, "crc32b")`
func IsCRC32b(str string) bool {
	return IsHash(str, "crc32b")
}

// IsMD5 checks is a string is a MD5 hash. Alias for `IsHash(str, "md5")`
func IsMD5(str string) bool {
	return IsHash(str, "md5")
}

// IsMD4 checks is a string is a MD4 hash. Alias for `IsHash(str, "md4")`
func IsMD4(str string) bool {
	return IsHash(str, "md4")
}

// IsDialString validates the given string for usage with the various Dial() functions
func IsDialString(str string) bool {
	if h, p, err := net.SplitHostPort(str); err == nil && h != "" && p != "" && (IsDNSName(h) || IsIP(h)) && IsPort(p) {
		return true
	}

	return false
}

// IsIP checks if a string is either IP version 4 or 6. Alias for `net.ParseIP`
func IsIP(str string) bool {
	return net.ParseIP(str) != nil
}

// IsPort checks if a string represents a valid port
func IsPort(str string) bool {
	if i, err := strconv.Atoi(str); err == nil && i > 0 && i < 65536 {
		return true
	}
	return false
}

// IsIPv4 checks if the string is an IP version 4.
func IsIPv4(str string) bool {
	ip := net.ParseIP(str)
	return ip != nil && strings.Contains(str, ".")
}

// IsIPv6 checks if the string is an IP version 6.
func IsIPv6(str string) bool {
	ip := net.ParseIP(str)
	return ip != nil && strings.Contains(str, ":")
}

// IsCIDR checks if the string is an valid CIDR notiation (IPV4 & IPV6)
func IsCIDR(str string) bool {
	_, _, err := net.ParseCIDR(str)
	return err == nil
}

// IsMAC checks if a string is valid MAC address.
// Possible MAC formats:
// 01:23:45:67:89:ab
// 01:23:45:67:89:ab:cd:ef
// 01-23-45-67-89-ab
// 01-23-45-67-89-ab-cd-ef
// 0123.4567.89ab
// 0123.4567.89ab.cdef
func IsMAC(str string) bool {
	_, err := net.ParseMAC(str)
	return err == nil
}

// IsHost checks if the string is a valid IP (both v4 and v6) or a valid DNS name
func IsHost(str string) bool {
	return IsIP(str) || IsDNSName(str)
}

// IsMongoID checks if the string is a valid hex-encoded representation of a MongoDB ObjectId.
func IsMongoID(str string) bool {
	return rxHexadecimal.MatchString(str) && (len(str) == 24)
}

// IsLatitude checks if a string is valid latitude.
func IsLatitude(str string) bool {
	return rxLatitude.MatchString(str)
}

// IsLongitude checks if a string is valid longitude.
func IsLongitude(str string) bool {
	return rxLongitude.MatchString(str)
}

// IsIMEI checks if a string is valid IMEI
func IsIMEI(str string) bool {
	return rxIMEI.MatchString(str)
}

// IsIMSI checks if a string is valid IMSI
func IsIMSI(str string) bool {
	if !rxIMSI.MatchString(str) {
		return false
	}

	mcc, err := strconv.ParseInt(str[0:3], 10, 32)
	if err != nil {
		return false
	}

	switch mcc {
	case 202, 204, 206, 208, 212, 213, 214, 216, 218, 219:
	case 220, 221, 222, 226, 228, 230, 231, 232, 234, 235:
	case 238, 240, 242, 244, 246, 247, 248, 250, 255, 257:
	case 259, 260, 262, 266, 268, 270, 272, 274, 276, 278:
	case 280, 282, 283, 284, 286, 288, 289, 290, 292, 293:
	case 294, 295, 297, 302, 308, 310, 311, 312, 313, 314:
	case 315, 316, 330, 332, 334, 338, 340, 342, 344, 346:
	case 348, 350, 352, 354, 356, 358, 360, 362, 363, 364:
	case 365, 366, 368, 370, 372, 374, 376, 400, 401, 402:
	case 404, 405, 406, 410, 412, 413, 414, 415, 416, 417:
	case 418, 419, 420, 421, 422, 424, 425, 426, 427, 428:
	case 429, 430, 431, 432, 434, 436, 437, 438, 440, 441:
	case 450, 452, 454, 455, 456, 457, 460, 461, 466, 467:
	case 470, 472, 502, 505, 510, 514, 515, 520, 525, 528:
	case 530, 536, 537, 539, 540, 541, 542, 543, 544, 545:
	case 546, 547, 548, 549, 550, 551, 552, 553, 554, 555:
	case 602, 603, 604, 605, 606, 607, 608, 609, 610, 611:
	case 612, 613, 614, 615, 616, 617, 618, 619, 620, 621:
	case 622, 623, 624, 625, 626, 627, 628, 629, 630, 631:
	case 632, 633, 634, 635, 636, 637, 638, 639, 640, 641:
	case 642, 643, 645, 646, 647, 648, 649, 650, 651, 652:
	case 653, 654, 655, 657, 658, 659, 702, 704, 706, 708:
	case 710, 712, 714, 716, 722, 724, 730, 732, 734, 736:
	case 738, 740, 742, 744, 746, 748, 750, 995:
		return true
	default:
		return false
	}
	return true
}

// IsRsaPublicKey checks if a string is valid public key with provided length
func IsRsaPublicKey(str string, keylen int) bool {
	bb := bytes.NewBufferString(str)
	pemBytes, err := ioutil.ReadAll(bb)
	if err != nil {
		return false
	}
	block, _ := pem.Decode(pemBytes)
	if block != nil && block.Type != "PUBLIC KEY" {
		return false
	}
	var der []byte

	if block != nil {
		der = block.Bytes
	} else {
		der, err = base64.StdEncoding.DecodeString(str)
		if err != nil {
			return false
		}
	}

	key, err := x509.ParsePKIXPublicKey(der)
	if err != nil {
		return false
	}
	pubkey, ok := key.(*rsa.PublicKey)
	if !ok {
		return false
	}
	bitlen := len(pubkey.N.Bytes()) * 8
	return bitlen == int(keylen)
}

// IsRegex checks if a give string is a valid regex with RE2 syntax or not
func IsRegex(str string) bool {
	if _, err := regexp.Compile(str); err == nil {
		return true
	}
	return false
}

func toJSONName(tag string) string {
	if tag == "" {
		return ""
	}

	// JSON name always comes first. If there's no options then split[0] is
	// JSON name, if JSON name is not set, then split[0] is an empty string.
	split := strings.SplitN(tag, ",", 2)

	name := split[0]

	// However it is possible that the field is skipped when
	// (de-)serializing from/to JSON, in which case assume that there is no
	// tag name to use
	if name == "-" {
		return ""
	}
	return name
}

func prependPathToErrors(err error, path string) error {
	switch err2 := err.(type) {
	case Error:
		err2.Path = append([]string{path}, err2.Path...)
		return err2
	case Errors:
		errors := err2.Errors()
		for i, err3 := range errors {
			errors[i] = prependPathToErrors(err3, path)
		}
		return err2
	}
	return err
}

// ValidateArray performs validation according to condition iterator that validates every element of the array
func ValidateArray(array []interface{}, iterator ConditionIterator) bool {
	return Every(array, iterator)
}

// ValidateMap use validation map for fields.
// result will be equal to `false` if there are any errors.
// s is the map containing the data to be validated.
// m is the validation map in the form:
//   map[string]interface{}{"name":"required,alpha","address":map[string]interface{}{"line1":"required,alphanum"}}
func ValidateMap(s map[string]interface{}, m map[string]interface{}) (bool, error) {
	if s == nil {
		return true, nil
	}
	result := true
	var err error
	var errs Errors
	var index int
	val := reflect.ValueOf(s)
	for key, value := range s {
		presentResult := true
		validator, ok := m[key]
		if !ok {
			presentResult = false
			var err error
			err = fmt.Errorf("all map keys has to be present in the validation map; got %s", key)
			err = prependPathToErrors(err, key)
			errs = append(errs, err)
		}
		valueField := reflect.ValueOf(value)
		mapResult := true
		typeResult := true
		structResult := true
		resultField := true
		switch subValidator := validator.(type) {
		case map[string]interface{}:
			var err error
			if v, ok := value.(map[string]interface{}); !ok {
				mapResult = false
				err = fmt.Errorf("map validator has to be for the map type only; got %s", valueField.Type().String())
				err = prependPathToErrors(err, key)
				errs = append(errs, err)
			} else {
				mapResult, err = ValidateMap(v, subValidator)
				if err != nil {
					mapResult = false
					err = prependPathToErrors(err, key)
					errs = append(errs, err)
				}
			}
		case string:
			if (valueField.Kind() == reflect.Struct ||
				(valueField.Kind() == reflect.Ptr && valueField.Elem().Kind() == reflect.Struct)) &&
				subValidator != "-" {
				var err error
				structResult, err = ValidateStruct(valueField.Interface())
				if err != nil {
					err = prependPathToErrors(err, key)
					errs = append(errs, err)
				}
			}
			resultField, err = typeCheck(valueField, reflect.StructField{
				Name:      key,
				PkgPath:   "",
				Type:      val.Type(),
				Tag:       reflect.StructTag(fmt.Sprintf("%s:%q", tagName, subValidator)),
				Offset:    0,
				Index:     []int{index},
				Anonymous: false,
			}, val, nil)
			if err != nil {
				errs = append(errs, err)
			}
		case nil:
			// already handlerd when checked before
		default:
			typeResult = false
			err = fmt.Errorf("map validator has to be either map[string]interface{} or string; got %s", valueField.Type().String())
			err = prependPathToErrors(err, key)
			errs = append(errs, err)
		}
		result = result && presentResult && typeResult && resultField && structResult && mapResult
		index++
	}
	// checks required keys
	requiredResult := true
	for key, value := range m {
		if schema, ok := value.(string); ok {
			tags := parseTagIntoMap(schema)
			if required, ok := tags["required"]; ok {
				if _, ok := s[key]; !ok {
					requiredResult = false
					if required.customErrorMessage != "" {
						err = Error{key, fmt.Errorf(required.customErrorMessage), true, "required", []string{}}
					} else {
						err = Error{key, fmt.Errorf("required field missing"), false, "required", []string{}}
					}
					errs = append(errs, err)
				}
			}
		}
	}

	if len(errs) > 0 {
		err = errs
	}
	return result && requiredResult, err
}

// ValidateStruct use tags for fields.
// result will be equal to `false` if there are any errors.
// todo currently there is no guarantee that errors will be returned in predictable order (tests may to fail)
func ValidateStruct(s interface{}) (bool, error) {
	if s == nil {
		return true, nil
	}
	result := true
	var err error
	val := reflect.ValueOf(s)
	if val.Kind() == reflect.Interface || val.Kind() == reflect.Ptr {
		val = val.Elem()
	}
	// we only accept structs
	if val.Kind() != reflect.Struct {
		return false, fmt.Errorf("function only accepts structs; got %s", val.Kind())
	}
	var errs Errors
	for i := 0; i < val.NumField(); i++ {
		valueField := val.Field(i)
		typeField := val.Type().Field(i)
		if typeField.PkgPath != "" {
			continue // Private field
		}
		structResult := true
		if valueField.Kind() == reflect.Interface {
			valueField = valueField.Elem()
		}
		if (valueField.Kind() == reflect.Struct ||
			(valueField.Kind() == reflect.Ptr && valueField.Elem().Kind() == reflect.Struct)) &&
			typeField.Tag.Get(tagName) != "-" {
			var err error
			structResult, err = ValidateStruct(valueField.Interface())
			if err != nil {
				err = prependPathToErrors(err, typeField.Name)
				errs = append(errs, err)
			}
		}
		resultField, err2 := typeCheck(valueField, typeField, val, nil)
		if err2 != nil {

			// Replace structure name with JSON name if there is a tag on the variable
			jsonTag := toJSONName(typeField.Tag.Get("json"))
			if jsonTag != "" {
				switch jsonError := err2.(type) {
				case Error:
					jsonError.Name = jsonTag
					err2 = jsonError
				case Errors:
					for i2, err3 := range jsonError {
						switch customErr := err3.(type) {
						case Error:
							customErr.Name = jsonTag
							jsonError[i2] = customErr
						}
					}

					err2 = jsonError
				}
			}

			errs = append(errs, err2)
		}
		result = result && resultField && structResult
	}
	if len(errs) > 0 {
		err = errs
	}
	return result, err
}

// ValidateStructAsync performs async validation of the struct and returns results through the channels
func ValidateStructAsync(s interface{}) (<-chan bool, <-chan error) {
	res := make(chan bool)
	errors := make(chan error)

	go func() {
		defer close(res)
		defer close(errors)

		isValid, isFailed := ValidateStruct(s)

		res <- isValid
		errors <- isFailed
	}()

	return res, errors
}

// ValidateMapAsync performs async validation of the map and returns results through the channels
func ValidateMapAsync(s map[string]interface{}, m map[string]interface{}) (<-chan bool, <-chan error) {
	res := make(chan bool)
	errors := make(chan error)

	go func() {
		defer close(res)
		defer close(errors)

		isValid, isFailed := ValidateMap(s, m)

		res <- isValid
		errors <- isFailed
	}()

	return res, errors
}

// parseTagIntoMap parses a struct tag `valid:required~Some error message,length(2|3)` into map[string]string{"required": "Some error message", "length(2|3)": ""}
func parseTagIntoMap(tag string) tagOptionsMap {
	optionsMap := make(tagOptionsMap)
	options := strings.Split(tag, ",")

	for i, option := range options {
		option = strings.TrimSpace(option)

		validationOptions := strings.Split(option, "~")
		if !isValidTag(validationOptions[0]) {
			continue
		}
		if len(validationOptions) == 2 {
			optionsMap[validationOptions[0]] = tagOption{validationOptions[0], validationOptions[1], i}
		} else {
			optionsMap[validationOptions[0]] = tagOption{validationOptions[0], "", i}
		}
	}
	return optionsMap
}

func isValidTag(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		switch {
		case strings.ContainsRune("\\'\"!#$%&()*+-./:<=>?@[]^_{|}~ ", c):
			// Backslash and quote chars are reserved, but
			// otherwise any punctuation chars are allowed
			// in a tag name.
		default:
			if !unicode.IsLetter(c) && !unicode.IsDigit(c) {
				return false
			}
		}
	}
	return true
}

// IsSSN will validate the given string as a U.S. Social Security Number
func IsSSN(str string) bool {
	if str == "" || len(str) != 11 {
		return false
	}
	return rxSSN.MatchString(str)
}

// IsSemver checks if string is valid semantic version
func IsSemver(str string) bool {
	return rxSemver.MatchString(str)
}

// IsType checks if interface is of some type
func IsType(v interface{}, params ...string) bool {
	if len(params) == 1 {
		typ := params[0]
		return strings.Replace(reflect.TypeOf(v).String(), " ", "", -1) == strings.Replace(typ, " ", "", -1)
	}
	return false
}

// IsTime checks if string is valid according to given format
func IsTime(str string, format string) bool {
	_, err := time.Parse(format, str)
	return err == nil
}

// IsUnixTime checks if string is valid unix timestamp value
func IsUnixTime(str string) bool {
	if _, err := strconv.Atoi(str); err == nil {
		return true
	}
	return false
}

// IsRFC3339 checks if string is valid timestamp value according to RFC3339
func IsRFC3339(str string) bool {
	return IsTime(str, time.RFC3339)
}

// IsRFC3339WithoutZone checks if string is valid timestamp value according to RFC3339 which excludes the timezone.
func IsRFC3339WithoutZone(str string) bool {
	return IsTime(str, rfc3339WithoutZone)
}

// IsISO4217 checks if string is valid ISO currency code
func IsISO4217(str string) bool {
	for _, currency := range ISO4217List {
		if str == currency {
			return true
		}
	}

	return false
}

// ByteLength checks string's length
func ByteLength(str string, params ...string) bool {
	if len(params) == 2 {
		min, _ := ToInt(params[0])
		max, _ := ToInt(params[1])
		return len(str) >= int(min) && len(str) <= int(max)
	}

	return false
}

// RuneLength checks string's length
// Alias for StringLength
func RuneLength(str string, params ...string) bool {
	return StringLength(str, params...)
}

// IsRsaPub checks whether string is valid RSA key
// Alias for IsRsaPublicKey
func IsRsaPub(str string, params ...string) bool {
	if len(params) == 1 {
		len, _ := ToInt(params[0])
		return IsRsaPublicKey(str, int(len))
	}

	return false
}

// StringMatches checks if a string matches a given pattern.
func StringMatches(s string, params ...string) bool {
	if len(params) == 1 {
		pattern := params[0]
		return Matches(s, pattern)
	}
	return false
}

// StringLength checks string's length (including multi byte strings)
func StringLength(str string, params ...string) bool {

	if len(params) == 2 {
		strLength := utf8.RuneCountInString(str)
		min, _ := ToInt(params[0])
		max, _ := ToInt(params[1])
		return strLength >= int(min) && strLength <= int(max)
	}

	return false
}

// MinStringLength checks string's minimum length (including multi byte strings)
func MinStringLength(str string, params ...string) bool {

	if len(params) == 1 {
		strLength := utf8.RuneCountInString(str)
		min, _ := ToInt(params[0])
		return strLength >= int(min)
	}

	return false
}

// MaxStringLength checks string's maximum length (including multi byte strings)
func MaxStringLength(str string, params ...string) bool {

	if len(params) == 1 {
		strLength := utf8.RuneCountInString(str)
		max, _ := ToInt(params[0])
		return strLength <= int(max)
	}

	return false
}

// Range checks string's length
func Range(str string, params ...string) bool {
	if len(params) == 2 {
		value, _ := ToFloat(str)
		min, _ := ToFloat(params[0])
		max, _ := ToFloat(params[1])
		return InRange(value, min, max)
	}

	return false
}

// IsInRaw checks if string is in list of allowed values
func IsInRaw(str string, params ...string) bool {
	if len(params) == 1 {
		rawParams := params[0]

		parsedParams := strings.Split(rawParams, "|")

		return IsIn(str, parsedParams...)
	}

	return false
}

// IsIn checks if string str is a member of the set of strings params
func IsIn(str string, params ...string) bool {
	for _, param := range params {
		if str == param {
			return true
		}
	}

	return false
}

func checkRequired(v reflect.Value, t reflect.StructField, options tagOptionsMap) (bool, error) {
	if nilPtrAllowedByRequired {
		k := v.Kind()
		if (k == reflect.Ptr || k == reflect.Interface) && v.IsNil() {
			return true, nil
		}
	}

	if requiredOption, isRequired := options["required"]; isRequired {
		if len(requiredOption.customErrorMessage) > 0 {
			return false, Error{t.Name, fmt.Errorf(requiredOption.customErrorMessage), true, "required", []string{}}
		}
		return false, Error{t.Name, fmt.Errorf("non zero value required"), false, "required", []string{}}
	} else if _, isOptional := options["optional"]; fieldsRequiredByDefault && !isOptional {
		return false, Error{t.Name, fmt.Errorf("Missing required field"), false, "required", []string{}}
	}
	// not required and empty is valid
	return true, nil
}

func typeCheck(v reflect.Value, t reflect.StructField, o reflect.Value, options tagOptionsMap) (isValid bool, resultErr error) {
	if !v.IsValid() {
		return false, nil
	}

	tag := t.Tag.Get(tagName)

	// checks if the field should be ignored
	switch tag {
	case "":
		if v.Kind() != reflect.Slice && v.Kind() != reflect.Map {
			if !fieldsRequiredByDefault {
				return true, nil
			}
			return false, Error{t.Name, fmt.Errorf("All fields are required to at least have one validation defined"), false, "required", []string{}}
		}
	case "-":
		return true, nil
	}

	isRootType := false
	if options == nil {
		isRootType = true
		options = parseTagIntoMap(tag)
	}

	if isEmptyValue(v) {
		// an empty value is not validated, checks only required
		isValid, resultErr = checkRequired(v, t, options)
		for key := range options {
			delete(options, key)
		}
		return isValid, resultErr
	}

	var customTypeErrors Errors
	optionsOrder := options.orderedKeys()
	for _, validatorName := range optionsOrder {
		validatorStruct := options[validatorName]
		if validatefunc, ok := CustomTypeTagMap.Get(validatorName); ok {
			delete(options, validatorName)

			if result := validatefunc(v.Interface(), o.Interface()); !result {
				if len(validatorStruct.customErrorMessage) > 0 {
					customTypeErrors = append(customTypeErrors, Error{Name: t.Name, Err: TruncatingErrorf(validatorStruct.customErrorMessage, fmt.Sprint(v), validatorName), CustomErrorMessageExists: true, Validator: stripParams(validatorName)})
					continue
				}
				customTypeErrors = append(customTypeErrors, Error{Name: t.Name, Err: fmt.Errorf("%s does not validate as %s", fmt.Sprint(v), validatorName), CustomErrorMessageExists: false, Validator: stripParams(validatorName)})
			}
		}
	}

	if len(customTypeErrors.Errors()) > 0 {
		return false, customTypeErrors
	}

	if isRootType {
		// Ensure that we've checked the value by all specified validators before report that the value is valid
		defer func() {
			delete(options, "optional")
			delete(options, "required")

			if isValid && resultErr == nil && len(options) != 0 {
				optionsOrder := options.orderedKeys()
				for _, validator := range optionsOrder {
					isValid = false
					resultErr = Error{t.Name, fmt.Errorf(
						"The following validator is invalid or can't be applied to the field: %q", validator), false, stripParams(validator), []string{}}
					return
				}
			}
		}()
	}

	for _, validatorSpec := range optionsOrder {
		validatorStruct := options[validatorSpec]
		var negate bool
		validator := validatorSpec
		customMsgExists := len(validatorStruct.customErrorMessage) > 0

		// checks whether the tag looks like '!something' or 'something'
		if validator[0] == '!' {
			validator = validator[1:]
			negate = true
		}

		// checks for interface param validators
		for key, value := range InterfaceParamTagRegexMap {
			ps := value.FindStringSubmatch(validator)
			if len(ps) == 0 {
				continue
			}

			validatefunc, ok := InterfaceParamTagMap[key]
			if !ok {
				continue
			}

			delete(options, validatorSpec)

			field := fmt.Sprint(v)
			if result := validatefunc(v.Interface(), ps[1:]...); (!result && !negate) || (result && negate) {
				if customMsgExists {
					return false, Error{t.Name, TruncatingErrorf(validatorStruct.customErrorMessage, field, validator), customMsgExists, stripParams(validatorSpec), []string{}}
				}
				if negate {
					return false, Error{t.Name, fmt.Errorf("%s does validate as %s", field, validator), customMsgExists, stripParams(validatorSpec), []string{}}
				}
				return false, Error{t.Name, fmt.Errorf("%s does not validate as %s", field, validator), customMsgExists, stripParams(validatorSpec), []string{}}
			}
		}
	}

	switch v.Kind() {
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64,
		reflect.String:
		// for each tag option checks the map of validator functions
		for _, validatorSpec := range optionsOrder {
			validatorStruct := options[validatorSpec]
			var negate bool
			validator := validatorSpec
			customMsgExists := len(validatorStruct.customErrorMessage) > 0

			// checks whether the tag looks like '!something' or 'something'
			if validator[0] == '!' {
				validator = validator[1:]
				negate = true
			}

			// checks for param validators
			for key, value := range ParamTagRegexMap {
				ps := value.FindStringSubmatch(validator)
				if len(ps) == 0 {
					continue
				}

				validatefunc, ok := ParamTagMap[key]
				if !ok {
					continue
				}

				delete(options, validatorSpec)

				switch v.Kind() {
				case reflect.String,
					reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
					reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
					reflect.Float32, reflect.Float64:

					field := fmt.Sprint(v) // make value into string, then validate with regex
					if result := validatefunc(field, ps[1:]...); (!result && !negate) || (result && negate) {
						if customMsgExists {
							return false, Error{t.Name, TruncatingErrorf(validatorStruct.customErrorMessage, field, validator), customMsgExists, stripParams(validatorSpec), []string{}}
						}
						if negate {
							return false, Error{t.Name, fmt.Errorf("%s does validate as %s", field, validator), customMsgExists, stripParams(validatorSpec), []string{}}
						}
						return false, Error{t.Name, fmt.Errorf("%s does not validate as %s", field, validator), customMsgExists, stripParams(validatorSpec), []string{}}
					}
				default:
					// type not yet supported, fail
					return false, Error{t.Name, fmt.Errorf("Validator %s doesn't support kind %s", validator, v.Kind()), false, stripParams(validatorSpec), []string{}}
				}
			}

			if validatefunc, ok := TagMap[validator]; ok {
				delete(options, validatorSpec)

				switch v.Kind() {
				case reflect.String,
					reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
					reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
					reflect.Float32, reflect.Float64:
					field := fmt.Sprint(v) // make value into string, then validate with regex
					if result := validatefunc(field); !result && !negate || result && negate {
						if customMsgExists {
							return false, Error{t.Name, TruncatingErrorf(validatorStruct.customErrorMessage, field, validator), customMsgExists, stripParams(validatorSpec), []string{}}
						}
						if negate {
							return false, Error{t.Name, fmt.Errorf("%s does validate as %s", field, validator), customMsgExists, stripParams(validatorSpec), []string{}}
						}
						return false, Error{t.Name, fmt.Errorf("%s does not validate as %s", field, validator), customMsgExists, stripParams(validatorSpec), []string{}}
					}
				default:
					//Not Yet Supported Types (Fail here!)
					err := fmt.Errorf("Validator %s doesn't support kind %s for value %v", validator, v.Kind(), v)
					return false, Error{t.Name, err, false, stripParams(validatorSpec), []string{}}
				}
			}
		}
		return true, nil
	case reflect.Map:
		if v.Type().Key().Kind() != reflect.String {
			return false, &UnsupportedTypeError{v.Type()}
		}
		var sv stringValues
		sv = v.MapKeys()
		sort.Sort(sv)
		result := true
		for i, k := range sv {
			var resultItem bool
			var err error
			if v.MapIndex(k).Kind() != reflect.Struct {
				resultItem, err = typeCheck(v.MapIndex(k), t, o, options)
				if err != nil {
					return false, err
				}
			} else {
				resultItem, err = ValidateStruct(v.MapIndex(k).Interface())
				if err != nil {
					err = prependPathToErrors(err, t.Name+"."+sv[i].Interface().(string))
					return false, err
				}
			}
			result = result && resultItem
		}
		return result, nil
	case reflect.Slice, reflect.Array:
		result := true
		for i := 0; i < v.Len(); i++ {
			var resultItem bool
			var err error
			if v.Index(i).Kind() != reflect.Struct {
				resultItem, err = typeCheck(v.Index(i), t, o, options)
				if err != nil {
					return false, err
				}
			} else {
				resultItem, err = ValidateStruct(v.Index(i).Interface())
				if err != nil {
					err = prependPathToErrors(err, t.Name+"."+strconv.Itoa(i))
					return false, err
				}
			}
			result = result && resultItem
		}
		return result, nil
	case reflect.Interface:
		// If the value is an interface then encode its element
		if v.IsNil() {
			return true, nil
		}
		return ValidateStruct(v.Interface())
	case reflect.Ptr:
		// If the value is a pointer then checks its element
		if v.IsNil() {
			return true, nil
		}
		return typeCheck(v.Elem(), t, o, options)
	case reflect.Struct:
		return true, nil
	default:
		return false, &UnsupportedTypeError{v.Type()}
	}
}

func stripParams(validatorString string) string {
	return paramsRegexp.ReplaceAllString(validatorString, "")
}

// isEmptyValue checks whether value empty or not
func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.String, reflect.Array:
		return v.Len() == 0
	case reflect.Map, reflect.Slice:
		return v.Len() == 0 || v.IsNil()
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}

	return reflect.DeepEqual(v.Interface(), reflect.Zero(v.Type()).Interface())
}

// ErrorByField returns error for specified field of the struct
// validated by ValidateStruct or empty string if there are no errors
// or this field doesn't exists or doesn't have any errors.
func ErrorByField(e error, field string) string {
	if e == nil {
		return ""
	}
	return ErrorsByField(e)[field]
}

// ErrorsByField returns map of errors of the struct validated
// by ValidateStruct or empty map if there are no errors.
func ErrorsByField(e error) map[string]string {
	m := make(map[string]string)
	if e == nil {
		return m
	}
	// prototype for ValidateStruct

	switch e := e.(type) {
	case Error:
		m[e.Name] = e.Err.Error()
	case Errors:
		for _, item := range e.Errors() {
			n := ErrorsByField(item)
			for k, v := range n {
				m[k] = v
			}
		}
	}

	return m
}

// Error returns string equivalent for reflect.Type
func (e *UnsupportedTypeError) Error() string {
	return "validator: unsupported type: " + e.Type.String()
}

func (sv stringValues) Len() int           { return len(sv) }
func (sv stringValues) Swap(i, j int)      { sv[i], sv[j] = sv[j], sv[i] }
func (sv stringValues) Less(i, j int) bool { return sv.get(i) < sv.get(j) }
func (sv stringValues) get(i int) string   { return sv[i].String() }

func IsE164(str string) bool {
	return rxE164.MatchString(str)
}
