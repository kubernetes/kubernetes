package gojsonschema

import (
	"net"
	"net/mail"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"
)

type (
	// FormatChecker is the interface all formatters added to FormatCheckerChain must implement
	FormatChecker interface {
		// IsFormat checks if input has the correct format and type
		IsFormat(input interface{}) bool
	}

	// FormatCheckerChain holds the formatters
	FormatCheckerChain struct {
		formatters map[string]FormatChecker
	}

	// EmailFormatChecker verifies email address formats
	EmailFormatChecker struct{}

	// IPV4FormatChecker verifies IP addresses in the IPv4 format
	IPV4FormatChecker struct{}

	// IPV6FormatChecker verifies IP addresses in the IPv6 format
	IPV6FormatChecker struct{}

	// DateTimeFormatChecker verifies date/time formats per RFC3339 5.6
	//
	// Valid formats:
	// 		Partial Time: HH:MM:SS
	//		Full Date: YYYY-MM-DD
	// 		Full Time: HH:MM:SSZ-07:00
	//		Date Time: YYYY-MM-DDTHH:MM:SSZ-0700
	//
	// 	Where
	//		YYYY = 4DIGIT year
	//		MM = 2DIGIT month ; 01-12
	//		DD = 2DIGIT day-month ; 01-28, 01-29, 01-30, 01-31 based on month/year
	//		HH = 2DIGIT hour ; 00-23
	//		MM = 2DIGIT ; 00-59
	//		SS = 2DIGIT ; 00-58, 00-60 based on leap second rules
	//		T = Literal
	//		Z = Literal
	//
	//	Note: Nanoseconds are also suported in all formats
	//
	// http://tools.ietf.org/html/rfc3339#section-5.6
	DateTimeFormatChecker struct{}

	// DateFormatChecker verifies date formats
	//
	// Valid format:
	//		Full Date: YYYY-MM-DD
	//
	// 	Where
	//		YYYY = 4DIGIT year
	//		MM = 2DIGIT month ; 01-12
	//		DD = 2DIGIT day-month ; 01-28, 01-29, 01-30, 01-31 based on month/year
	DateFormatChecker struct{}

	// TimeFormatChecker verifies time formats
	//
	// Valid formats:
	// 		Partial Time: HH:MM:SS
	// 		Full Time: HH:MM:SSZ-07:00
	//
	// 	Where
	//		HH = 2DIGIT hour ; 00-23
	//		MM = 2DIGIT ; 00-59
	//		SS = 2DIGIT ; 00-58, 00-60 based on leap second rules
	//		T = Literal
	//		Z = Literal
	TimeFormatChecker struct{}

	// URIFormatChecker validates a URI with a valid Scheme per RFC3986
	URIFormatChecker struct{}

	// URIReferenceFormatChecker validates a URI or relative-reference per RFC3986
	URIReferenceFormatChecker struct{}

	// URITemplateFormatChecker validates a URI template per RFC6570
	URITemplateFormatChecker struct{}

	// HostnameFormatChecker validates a hostname is in the correct format
	HostnameFormatChecker struct{}

	// UUIDFormatChecker validates a UUID is in the correct format
	UUIDFormatChecker struct{}

	// RegexFormatChecker validates a regex is in the correct format
	RegexFormatChecker struct{}

	// JSONPointerFormatChecker validates a JSON Pointer per RFC6901
	JSONPointerFormatChecker struct{}

	// RelativeJSONPointerFormatChecker validates a relative JSON Pointer is in the correct format
	RelativeJSONPointerFormatChecker struct{}
)

var (
	// FormatCheckers holds the valid formatters, and is a public variable
	// so library users can add custom formatters
	FormatCheckers = FormatCheckerChain{
		formatters: map[string]FormatChecker{
			"date":                  DateFormatChecker{},
			"time":                  TimeFormatChecker{},
			"date-time":             DateTimeFormatChecker{},
			"hostname":              HostnameFormatChecker{},
			"email":                 EmailFormatChecker{},
			"idn-email":             EmailFormatChecker{},
			"ipv4":                  IPV4FormatChecker{},
			"ipv6":                  IPV6FormatChecker{},
			"uri":                   URIFormatChecker{},
			"uri-reference":         URIReferenceFormatChecker{},
			"iri":                   URIFormatChecker{},
			"iri-reference":         URIReferenceFormatChecker{},
			"uri-template":          URITemplateFormatChecker{},
			"uuid":                  UUIDFormatChecker{},
			"regex":                 RegexFormatChecker{},
			"json-pointer":          JSONPointerFormatChecker{},
			"relative-json-pointer": RelativeJSONPointerFormatChecker{},
		},
	}

	// Regex credit: https://www.socketloop.com/tutorials/golang-validate-hostname
	rxHostname = regexp.MustCompile(`^([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])(\.([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]{0,61}[a-zA-Z0-9]))*$`)

	// Use a regex to make sure curly brackets are balanced properly after validating it as a AURI
	rxURITemplate = regexp.MustCompile("^([^{]*({[^}]*})?)*$")

	rxUUID = regexp.MustCompile("^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")

	rxJSONPointer = regexp.MustCompile("^(?:/(?:[^~/]|~0|~1)*)*$")

	rxRelJSONPointer = regexp.MustCompile("^(?:0|[1-9][0-9]*)(?:#|(?:/(?:[^~/]|~0|~1)*)*)$")

	lock = new(sync.RWMutex)
)

// Add adds a FormatChecker to the FormatCheckerChain
// The name used will be the value used for the format key in your json schema
func (c *FormatCheckerChain) Add(name string, f FormatChecker) *FormatCheckerChain {
	lock.Lock()
	c.formatters[name] = f
	lock.Unlock()

	return c
}

// Remove deletes a FormatChecker from the FormatCheckerChain (if it exists)
func (c *FormatCheckerChain) Remove(name string) *FormatCheckerChain {
	lock.Lock()
	delete(c.formatters, name)
	lock.Unlock()

	return c
}

// Has checks to see if the FormatCheckerChain holds a FormatChecker with the given name
func (c *FormatCheckerChain) Has(name string) bool {
	lock.RLock()
	_, ok := c.formatters[name]
	lock.RUnlock()

	return ok
}

// IsFormat will check an input against a FormatChecker with the given name
// to see if it is the correct format
func (c *FormatCheckerChain) IsFormat(name string, input interface{}) bool {
	lock.RLock()
	f, ok := c.formatters[name]
	lock.RUnlock()

	// If a format is unrecognized it should always pass validation
	if !ok {
		return true
	}

	return f.IsFormat(input)
}

// IsFormat checks if input is a correctly formatted e-mail address
func (f EmailFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	_, err := mail.ParseAddress(asString)
	return err == nil
}

// IsFormat checks if input is a correctly formatted IPv4-address
func (f IPV4FormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	// Credit: https://github.com/asaskevich/govalidator
	ip := net.ParseIP(asString)
	return ip != nil && strings.Contains(asString, ".")
}

// IsFormat checks if input is a correctly formatted IPv6=address
func (f IPV6FormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	// Credit: https://github.com/asaskevich/govalidator
	ip := net.ParseIP(asString)
	return ip != nil && strings.Contains(asString, ":")
}

// IsFormat checks if input is a correctly formatted  date/time per RFC3339 5.6
func (f DateTimeFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	formats := []string{
		"15:04:05",
		"15:04:05Z07:00",
		"2006-01-02",
		time.RFC3339,
		time.RFC3339Nano,
	}

	for _, format := range formats {
		if _, err := time.Parse(format, asString); err == nil {
			return true
		}
	}

	return false
}

// IsFormat checks if input is a correctly formatted  date (YYYY-MM-DD)
func (f DateFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}
	_, err := time.Parse("2006-01-02", asString)
	return err == nil
}

// IsFormat checks if input correctly formatted time (HH:MM:SS or HH:MM:SSZ-07:00)
func (f TimeFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	if _, err := time.Parse("15:04:05Z07:00", asString); err == nil {
		return true
	}

	_, err := time.Parse("15:04:05", asString)
	return err == nil
}

// IsFormat checks if input is correctly formatted  URI with a valid Scheme per RFC3986
func (f URIFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	u, err := url.Parse(asString)

	if err != nil || u.Scheme == "" {
		return false
	}

	return !strings.Contains(asString, `\`)
}

// IsFormat checks if input is a correctly formatted URI or relative-reference per RFC3986
func (f URIReferenceFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	_, err := url.Parse(asString)
	return err == nil && !strings.Contains(asString, `\`)
}

// IsFormat checks if input is a correctly formatted URI template per RFC6570
func (f URITemplateFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	u, err := url.Parse(asString)
	if err != nil || strings.Contains(asString, `\`) {
		return false
	}

	return rxURITemplate.MatchString(u.Path)
}

// IsFormat checks if input is a correctly formatted hostname
func (f HostnameFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	return rxHostname.MatchString(asString) && len(asString) < 256
}

// IsFormat checks if input is a correctly formatted UUID
func (f UUIDFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	return rxUUID.MatchString(asString)
}

// IsFormat checks if input is a correctly formatted regular expression
func (f RegexFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	if asString == "" {
		return true
	}
	_, err := regexp.Compile(asString)
	return err == nil
}

// IsFormat checks if input is a correctly formatted JSON Pointer per RFC6901
func (f JSONPointerFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	return rxJSONPointer.MatchString(asString)
}

// IsFormat checks if input is a correctly formatted relative JSON Pointer
func (f RelativeJSONPointerFormatChecker) IsFormat(input interface{}) bool {
	asString, ok := input.(string)
	if !ok {
		return false
	}

	return rxRelJSONPointer.MatchString(asString)
}
