package x509

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
)

// Error implements the error interface and describes a single error in an X.509 certificate or CRL.
type Error struct {
	ID       ErrorID
	Category ErrCategory
	Summary  string
	Field    string
	SpecRef  string
	SpecText string
	// Fatal indicates that parsing has been aborted.
	Fatal bool
}

func (err Error) Error() string {
	var msg bytes.Buffer
	if err.ID != ErrInvalidID {
		if err.Fatal {
			msg.WriteRune('E')
		} else {
			msg.WriteRune('W')
		}
		msg.WriteString(fmt.Sprintf("%03d: ", err.ID))
	}
	msg.WriteString(err.Summary)
	return msg.String()
}

// VerboseError creates a more verbose error string, including spec details.
func (err Error) VerboseError() string {
	var msg bytes.Buffer
	msg.WriteString(err.Error())
	if len(err.Field) > 0 || err.Category != UnknownCategory || len(err.SpecRef) > 0 || len(err.SpecText) > 0 {
		msg.WriteString(" (")
		needSep := false
		if len(err.Field) > 0 {
			msg.WriteString(err.Field)
			needSep = true
		}
		if err.Category != UnknownCategory {
			if needSep {
				msg.WriteString(": ")
			}
			msg.WriteString(err.Category.String())
			needSep = true
		}
		if len(err.SpecRef) > 0 {
			if needSep {
				msg.WriteString(": ")
			}
			msg.WriteString(err.SpecRef)
			needSep = true
		}
		if len(err.SpecText) > 0 {
			if needSep {
				if len(err.SpecRef) > 0 {
					msg.WriteString(", ")
				} else {
					msg.WriteString(": ")
				}
			}
			msg.WriteString("'")
			msg.WriteString(err.SpecText)
			msg.WriteString("'")
		}
		msg.WriteString(")")
	}

	return msg.String()
}

// ErrCategory indicates the category of an x509.Error.
type ErrCategory int

// ErrCategory values.
const (
	UnknownCategory ErrCategory = iota
	// Errors in ASN.1 encoding
	InvalidASN1Encoding
	InvalidASN1Content
	InvalidASN1DER
	// Errors in ASN.1 relative to schema
	InvalidValueRange
	InvalidASN1Type
	UnexpectedAdditionalData
	// Errors in X.509
	PoorlyFormedCertificate // Fails a SHOULD clause
	MalformedCertificate    // Fails a MUST clause
	PoorlyFormedCRL         // Fails a SHOULD clause
	MalformedCRL            // Fails a MUST clause
	// Errors relative to CA/Browser Forum guidelines
	BaselineRequirementsFailure
	EVRequirementsFailure
	// Other errors
	InsecureAlgorithm
	UnrecognizedValue
)

func (category ErrCategory) String() string {
	switch category {
	case InvalidASN1Encoding:
		return "Invalid ASN.1 encoding"
	case InvalidASN1Content:
		return "Invalid ASN.1 content"
	case InvalidASN1DER:
		return "Invalid ASN.1 distinguished encoding"
	case InvalidValueRange:
		return "Invalid value for range given in schema"
	case InvalidASN1Type:
		return "Invalid ASN.1 type for schema"
	case UnexpectedAdditionalData:
		return "Unexpected additional data present"
	case PoorlyFormedCertificate:
		return "Certificate does not comply with SHOULD clause in spec"
	case MalformedCertificate:
		return "Certificate does not comply with MUST clause in spec"
	case PoorlyFormedCRL:
		return "Certificate Revocation List does not comply with SHOULD clause in spec"
	case MalformedCRL:
		return "Certificate Revocation List does not comply with MUST clause in spec"
	case BaselineRequirementsFailure:
		return "Certificate does not comply with CA/BF baseline requirements"
	case EVRequirementsFailure:
		return "Certificate does not comply with CA/BF EV requirements"
	case InsecureAlgorithm:
		return "Certificate uses an insecure algorithm"
	case UnrecognizedValue:
		return "Certificate uses an unrecognized value"
	default:
		return fmt.Sprintf("Unknown (%d)", category)
	}
}

// ErrorID is an identifier for an x509.Error, to allow filtering.
type ErrorID int

// Errors implements the error interface and holds a collection of errors found in a certificate or CRL.
type Errors struct {
	Errs []Error
}

// Error converts to a string.
func (e *Errors) Error() string {
	return e.combineErrors(Error.Error)
}

// VerboseError creates a more verbose error string, including spec details.
func (e *Errors) VerboseError() string {
	return e.combineErrors(Error.VerboseError)
}

// Fatal indicates whether e includes a fatal error
func (e *Errors) Fatal() bool {
	return (e.FirstFatal() != nil)
}

// Empty indicates whether e has no errors.
func (e *Errors) Empty() bool {
	return len(e.Errs) == 0
}

// FirstFatal returns the first fatal error in e, or nil
// if there is no fatal error.
func (e *Errors) FirstFatal() error {
	for _, err := range e.Errs {
		if err.Fatal {
			return err
		}
	}
	return nil

}

// AddID adds the Error identified by the given id to an x509.Errors.
func (e *Errors) AddID(id ErrorID, args ...interface{}) {
	e.Errs = append(e.Errs, NewError(id, args...))
}

func (e Errors) combineErrors(errfn func(Error) string) string {
	if len(e.Errs) == 0 {
		return ""
	}
	if len(e.Errs) == 1 {
		return errfn((e.Errs)[0])
	}
	var msg bytes.Buffer
	msg.WriteString("Errors:")
	for _, err := range e.Errs {
		msg.WriteString("\n  ")
		msg.WriteString(errfn(err))
	}
	return msg.String()
}

// Filter creates a new Errors object with any entries from the filtered
// list of IDs removed.
func (e Errors) Filter(filtered []ErrorID) Errors {
	var results Errors
eloop:
	for _, v := range e.Errs {
		for _, f := range filtered {
			if v.ID == f {
				break eloop
			}
		}
		results.Errs = append(results.Errs, v)
	}
	return results
}

// ErrorFilter builds a list of error IDs (suitable for use with Errors.Filter) from a comma-separated string.
func ErrorFilter(ignore string) []ErrorID {
	var ids []ErrorID
	filters := strings.Split(ignore, ",")
	for _, f := range filters {
		v, err := strconv.Atoi(f)
		if err != nil {
			continue
		}
		ids = append(ids, ErrorID(v))
	}
	return ids
}
