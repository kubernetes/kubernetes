package types

import (
	"encoding/json"
	"fmt"
	"time"
)

//ReportEntryValue wraps a report entry's value ensuring it can be encoded and decoded safely into reports
//and across the network connection when running in parallel
type ReportEntryValue struct {
	raw            interface{} //unexported to prevent gob from freaking out about unregistered structs
	AsJSON         string
	Representation string
}

func WrapEntryValue(value interface{}) ReportEntryValue {
	return ReportEntryValue{
		raw: value,
	}
}

func (rev ReportEntryValue) GetRawValue() interface{} {
	return rev.raw
}

func (rev ReportEntryValue) String() string {
	if rev.raw == nil {
		return ""
	}
	if colorableStringer, ok := rev.raw.(ColorableStringer); ok {
		return colorableStringer.ColorableString()
	}

	if stringer, ok := rev.raw.(fmt.Stringer); ok {
		return stringer.String()
	}
	if rev.Representation != "" {
		return rev.Representation
	}
	return fmt.Sprintf("%+v", rev.raw)
}

func (rev ReportEntryValue) MarshalJSON() ([]byte, error) {
	//All this to capture the representaiton at encoding-time, not creating time
	//This way users can Report on pointers and get their final values at reporting-time
	out := struct {
		AsJSON         string
		Representation string
	}{
		Representation: rev.String(),
	}

	asJSON, err := json.Marshal(rev.raw)
	if err != nil {
		return nil, err
	}
	out.AsJSON = string(asJSON)

	return json.Marshal(out)
}

func (rev *ReportEntryValue) UnmarshalJSON(data []byte) error {
	in := struct {
		AsJSON         string
		Representation string
	}{}
	err := json.Unmarshal(data, &in)
	if err != nil {
		return err
	}
	rev.AsJSON = in.AsJSON
	rev.Representation = in.Representation
	return json.Unmarshal([]byte(in.AsJSON), &(rev.raw))
}

func (rev ReportEntryValue) GobEncode() ([]byte, error) {
	return rev.MarshalJSON()
}

func (rev *ReportEntryValue) GobDecode(data []byte) error {
	return rev.UnmarshalJSON(data)
}

// ReportEntry captures information attached to `SpecReport` via `AddReportEntry`
type ReportEntry struct {
	// Visibility captures the visibility policy for this ReportEntry
	Visibility ReportEntryVisibility
	// Time captures the time the AddReportEntry was called
	Time time.Time
	// Location captures the location of the AddReportEntry call
	Location CodeLocation
	// Name captures the name of this report
	Name string
	// Value captures the (optional) object passed into AddReportEntry - this can be
	// anything the user wants.  The value passed to AddReportEntry is wrapped in a ReportEntryValue to make
	// encoding/decoding the value easier.  To access the raw value call entry.GetRawValue()
	Value ReportEntryValue
}

// ColorableStringer is an interface that ReportEntry values can satisfy.  If they do then ColorableStirng() is used to generate their representation.
type ColorableStringer interface {
	ColorableString() string
}

// StringRepresentation() returns the string representation of the value associated with the ReportEntry --
// if value is nil, empty string is returned
// if value is a `ColorableStringer` then `Value.ColorableString()` is returned
// if value is a `fmt.Stringer` then `Value.String()` is returned
// otherwise the value is formatted with "%+v"
func (entry ReportEntry) StringRepresentation() string {
	return entry.Value.String()
}

// GetRawValue returns the Value object that was passed to AddReportEntry
// If called in-process this will be the same object that was passed into AddReportEntry.
// If used from a rehydrated JSON file _or_ in a ReportAfterSuite when running in parallel this will be
// a JSON-decoded {}interface.  If you want to reconstitute your original object you can decode the entry.Value.AsJSON
// field yourself.
func (entry ReportEntry) GetRawValue() interface{} {
	return entry.Value.GetRawValue()
}

type ReportEntries []ReportEntry

func (re ReportEntries) HasVisibility(visibilities ...ReportEntryVisibility) bool {
	for _, entry := range re {
		if entry.Visibility.Is(visibilities...) {
			return true
		}
	}
	return false
}

func (re ReportEntries) WithVisibility(visibilities ...ReportEntryVisibility) ReportEntries {
	out := ReportEntries{}

	for _, entry := range re {
		if entry.Visibility.Is(visibilities...) {
			out = append(out, entry)
		}
	}

	return out
}

// ReportEntryVisibility governs the visibility of ReportEntries in Ginkgo's console reporter
type ReportEntryVisibility uint

const (
	// Always print out this ReportEntry
	ReportEntryVisibilityAlways ReportEntryVisibility = iota
	// Only print out this ReportEntry if the spec fails or if the test is run with -v
	ReportEntryVisibilityFailureOrVerbose
	// Never print out this ReportEntry (note that ReportEntrys are always encoded in machine readable reports (e.g. JSON, JUnit, etc.))
	ReportEntryVisibilityNever
)

var revEnumSupport = NewEnumSupport(map[uint]string{
	uint(ReportEntryVisibilityAlways):           "always",
	uint(ReportEntryVisibilityFailureOrVerbose): "failure-or-verbose",
	uint(ReportEntryVisibilityNever):            "never",
})

func (rev ReportEntryVisibility) String() string {
	return revEnumSupport.String(uint(rev))
}
func (rev *ReportEntryVisibility) UnmarshalJSON(b []byte) error {
	out, err := revEnumSupport.UnmarshJSON(b)
	*rev = ReportEntryVisibility(out)
	return err
}
func (rev ReportEntryVisibility) MarshalJSON() ([]byte, error) {
	return revEnumSupport.MarshJSON(uint(rev))
}

func (v ReportEntryVisibility) Is(visibilities ...ReportEntryVisibility) bool {
	for _, visibility := range visibilities {
		if v == visibility {
			return true
		}
	}

	return false
}
