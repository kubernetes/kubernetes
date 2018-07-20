package progress

import (
	"fmt"
)

// Progress represents the progress of a transfer.
type Progress struct {
	ID string

	// Progress contains a Message or...
	Message string

	// ...progress of an action
	Action  string
	Current int64
	Total   int64

	// If true, don't show xB/yB
	HideCounts bool
	// If not empty, use units instead of bytes for counts
	Units string

	// Aux contains extra information not presented to the user, such as
	// digests for push signing.
	Aux interface{}

	LastUpdate bool
}

// Output is an interface for writing progress information. It's
// like a writer for progress, but we don't call it Writer because
// that would be confusing next to ProgressReader (also, because it
// doesn't implement the io.Writer interface).
type Output interface {
	WriteProgress(Progress) error
}

type chanOutput chan<- Progress

func (out chanOutput) WriteProgress(p Progress) error {
	out <- p
	return nil
}

// ChanOutput returns an Output that writes progress updates to the
// supplied channel.
func ChanOutput(progressChan chan<- Progress) Output {
	return chanOutput(progressChan)
}

type discardOutput struct{}

func (discardOutput) WriteProgress(Progress) error {
	return nil
}

// DiscardOutput returns an Output that discards progress
func DiscardOutput() Output {
	return discardOutput{}
}

// Update is a convenience function to write a progress update to the channel.
func Update(out Output, id, action string) {
	out.WriteProgress(Progress{ID: id, Action: action})
}

// Updatef is a convenience function to write a printf-formatted progress update
// to the channel.
func Updatef(out Output, id, format string, a ...interface{}) {
	Update(out, id, fmt.Sprintf(format, a...))
}

// Message is a convenience function to write a progress message to the channel.
func Message(out Output, id, message string) {
	out.WriteProgress(Progress{ID: id, Message: message})
}

// Messagef is a convenience function to write a printf-formatted progress
// message to the channel.
func Messagef(out Output, id, format string, a ...interface{}) {
	Message(out, id, fmt.Sprintf(format, a...))
}

// Aux sends auxiliary information over a progress interface, which will not be
// formatted for the UI. This is used for things such as push signing.
func Aux(out Output, a interface{}) {
	out.WriteProgress(Progress{Aux: a})
}
