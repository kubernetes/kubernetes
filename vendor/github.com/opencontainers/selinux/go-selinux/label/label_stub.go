//go:build !linux
// +build !linux

package label

// InitLabels returns the process label and file labels to be used within
// the container.  A list of options can be passed into this function to alter
// the labels.
func InitLabels([]string) (string, string, error) {
	return "", "", nil
}

func SetFileLabel(string, string) error {
	return nil
}

func SetFileCreateLabel(string) error {
	return nil
}

func Relabel(string, string, bool) error {
	return nil
}

// DisableSecOpt returns a security opt that can disable labeling
// support for future container processes
func DisableSecOpt() []string {
	return nil
}

// Validate checks that the label does not include unexpected options
func Validate(string) error {
	return nil
}

// RelabelNeeded checks whether the user requested a relabel
func RelabelNeeded(string) bool {
	return false
}

// IsShared checks that the label includes a "shared" mark
func IsShared(string) bool {
	return false
}
