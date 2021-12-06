//go:build !linux
// +build !linux

package label

// InitLabels returns the process label and file labels to be used within
// the container.  A list of options can be passed into this function to alter
// the labels.
func InitLabels(options []string) (string, string, error) {
	return "", "", nil
}

// Deprecated: The GenLabels function is only to be used during the transition
// to the official API. Use InitLabels(strings.Fields(options)) instead.
func GenLabels(options string) (string, string, error) {
	return "", "", nil
}

func SetFileLabel(path string, fileLabel string) error {
	return nil
}

func SetFileCreateLabel(fileLabel string) error {
	return nil
}

func Relabel(path string, fileLabel string, shared bool) error {
	return nil
}

// DisableSecOpt returns a security opt that can disable labeling
// support for future container processes
func DisableSecOpt() []string {
	return nil
}

// Validate checks that the label does not include unexpected options
func Validate(label string) error {
	return nil
}

// RelabelNeeded checks whether the user requested a relabel
func RelabelNeeded(label string) bool {
	return false
}

// IsShared checks that the label includes a "shared" mark
func IsShared(label string) bool {
	return false
}
