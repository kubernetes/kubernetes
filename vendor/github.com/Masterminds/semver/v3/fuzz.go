// +build gofuzz

package semver

func Fuzz(data []byte) int {
	d := string(data)

	// Test NewVersion
	_, _ = NewVersion(d)

	// Test StrictNewVersion
	_, _ = StrictNewVersion(d)

	// Test NewConstraint
	_, _ = NewConstraint(d)

	// The return value should be 0 normally, 1 if the priority in future tests
	// should be increased, and -1 if future tests should skip passing in that
	// data. We do not have a reason to change priority so 0 is always returned.
	// There are example tests that do this.
	return 0
}
