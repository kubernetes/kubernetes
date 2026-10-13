// +build windows

package winterm

// AddInRange increments a value by the passed quantity while ensuring the values
// always remain within the supplied min / max range.
func addInRange(n int16, increment int16, min int16, max int16) int16 {
	return ensureInRange(n+increment, min, max)
}
