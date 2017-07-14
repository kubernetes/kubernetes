// +build windows

package winterm

// AddInRange increments a value by the passed quantity while ensuring the values
// always remain within the supplied min / max range.
func AddInRange(n SHORT, increment SHORT, min SHORT, max SHORT) SHORT {
	return ensureInRange(n+increment, min, max)
}
