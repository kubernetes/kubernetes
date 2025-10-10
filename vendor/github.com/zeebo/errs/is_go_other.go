//go:build !go1.20
// +build !go1.20

package errs

// Is checks if any of the underlying errors matches target
func Is(err, target error) bool {
	return IsFunc(err, func(err error) bool {
		if err == target {
			return true
		}
		if x, ok := err.(interface{ Is(error) bool }); ok && x.Is(target) {
			return true
		}
		return false
	})
}
