// Package uuid provides simple UUID generation. Only version 4 style UUIDs
// can be generated.
//
// Please see http://tools.ietf.org/html/rfc4122 for details on UUIDs.
package uuid

import (
	"crypto/rand"
	"fmt"
	"io"
	"os"
	"syscall"
	"time"
)

const (
	// Bits is the number of bits in a UUID
	Bits = 128

	// Size is the number of bytes in a UUID
	Size = Bits / 8

	format = "%08x-%04x-%04x-%04x-%012x"
)

var (
	// ErrUUIDInvalid indicates a parsed string is not a valid uuid.
	ErrUUIDInvalid = fmt.Errorf("invalid uuid")

	// Loggerf can be used to override the default logging destination. Such
	// log messages in this library should be logged at warning or higher.
	Loggerf = func(format string, args ...interface{}) {}
)

// UUID represents a UUID value. UUIDs can be compared and set to other values
// and accessed by byte.
type UUID [Size]byte

// Generate creates a new, version 4 uuid.
func Generate() (u UUID) {
	const (
		// ensures we backoff for less than 450ms total. Use the following to
		// select new value, in units of 10ms:
		// 	n*(n+1)/2 = d -> n^2 + n - 2d -> n = (sqrt(8d + 1) - 1)/2
		maxretries = 9
		backoff    = time.Millisecond * 10
	)

	var (
		totalBackoff time.Duration
		count        int
		retries      int
	)

	for {
		// This should never block but the read may fail. Because of this,
		// we just try to read the random number generator until we get
		// something. This is a very rare condition but may happen.
		b := time.Duration(retries) * backoff
		time.Sleep(b)
		totalBackoff += b

		n, err := io.ReadFull(rand.Reader, u[count:])
		if err != nil {
			if retryOnError(err) && retries < maxretries {
				count += n
				retries++
				Loggerf("error generating version 4 uuid, retrying: %v", err)
				continue
			}

			// Any other errors represent a system problem. What did someone
			// do to /dev/urandom?
			panic(fmt.Errorf("error reading random number generator, retried for %v: %v", totalBackoff.String(), err))
		}

		break
	}

	u[6] = (u[6] & 0x0f) | 0x40 // set version byte
	u[8] = (u[8] & 0x3f) | 0x80 // set high order byte 0b10{8,9,a,b}

	return u
}

// Parse attempts to extract a uuid from the string or returns an error.
func Parse(s string) (u UUID, err error) {
	if len(s) != 36 {
		return UUID{}, ErrUUIDInvalid
	}

	// create stack addresses for each section of the uuid.
	p := make([][]byte, 5)

	if _, err := fmt.Sscanf(s, format, &p[0], &p[1], &p[2], &p[3], &p[4]); err != nil {
		return u, err
	}

	copy(u[0:4], p[0])
	copy(u[4:6], p[1])
	copy(u[6:8], p[2])
	copy(u[8:10], p[3])
	copy(u[10:16], p[4])

	return
}

func (u UUID) String() string {
	return fmt.Sprintf(format, u[:4], u[4:6], u[6:8], u[8:10], u[10:])
}

// retryOnError tries to detect whether or not retrying would be fruitful.
func retryOnError(err error) bool {
	switch err := err.(type) {
	case *os.PathError:
		return retryOnError(err.Err) // unpack the target error
	case syscall.Errno:
		if err == syscall.EPERM {
			// EPERM represents an entropy pool exhaustion, a condition under
			// which we backoff and retry.
			return true
		}
	}

	return false
}
