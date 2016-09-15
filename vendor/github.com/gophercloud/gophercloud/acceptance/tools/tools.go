package tools

import (
	"crypto/rand"
	"errors"
	mrand "math/rand"
	"time"
)

// ErrTimeout is returned if WaitFor takes longer than 300 second to happen.
var ErrTimeout = errors.New("Timed out")

// WaitFor polls a predicate function once per second to wait for a certain state to arrive.
func WaitFor(predicate func() (bool, error)) error {
	for i := 0; i < 300; i++ {
		time.Sleep(1 * time.Second)

		satisfied, err := predicate()
		if err != nil {
			return err
		}
		if satisfied {
			return nil
		}
	}
	return ErrTimeout
}

// MakeNewPassword generates a new string that's guaranteed to be different than the given one.
func MakeNewPassword(oldPass string) string {
	randomPassword := RandomString("", 16)
	for randomPassword == oldPass {
		randomPassword = RandomString("", 16)
	}
	return randomPassword
}

// RandomString generates a string of given length, but random content.
// All content will be within the ASCII graphic character set.
// (Implementation from Even Shaw's contribution on
// http://stackoverflow.com/questions/12771930/what-is-the-fastest-way-to-generate-a-long-random-string-in-go).
func RandomString(prefix string, n int) string {
	const alphanum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	var bytes = make([]byte, n)
	rand.Read(bytes)
	for i, b := range bytes {
		bytes[i] = alphanum[b%byte(len(alphanum))]
	}
	return prefix + string(bytes)
}

// RandomInt will return a random integer between a specified range.
func RandomInt(min, max int) int {
	mrand.Seed(time.Now().Unix())
	return mrand.Intn(max-min) + min
}

// Elide returns the first bit of its input string with a suffix of "..." if it's longer than
// a comfortable 40 characters.
func Elide(value string) string {
	if len(value) > 40 {
		return value[0:37] + "..."
	}
	return value
}
