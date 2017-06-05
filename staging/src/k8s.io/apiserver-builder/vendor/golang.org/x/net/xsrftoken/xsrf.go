// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package xsrftoken provides methods for generating and validating secure XSRF tokens.
package xsrftoken // import "golang.org/x/net/xsrftoken"

import (
	"crypto/hmac"
	"crypto/sha1"
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Timeout is the duration for which XSRF tokens are valid.
// It is exported so clients may set cookie timeouts that match generated tokens.
const Timeout = 24 * time.Hour

// clean sanitizes a string for inclusion in a token by replacing all ":"s.
func clean(s string) string {
	return strings.Replace(s, ":", "_", -1)
}

// Generate returns a URL-safe secure XSRF token that expires in 24 hours.
//
// key is a secret key for your application.
// userID is a unique identifier for the user.
// actionID is the action the user is taking (e.g. POSTing to a particular path).
func Generate(key, userID, actionID string) string {
	return generateTokenAtTime(key, userID, actionID, time.Now())
}

// generateTokenAtTime is like Generate, but returns a token that expires 24 hours from now.
func generateTokenAtTime(key, userID, actionID string, now time.Time) string {
	// Round time up and convert to milliseconds.
	milliTime := (now.UnixNano() + 1e6 - 1) / 1e6

	h := hmac.New(sha1.New, []byte(key))
	fmt.Fprintf(h, "%s:%s:%d", clean(userID), clean(actionID), milliTime)

	// Get the padded base64 string then removing the padding.
	tok := string(h.Sum(nil))
	tok = base64.URLEncoding.EncodeToString([]byte(tok))
	tok = strings.TrimRight(tok, "=")

	return fmt.Sprintf("%s:%d", tok, milliTime)
}

// Valid reports whether a token is a valid, unexpired token returned by Generate.
func Valid(token, key, userID, actionID string) bool {
	return validTokenAtTime(token, key, userID, actionID, time.Now())
}

// validTokenAtTime reports whether a token is valid at the given time.
func validTokenAtTime(token, key, userID, actionID string, now time.Time) bool {
	// Extract the issue time of the token.
	sep := strings.LastIndex(token, ":")
	if sep < 0 {
		return false
	}
	millis, err := strconv.ParseInt(token[sep+1:], 10, 64)
	if err != nil {
		return false
	}
	issueTime := time.Unix(0, millis*1e6)

	// Check that the token is not expired.
	if now.Sub(issueTime) >= Timeout {
		return false
	}

	// Check that the token is not from the future.
	// Allow 1 minute grace period in case the token is being verified on a
	// machine whose clock is behind the machine that issued the token.
	if issueTime.After(now.Add(1 * time.Minute)) {
		return false
	}

	expected := generateTokenAtTime(key, userID, actionID, issueTime)

	// Check that the token matches the expected value.
	// Use constant time comparison to avoid timing attacks.
	return subtle.ConstantTimeCompare([]byte(token), []byte(expected)) == 1
}
