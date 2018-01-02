// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xsrftoken

import (
	"encoding/base64"
	"testing"
	"time"
)

const (
	key      = "quay"
	userID   = "12345678"
	actionID = "POST /form"
)

var (
	now              = time.Now()
	oneMinuteFromNow = now.Add(1 * time.Minute)
)

func TestValidToken(t *testing.T) {
	tok := generateTokenAtTime(key, userID, actionID, now)
	if !validTokenAtTime(tok, key, userID, actionID, oneMinuteFromNow) {
		t.Error("One second later: Expected token to be valid")
	}
	if !validTokenAtTime(tok, key, userID, actionID, now.Add(Timeout-1*time.Nanosecond)) {
		t.Error("Just before timeout: Expected token to be valid")
	}
	if !validTokenAtTime(tok, key, userID, actionID, now.Add(-1*time.Minute+1*time.Millisecond)) {
		t.Error("One minute in the past: Expected token to be valid")
	}
}

// TestSeparatorReplacement tests that separators are being correctly substituted
func TestSeparatorReplacement(t *testing.T) {
	tok := generateTokenAtTime("foo:bar", "baz", "wah", now)
	tok2 := generateTokenAtTime("foo", "bar:baz", "wah", now)
	if tok == tok2 {
		t.Errorf("Expected generated tokens to be different")
	}
}

func TestInvalidToken(t *testing.T) {
	invalidTokenTests := []struct {
		name, key, userID, actionID string
		t                           time.Time
	}{
		{"Bad key", "foobar", userID, actionID, oneMinuteFromNow},
		{"Bad userID", key, "foobar", actionID, oneMinuteFromNow},
		{"Bad actionID", key, userID, "foobar", oneMinuteFromNow},
		{"Expired", key, userID, actionID, now.Add(Timeout + 1*time.Millisecond)},
		{"More than 1 minute from the future", key, userID, actionID, now.Add(-1*time.Nanosecond - 1*time.Minute)},
	}

	tok := generateTokenAtTime(key, userID, actionID, now)
	for _, itt := range invalidTokenTests {
		if validTokenAtTime(tok, itt.key, itt.userID, itt.actionID, itt.t) {
			t.Errorf("%v: Expected token to be invalid", itt.name)
		}
	}
}

// TestValidateBadData primarily tests that no unexpected panics are triggered
// during parsing
func TestValidateBadData(t *testing.T) {
	badDataTests := []struct {
		name, tok string
	}{
		{"Invalid Base64", "ASDab24(@)$*=="},
		{"No delimiter", base64.URLEncoding.EncodeToString([]byte("foobar12345678"))},
		{"Invalid time", base64.URLEncoding.EncodeToString([]byte("foobar:foobar"))},
		{"Wrong length", "1234" + generateTokenAtTime(key, userID, actionID, now)},
	}

	for _, bdt := range badDataTests {
		if validTokenAtTime(bdt.tok, key, userID, actionID, oneMinuteFromNow) {
			t.Errorf("%v: Expected token to be invalid", bdt.name)
		}
	}
}
