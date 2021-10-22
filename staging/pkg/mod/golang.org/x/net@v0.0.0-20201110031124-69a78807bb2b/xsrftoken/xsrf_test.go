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
	if !validTokenAtTime(tok, key, userID, actionID, oneMinuteFromNow, Timeout) {
		t.Error("One second later: Expected token to be valid")
	}
	if !validTokenAtTime(tok, key, userID, actionID, now.Add(Timeout-1*time.Nanosecond), Timeout) {
		t.Error("Just before timeout: Expected token to be valid")
	}
	if !validTokenAtTime(tok, key, userID, actionID, now.Add(-1*time.Minute+1*time.Millisecond), Timeout) {
		t.Error("One minute in the past: Expected token to be valid")
	}
	if !validTokenAtTime(tok, key, userID, actionID, oneMinuteFromNow, time.Hour) {
		t.Error("One second later: Expected token to be valid")
	}
	if !validTokenAtTime(tok, key, userID, actionID, now.Add(time.Minute-1*time.Nanosecond), time.Minute) {
		t.Error("Just before timeout: Expected token to be valid")
	}
	if !validTokenAtTime(tok, key, userID, actionID, now.Add(-1*time.Minute+1*time.Millisecond), time.Hour) {
		t.Error("One minute in the past: Expected token to be valid")
	}
}

// TestSeparatorReplacement tests that separators are being correctly substituted
func TestSeparatorReplacement(t *testing.T) {
	separatorTests := []struct {
		name   string
		token1 string
		token2 string
	}{
		{
			"Colon",
			generateTokenAtTime("foo:bar", "baz", "wah", now),
			generateTokenAtTime("foo", "bar:baz", "wah", now),
		},
		{
			"Colon and Underscore",
			generateTokenAtTime("key", ":foo:", "wah", now),
			generateTokenAtTime("key", "_foo_", "wah", now),
		},
		{
			"Colon and Double Colon",
			generateTokenAtTime("key", ":foo:", "wah", now),
			generateTokenAtTime("key", "::foo::", "wah", now),
		},
	}

	for _, st := range separatorTests {
		if st.token1 == st.token2 {
			t.Errorf("%v: Expected generated tokens to be different", st.name)
		}
	}
}

func TestInvalidToken(t *testing.T) {
	invalidTokenTests := []struct {
		name, key, userID, actionID string
		t                           time.Time
		timeout                     time.Duration
	}{
		{"Bad key", "foobar", userID, actionID, oneMinuteFromNow, Timeout},
		{"Bad userID", key, "foobar", actionID, oneMinuteFromNow, Timeout},
		{"Bad actionID", key, userID, "foobar", oneMinuteFromNow, Timeout},
		{"Expired", key, userID, actionID, now.Add(Timeout + 1*time.Millisecond), Timeout},
		{"More than 1 minute from the future", key, userID, actionID, now.Add(-1*time.Nanosecond - 1*time.Minute), Timeout},
		{"Expired with 1 minute timeout", key, userID, actionID, now.Add(time.Minute + 1*time.Millisecond), time.Minute},
	}

	tok := generateTokenAtTime(key, userID, actionID, now)
	for _, itt := range invalidTokenTests {
		if validTokenAtTime(tok, itt.key, itt.userID, itt.actionID, itt.t, itt.timeout) {
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
		if validTokenAtTime(bdt.tok, key, userID, actionID, oneMinuteFromNow, Timeout) {
			t.Errorf("%v: Expected token to be invalid", bdt.name)
		}
	}
}
