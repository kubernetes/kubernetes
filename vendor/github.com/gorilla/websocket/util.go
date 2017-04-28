// Copyright 2013 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package websocket

import (
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"io"
	"net/http"
	"strings"
)

// tokenListContainsValue returns true if the 1#token header with the given
// name contains token.
func tokenListContainsValue(header http.Header, name string, value string) bool {
	for _, v := range header[name] {
		for _, s := range strings.Split(v, ",") {
			if strings.EqualFold(value, strings.TrimSpace(s)) {
				return true
			}
		}
	}
	return false
}

var keyGUID = []byte("258EAFA5-E914-47DA-95CA-C5AB0DC85B11")

func computeAcceptKey(challengeKey string) string {
	h := sha1.New()
	h.Write([]byte(challengeKey))
	h.Write(keyGUID)
	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

func generateChallengeKey() (string, error) {
	p := make([]byte, 16)
	if _, err := io.ReadFull(rand.Reader, p); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(p), nil
}
