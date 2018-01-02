package handlers

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"
)

// blobUploadState captures the state serializable state of the blob upload.
type blobUploadState struct {
	// name is the primary repository under which the blob will be linked.
	Name string

	// UUID identifies the upload.
	UUID string

	// offset contains the current progress of the upload.
	Offset int64

	// StartedAt is the original start time of the upload.
	StartedAt time.Time
}

type hmacKey string

var errInvalidSecret = fmt.Errorf("invalid secret")

// unpackUploadState unpacks and validates the blob upload state from the
// token, using the hmacKey secret.
func (secret hmacKey) unpackUploadState(token string) (blobUploadState, error) {
	var state blobUploadState

	tokenBytes, err := base64.URLEncoding.DecodeString(token)
	if err != nil {
		return state, err
	}
	mac := hmac.New(sha256.New, []byte(secret))

	if len(tokenBytes) < mac.Size() {
		return state, errInvalidSecret
	}

	macBytes := tokenBytes[:mac.Size()]
	messageBytes := tokenBytes[mac.Size():]

	mac.Write(messageBytes)
	if !hmac.Equal(mac.Sum(nil), macBytes) {
		return state, errInvalidSecret
	}

	if err := json.Unmarshal(messageBytes, &state); err != nil {
		return state, err
	}

	return state, nil
}

// packUploadState packs the upload state signed with and hmac digest using
// the hmacKey secret, encoding to url safe base64. The resulting token can be
// used to share data with minimized risk of external tampering.
func (secret hmacKey) packUploadState(lus blobUploadState) (string, error) {
	mac := hmac.New(sha256.New, []byte(secret))
	p, err := json.Marshal(lus)
	if err != nil {
		return "", err
	}

	mac.Write(p)

	return base64.URLEncoding.EncodeToString(append(mac.Sum(nil), p...)), nil
}
