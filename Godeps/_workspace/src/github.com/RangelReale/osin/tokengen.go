package osin

import (
	"code.google.com/p/go-uuid/uuid"
	"encoding/base64"
)

// AuthorizeTokenGenDefault is the default authorization token generator
type AuthorizeTokenGenDefault struct {
}

// GenerateAuthorizeToken generates a base64-encoded UUID code
func (a *AuthorizeTokenGenDefault) GenerateAuthorizeToken(data *AuthorizeData) (ret string, err error) {
	token := uuid.New()
	return base64.StdEncoding.EncodeToString([]byte(token)), nil
}

// AccessTokenGenDefault is the default authorization token generator
type AccessTokenGenDefault struct {
}

// GenerateAccessToken generates base64-encoded UUID access and refresh tokens
func (a *AccessTokenGenDefault) GenerateAccessToken(data *AccessData, generaterefresh bool) (accesstoken string, refreshtoken string, err error) {
	accesstoken = uuid.New()
	accesstoken = base64.StdEncoding.EncodeToString([]byte(accesstoken))

	if generaterefresh {
		refreshtoken = uuid.New()
		refreshtoken = base64.StdEncoding.EncodeToString([]byte(refreshtoken))
	}
	return
}
