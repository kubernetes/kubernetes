// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jwt

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

var dummyPrivateKey = []byte(`-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAx4fm7dngEmOULNmAs1IGZ9Apfzh+BkaQ1dzkmbUgpcoghucE
DZRnAGd2aPyB6skGMXUytWQvNYav0WTR00wFtX1ohWTfv68HGXJ8QXCpyoSKSSFY
fuP9X36wBSkSX9J5DVgiuzD5VBdzUISSmapjKm+DcbRALjz6OUIPEWi1Tjl6p5RK
1w41qdbmt7E5/kGhKLDuT7+M83g4VWhgIvaAXtnhklDAggilPPa8ZJ1IFe31lNlr
k4DRk38nc6sEutdf3RL7QoH7FBusI7uXV03DC6dwN1kP4GE7bjJhcRb/7jYt7CQ9
/E9Exz3c0yAp0yrTg0Fwh+qxfH9dKwN52S7SBwIDAQABAoIBAQCaCs26K07WY5Jt
3a2Cw3y2gPrIgTCqX6hJs7O5ByEhXZ8nBwsWANBUe4vrGaajQHdLj5OKfsIDrOvn
2NI1MqflqeAbu/kR32q3tq8/Rl+PPiwUsW3E6Pcf1orGMSNCXxeducF2iySySzh3
nSIhCG5uwJDWI7a4+9KiieFgK1pt/Iv30q1SQS8IEntTfXYwANQrfKUVMmVF9aIK
6/WZE2yd5+q3wVVIJ6jsmTzoDCX6QQkkJICIYwCkglmVy5AeTckOVwcXL0jqw5Kf
5/soZJQwLEyBoQq7Kbpa26QHq+CJONetPP8Ssy8MJJXBT+u/bSseMb3Zsr5cr43e
DJOhwsThAoGBAPY6rPKl2NT/K7XfRCGm1sbWjUQyDShscwuWJ5+kD0yudnT/ZEJ1
M3+KS/iOOAoHDdEDi9crRvMl0UfNa8MAcDKHflzxg2jg/QI+fTBjPP5GOX0lkZ9g
z6VePoVoQw2gpPFVNPPTxKfk27tEzbaffvOLGBEih0Kb7HTINkW8rIlzAoGBAM9y
1yr+jvfS1cGFtNU+Gotoihw2eMKtIqR03Yn3n0PK1nVCDKqwdUqCypz4+ml6cxRK
J8+Pfdh7D+ZJd4LEG6Y4QRDLuv5OA700tUoSHxMSNn3q9As4+T3MUyYxWKvTeu3U
f2NWP9ePU0lV8ttk7YlpVRaPQmc1qwooBA/z/8AdAoGAW9x0HWqmRICWTBnpjyxx
QGlW9rQ9mHEtUotIaRSJ6K/F3cxSGUEkX1a3FRnp6kPLcckC6NlqdNgNBd6rb2rA
cPl/uSkZP42Als+9YMoFPU/xrrDPbUhu72EDrj3Bllnyb168jKLa4VBOccUvggxr
Dm08I1hgYgdN5huzs7y6GeUCgYEAj+AZJSOJ6o1aXS6rfV3mMRve9bQ9yt8jcKXw
5HhOCEmMtaSKfnOF1Ziih34Sxsb7O2428DiX0mV/YHtBnPsAJidL0SdLWIapBzeg
KHArByIRkwE6IvJvwpGMdaex1PIGhx5i/3VZL9qiq/ElT05PhIb+UXgoWMabCp84
OgxDK20CgYAeaFo8BdQ7FmVX2+EEejF+8xSge6WVLtkaon8bqcn6P0O8lLypoOhd
mJAYH8WU+UAy9pecUnDZj14LAGNVmYcse8HFX71MoshnvCTFEPVo4rZxIAGwMpeJ
5jgQ3slYLpqrGlcbLgUXBUgzEO684Wk/UV9DFPlHALVqCfXQ9dpJPg==
-----END RSA PRIVATE KEY-----`)

func TestJWTFetch_JSONResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"access_token": "90d64460d14870c08c81352a05dedd3465940a7c",
			"scope": "user",
			"token_type": "bearer",
			"expires_in": 3600
		}`))
	}))
	defer ts.Close()

	conf := &Config{
		Email:      "aaa@xxx.com",
		PrivateKey: dummyPrivateKey,
		TokenURL:   ts.URL,
	}
	tok, err := conf.TokenSource(context.Background()).Token()
	if err != nil {
		t.Fatal(err)
	}
	if !tok.Valid() {
		t.Errorf("got invalid token: %v", tok)
	}
	if got, want := tok.AccessToken, "90d64460d14870c08c81352a05dedd3465940a7c"; got != want {
		t.Errorf("access token = %q; want %q", got, want)
	}
	if got, want := tok.TokenType, "bearer"; got != want {
		t.Errorf("token type = %q; want %q", got, want)
	}
	if got := tok.Expiry.IsZero(); got {
		t.Errorf("token expiry = %v, want none", got)
	}
	scope := tok.Extra("scope")
	if got, want := scope, "user"; got != want {
		t.Errorf("scope = %q; want %q", got, want)
	}
}

func TestJWTFetch_BadResponse(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"scope": "user", "token_type": "bearer"}`))
	}))
	defer ts.Close()

	conf := &Config{
		Email:      "aaa@xxx.com",
		PrivateKey: dummyPrivateKey,
		TokenURL:   ts.URL,
	}
	tok, err := conf.TokenSource(context.Background()).Token()
	if err != nil {
		t.Fatal(err)
	}
	if tok == nil {
		t.Fatalf("got nil token; want token")
	}
	if tok.Valid() {
		t.Errorf("got invalid token: %v", tok)
	}
	if got, want := tok.AccessToken, ""; got != want {
		t.Errorf("access token = %q; want %q", got, want)
	}
	if got, want := tok.TokenType, "bearer"; got != want {
		t.Errorf("token type = %q; want %q", got, want)
	}
	scope := tok.Extra("scope")
	if got, want := scope, "user"; got != want {
		t.Errorf("token scope = %q; want %q", got, want)
	}
}

func TestJWTFetch_BadResponseType(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"access_token":123, "scope": "user", "token_type": "bearer"}`))
	}))
	defer ts.Close()
	conf := &Config{
		Email:      "aaa@xxx.com",
		PrivateKey: dummyPrivateKey,
		TokenURL:   ts.URL,
	}
	tok, err := conf.TokenSource(context.Background()).Token()
	if err == nil {
		t.Error("got a token; expected error")
		if got, want := tok.AccessToken, ""; got != want {
			t.Errorf("access token = %q; want %q", got, want)
		}
	}
}
