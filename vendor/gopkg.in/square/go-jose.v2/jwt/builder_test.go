/*-
 * Copyright 2016 Zbigniew Mandziejewicz
 * Copyright 2016 Square, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package jwt

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.in/square/go-jose.v2"
	"gopkg.in/square/go-jose.v2/json"
)

type testClaims struct {
	Subject string `json:"sub"`
}

type invalidMarshalClaims struct {
}

var errInvalidMarshalClaims = errors.New("Failed marshaling invalid claims.")

func (c invalidMarshalClaims) MarshalJSON() ([]byte, error) {
	return nil, errInvalidMarshalClaims
}

var sampleClaims = Claims{
	Subject:  "42",
	IssuedAt: NewNumericDate(time.Date(2016, 1, 1, 0, 0, 0, 0, time.UTC)),
	Issuer:   "issuer",
	Audience: Audience{"a1", "a2"},
}

type numberClaims struct {
	Int   int64   `json:"int"`
	Float float64 `json:"float"`
}

func TestIntegerAndFloatsNormalize(t *testing.T) {
	c := numberClaims{1 << 60, 12345.6789}

	normalized, err := normalize(c)
	if err != nil {
		t.Fatal(err)
	}

	ni, err := (normalized["int"].(json.Number)).Int64()
	nf, err := (normalized["float"].(json.Number)).Float64()

	if ni != c.Int {
		t.Error(fmt.Sprintf("normalize failed to preserve int64 (got %v, wanted %v, type %s)", normalized["int"], c.Int, reflect.TypeOf(normalized["int"])))
	}
	if nf != c.Float {
		t.Error(fmt.Sprintf("normalize failed to preserve float64 (got %v, wanted %v, type %s)", normalized["float"], c.Float, reflect.TypeOf(normalized["float"])))
	}
}

func TestBuilderCustomClaimsNonPointer(t *testing.T) {
	jwt, err := Signed(rsaSigner).Claims(testClaims{"foo"}).CompactSerialize()
	require.NoError(t, err, "Error creating JWT.")

	parsed, err := ParseSigned(jwt)
	require.NoError(t, err, "Error parsing JWT.")

	out := &testClaims{}
	if assert.NoError(t, parsed.Claims(&testPrivRSAKey1.PublicKey, out), "Error unmarshaling claims.") {
		assert.Equal(t, "foo", out.Subject)
	}
}

func TestBuilderCustomClaimsPointer(t *testing.T) {
	jwt, err := Signed(rsaSigner).Claims(&testClaims{"foo"}).CompactSerialize()
	require.NoError(t, err, "Error creating JWT.")

	parsed, err := ParseSigned(jwt)
	require.NoError(t, err, "Error parsing JWT.")

	out := &testClaims{}
	if assert.NoError(t, parsed.Claims(&testPrivRSAKey1.PublicKey, out), "Error unmarshaling claims.") {
		assert.Equal(t, "foo", out.Subject)
	}
}

func TestBuilderMergeClaims(t *testing.T) {
	jwt, err := Signed(rsaSigner).
		Claims(&Claims{
			Subject: "42",
		}).
		Claims(map[string]interface{}{
			"Scopes": []string{"read:users"},
		}).
		CompactSerialize()
	require.NoError(t, err, "Error creating JWT.")

	parsed, err := ParseSigned(jwt)
	require.NoError(t, err, "Error parsing JWT.")

	out := make(map[string]interface{})
	if assert.NoError(t, parsed.Claims(&testPrivRSAKey1.PublicKey, &out), "Error unmarshaling claims.") {
		assert.Equal(t, map[string]interface{}{
			"sub":    "42",
			"Scopes": []interface{}{"read:users"},
		}, out)
	}

	_, err = Signed(rsaSigner).Claims("invalid-claims").Claims(&testClaims{"foo"}).CompactSerialize()
	assert.Equal(t, err, ErrInvalidClaims)

	_, err = Signed(rsaSigner).Claims(&invalidMarshalClaims{}).CompactSerialize()
	assert.EqualError(t, err, "json: error calling MarshalJSON for type *jwt.invalidMarshalClaims: Failed marshaling invalid claims.")
}

func TestSignedFullSerializeAndToken(t *testing.T) {
	b := Signed(rsaSigner).Claims(&testClaims{"foo"})

	jwt, err := b.FullSerialize()
	require.NoError(t, err, "Error creating JWT.")
	parsed, err := ParseSigned(jwt)
	require.NoError(t, err, "Error parsing JWT.")
	out := &testClaims{}
	if assert.NoError(t, parsed.Claims(&testPrivRSAKey1.PublicKey, &out), "Error unmarshaling claims.") {
		assert.Equal(t, &testClaims{
			Subject: "foo",
		}, out)
	}

	jwt2, err := b.Token()
	require.NoError(t, err, "Error creating JWT.")
	out2 := &testClaims{}
	if assert.NoError(t, jwt2.Claims(&testPrivRSAKey1.PublicKey, &out2), "Error unmarshaling claims.") {
		assert.Equal(t, &testClaims{
			Subject: "foo",
		}, out2)
	}

	b2 := Signed(rsaSigner).Claims(&invalidMarshalClaims{})
	_, err = b2.FullSerialize()
	require.EqualError(t, err, "json: error calling MarshalJSON for type *jwt.invalidMarshalClaims: Failed marshaling invalid claims.")
	_, err = b2.Token()
	require.EqualError(t, err, "json: error calling MarshalJSON for type *jwt.invalidMarshalClaims: Failed marshaling invalid claims.")
}

func TestEncryptedFullSerializeAndToken(t *testing.T) {
	recipient := jose.Recipient{
		Algorithm: jose.RSA1_5,
		Key:       testPrivRSAKey1.Public(),
	}
	encrypter, err := jose.NewEncrypter(jose.A128CBC_HS256, recipient, nil)
	require.NoError(t, err, "Error creating encrypter.")

	b := Encrypted(encrypter).Claims(&testClaims{"foo"})

	jwt, err := b.FullSerialize()
	require.NoError(t, err, "Error creating JWT.")
	parsed, err := ParseEncrypted(jwt)
	require.NoError(t, err, "Error parsing JWT.")
	out := &testClaims{}
	if assert.NoError(t, parsed.Claims(testPrivRSAKey1, &out)) {
		assert.Equal(t, &testClaims{
			Subject: "foo",
		}, out)
	}

	jwt2, err := b.Token()
	require.NoError(t, err, "Error creating JWT.")
	out2 := &testClaims{}
	if assert.NoError(t, jwt2.Claims(testPrivRSAKey1, &out2)) {
		assert.Equal(t, &testClaims{
			Subject: "foo",
		}, out2)
	}

	b2 := Encrypted(encrypter).Claims(&invalidMarshalClaims{})

	_, err = b2.FullSerialize()
	require.EqualError(t, err, "json: error calling MarshalJSON for type *jwt.invalidMarshalClaims: Failed marshaling invalid claims.")
	_, err = b2.Token()
	require.EqualError(t, err, "json: error calling MarshalJSON for type *jwt.invalidMarshalClaims: Failed marshaling invalid claims.")
}

func TestBuilderSignedAndEncrypted(t *testing.T) {
	recipient := jose.Recipient{
		Algorithm: jose.RSA1_5,
		Key:       testPrivRSAKey1.Public(),
	}
	encrypter, err := jose.NewEncrypter(jose.A128CBC_HS256, recipient, (&jose.EncrypterOptions{}).WithContentType("JWT").WithType("JWT"))
	require.NoError(t, err, "Error creating encrypter.")

	jwt1, err := SignedAndEncrypted(rsaSigner, encrypter).Claims(&testClaims{"foo"}).Token()
	require.NoError(t, err, "Error marshaling signed-then-encrypted token.")
	if nested, err := jwt1.Decrypt(testPrivRSAKey1); assert.NoError(t, err, "Error decrypting signed-then-encrypted token.") {
		out := &testClaims{}
		assert.NoError(t, nested.Claims(&testPrivRSAKey1.PublicKey, out))
		assert.Equal(t, &testClaims{"foo"}, out)
	}

	b := SignedAndEncrypted(rsaSigner, encrypter).Claims(&testClaims{"foo"})
	tok1, err := b.CompactSerialize()
	if assert.NoError(t, err) {
		jwt, err := ParseSignedAndEncrypted(tok1)
		if assert.NoError(t, err, "Error parsing signed-then-encrypted compact token.") {
			if nested, err := jwt.Decrypt(testPrivRSAKey1); assert.NoError(t, err) {
				out := &testClaims{}
				assert.NoError(t, nested.Claims(&testPrivRSAKey1.PublicKey, out))
				assert.Equal(t, &testClaims{"foo"}, out)
			}
		}
	}

	tok2, err := b.FullSerialize()
	if assert.NoError(t, err) {
		jwe, err := ParseSignedAndEncrypted(tok2)
		if assert.NoError(t, err, "Error parsing signed-then-encrypted full token.") {
			assert.Equal(t, []jose.Header{{
				Algorithm: string(jose.RSA1_5),
				ExtraHeaders: map[jose.HeaderKey]interface{}{
					jose.HeaderType:        "JWT",
					jose.HeaderContentType: "JWT",
					"enc": "A128CBC-HS256",
				},
			}}, jwe.Headers)
			if jws, err := jwe.Decrypt(testPrivRSAKey1); assert.NoError(t, err) {
				assert.Equal(t, []jose.Header{{
					Algorithm: string(jose.RS256),
					ExtraHeaders: map[jose.HeaderKey]interface{}{
						jose.HeaderType: "JWT",
					},
				}}, jws.Headers)
				out := &testClaims{}
				assert.NoError(t, jws.Claims(&testPrivRSAKey1.PublicKey, out))
				assert.Equal(t, &testClaims{"foo"}, out)
			}
		}
	}

	b2 := SignedAndEncrypted(rsaSigner, encrypter).Claims(&invalidMarshalClaims{})
	_, err = b2.CompactSerialize()
	assert.EqualError(t, err, "json: error calling MarshalJSON for type *jwt.invalidMarshalClaims: Failed marshaling invalid claims.")
	_, err = b2.FullSerialize()
	assert.EqualError(t, err, "json: error calling MarshalJSON for type *jwt.invalidMarshalClaims: Failed marshaling invalid claims.")

	encrypter2, err := jose.NewEncrypter(jose.A128CBC_HS256, recipient, nil)
	require.NoError(t, err, "Error creating encrypter.")
	_, err = SignedAndEncrypted(rsaSigner, encrypter2).CompactSerialize()
	assert.EqualError(t, err, "square/go-jose/jwt: expected content type to be JWT (cty header)")
}

func TestBuilderHeadersSigner(t *testing.T) {
	tests := []struct {
		Keys   []*rsa.PrivateKey
		Claims interface{}
	}{
		{
			Keys:   []*rsa.PrivateKey{testPrivRSAKey1},
			Claims: &Claims{Issuer: "foo"},
		},
		{
			Keys:   []*rsa.PrivateKey{testPrivRSAKey1, testPrivRSAKey2},
			Claims: &Claims{Issuer: "foo"},
		},
	}

	for i, tc := range tests {
		wantKeyIDs := make([]string, len(tc.Keys))
		signingKeys := make([]jose.SigningKey, len(tc.Keys))

		for j, key := range tc.Keys {
			keyIDBytes := make([]byte, 20)
			if _, err := io.ReadFull(rand.Reader, keyIDBytes); err != nil {
				t.Fatalf("failed to read random bytes: %v", err)
			}
			keyID := hex.EncodeToString(keyIDBytes)

			wantKeyIDs[j] = keyID
			signingKeys[j] = jose.SigningKey{
				Algorithm: jose.RS256,
				Key: &jose.JSONWebKey{
					KeyID:     keyID,
					Algorithm: "RSA",
					Key:       key,
				},
			}
		}

		signer, err := jose.NewMultiSigner(signingKeys, nil)
		if err != nil {
			t.Errorf("case %d: NewMultiSigner(): %v", i, err)
			continue
		}

		var token string
		if len(tc.Keys) == 1 {
			token, err = Signed(signer).Claims(tc.Claims).CompactSerialize()
		} else {
			token, err = Signed(signer).Claims(tc.Claims).FullSerialize()
		}
		if err != nil {
			t.Errorf("case %d: failed to create token: %v", i, err)
			continue
		}
		jws, err := jose.ParseSigned(token)
		if err != nil {
			t.Errorf("case %d: parse signed: %v", i, err)
			continue
		}
		gotKeyIDs := make([]string, len(jws.Signatures))
		for i, sig := range jws.Signatures {
			gotKeyIDs[i] = sig.Header.KeyID
		}
		sort.Strings(wantKeyIDs)
		sort.Strings(gotKeyIDs)
		if !reflect.DeepEqual(wantKeyIDs, gotKeyIDs) {
			t.Errorf("case %d: wanted=%q got=%q", i, wantKeyIDs, gotKeyIDs)
		}
	}
}

func TestBuilderHeadersEncrypter(t *testing.T) {
	key := testPrivRSAKey1
	claims := &Claims{Issuer: "foo"}

	keyIDBytes := make([]byte, 20)
	if _, err := io.ReadFull(rand.Reader, keyIDBytes); err != nil {
		t.Fatalf("failed to read random bytes: %v", err)
	}
	keyID := hex.EncodeToString(keyIDBytes)

	wantKeyID := keyID
	recipient := jose.Recipient{
		Algorithm: jose.RSA1_5,
		Key:       key.Public(),
		KeyID:     keyID,
	}

	wantType := jose.ContentType("JWT")
	encrypter, err := jose.NewEncrypter(jose.A128CBC_HS256, recipient, (&jose.EncrypterOptions{}).WithType(wantType))
	require.NoError(t, err, "failed to create encrypter")

	token, err := Encrypted(encrypter).Claims(claims).CompactSerialize()
	require.NoError(t, err, "failed to create token")

	jwe, err := jose.ParseEncrypted(token)
	if assert.NoError(t, err, "error parsing encrypted token") {
		assert.Equal(t, jose.Header{
			ExtraHeaders: map[jose.HeaderKey]interface{}{
				jose.HeaderType: string(wantType),
				"enc":           "A128CBC-HS256",
			},
			Algorithm: string(jose.RSA1_5),
			KeyID:     wantKeyID,
		}, jwe.Header)
	}
}

func BenchmarkMapClaims(b *testing.B) {
	m := map[string]interface{}{
		"sub": "42",
		"iat": 1451606400,
		"iss": "issuer",
		"aud": []string{"a1", "a2"},
	}

	for i := 0; i < b.N; i++ {
		Signed(rsaSigner).Claims(m)
	}
}

func BenchmarkStructClaims(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Signed(rsaSigner).Claims(sampleClaims)
	}
}

func BenchmarkSignedCompactSerializeRSA(b *testing.B) {
	tb := Signed(rsaSigner).Claims(sampleClaims)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tb.CompactSerialize()
	}
}

func BenchmarkSignedCompactSerializeSHA(b *testing.B) {
	tb := Signed(hmacSigner).Claims(sampleClaims)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tb.CompactSerialize()
	}
}

func mustUnmarshalRSA(data string) *rsa.PrivateKey {
	block, _ := pem.Decode([]byte(data))
	if block == nil {
		panic("failed to decode PEM data")
	}
	key, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		panic("failed to parse RSA key: " + err.Error())
	}
	if key, ok := key.(*rsa.PrivateKey); ok {
		return key
	}
	panic("key is not of type *rsa.PrivateKey")
}

func mustMakeSigner(alg jose.SignatureAlgorithm, k interface{}) jose.Signer {
	sig, err := jose.NewSigner(jose.SigningKey{Algorithm: alg, Key: k}, (&jose.SignerOptions{}).WithType("JWT"))
	if err != nil {
		panic("failed to create signer:" + err.Error())
	}

	return sig
}

var (
	sharedKey           = []byte("secret")
	sharedEncryptionKey = []byte("itsa16bytesecret")

	testPrivRSAKey1 = mustUnmarshalRSA(`-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDIHBvDHAr7jh8h
xaqBCl11fjI9YZtdC5b3HtXTXZW3c2dIOImNUjffT8POP6p5OpzivmC1om7iOyuZ
3nJjC9LT3zqqs3f2i5d4mImxEuqG6uWdryFfkp0uIv5VkjVO+iQWd6pDAPGP7r1Z
foXCleyCtmyNH4JSkJneNPOk/4BxO8vcvRnCMT/Gv81IT6H+OQ6OovWOuJr8RX9t
1wuCjC9ezZxeI9ONffhiO5FMrVh5H9LJTl3dPOVa4aEcOvgd45hBmvxAyXqf8daE
6Kl2O7vQ4uwgnSTVXYIIjCjbepuersApIMGx/XPSgiU1K3Xtah/TBvep+S3VlwPc
q/QH25S9AgMBAAECggEAe+y8XKYfPw4SxY1uPB+5JSwT3ON3nbWxtjSIYy9Pqp5z
Vcx9kuFZ7JevQSk4X38m7VzM8282kC/ono+d8yy9Uayq3k/qeOqV0X9Vti1qxEbw
ECkG1/MqGApfy4qSLOjINInDDV+mOWa2KJgsKgdCwuhKbVMYGB2ozG2qfYIlfvlY
vLcBEpGWmswJHNmkcjTtGFIyJgPbsI6ndkkOeQbqQKAaadXtG1xUzH+vIvqaUl/l
AkNf+p4qhPkHsoAWXf1qu9cYa2T8T+mEo79AwlgVC6awXQWNRTiyClDJC7cu6NBy
ZHXCLFMbalzWF9qeI2OPaFX2x3IBWrbyDxcJ4TSdQQKBgQD/Fp/uQonMBh1h4Vi4
HlxZdqSOArTitXValdLFGVJ23MngTGV/St4WH6eRp4ICfPyldsfcv6MZpNwNm1Rn
lB5Gtpqpby1dsrOSfvVbY7U3vpLnd8+hJ/lT5zCYt5Eor46N6iWRkYWzNe4PixiF
z1puGUvFCbZdeeACVrPLmW3JKQKBgQDI0y9WTf8ezKPbtap4UEE6yBf49ftohVGz
p4iD6Ng1uqePwKahwoVXKOc179CjGGtW/UUBORAoKRmxdHajHq6LJgsBxpaARz21
COPy99BUyp9ER5P8vYn63lC7Cpd/K7uyMjaz1DAzYBZIeVZHIw8O9wuGNJKjRFy9
SZyD3V0ddQKBgFMdohrWH2QVEfnUnT3Q1rJn0BJdm2bLTWOosbZ7G72TD0xAWEnz
sQ1wXv88n0YER6X6YADziEdQykq8s/HT91F/KkHO8e83zP8M0xFmGaQCOoelKEgQ
aFMIX3NDTM7+9OoUwwz9Z50PE3SJFAJ1n7eEEoYvNfabQXxBl+/dHEKRAoGAPEvU
EaiXacrtg8EWrssB2sFLGU/ZrTciIbuybFCT4gXp22pvXXAHEvVP/kzDqsRhLhwb
BNP6OuSkNziNikpjA5pngZ/7fgZly54gusmW/m5bxWdsUl0iOXVYbeAvPlqGH2me
LP4Pfs1hw17S/cbT9Z1NE31jbavP4HFikeD73SUCgYEArQfuudml6ei7XZ1Emjq8
jZiD+fX6e6BD/ISatVnuyZmGj9wPFsEhY2BpLiAMQHMDIvH9nlKzsFvjkTPB86qG
jCh3D67Os8eSBk5uRC6iW3Fc4DXvB5EFS0W9/15Sl+V5vXAcrNMpYS82OTSMG2Gt
b9Ym/nxaqyTu0PxajXkKm5Q=
-----END PRIVATE KEY-----`)

	testPrivRSAKey2 = mustUnmarshalRSA(`-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCxJ09jkXZ5Okyq
FrEKrs+GTzZRvoLziyzDTIZLJC6BVryau4gaFjuBG+pnm4z53oDP0XVnjFsx1mBw
R6RHeXlXbxLXsMfJpMzU9I2SRen9DokpD187CAnjLOoN9QRl1h8CA+sqR5Jw9mdl
mdaBKC99M9QYAPK3vGNfPC4soo8LDSBiemmt5raL4WSfoYh/6qg5rHUTymY28uxV
ew3I9Yp+3ltIw+WlRDtW5l+MM5CSUofjj2zcgcG3LEuPtvyZ+CSObxxcZZugm9zc
JdiazNyUxtX8yAj3Xg8Hde0jt0QDXv7A+U0KMVi9lX6PJEaNj4tOhOmQhJVMzAyr
1W/bifZVAgMBAAECggEAduKnn21GMZLTUi4KP94SvNK55F/Sp7hVoPbhBNpSL1BT
IBAMBV24LyvZwhAcqq8MiOrLPGNv6+EvNQqPD7xQl0GeRouHeCYVpDA+NdSfc8jm
eVysjwQVBpTkudsdSW5JvuN8VRJVD2P8/a0gy+p4/C/k/Prd6DoQAiBz6FZrYoEd
iYgIegHOMXWd4vzO3ENOWSIUI6ci7Aro+Y0Z75kfiVokAGhUcFgrZ58E82fBYh8I
cxO20oMnucGrLicQzj536jx4wX3Cdd4jr9UVEJ9ZII1ldlp03nZlFLXqJH1547Aq
ZM+3vVcBGoJ8T9ZQ4VDAL++0K2DLC9JkTARAYCEi/QKBgQDebIc1+2zblhQtVQ/e
IbEErZcB7v+TkUoRoBfR0lj7bKBFJgRe37fgu1xf95/s63okdnOw/OuQqtGmgx/J
TL3yULBdNcwTCRm41t+cqoGymjK0VRbqk6CWBId0E3r5TaCVWedk2JI2XwTvIJ1A
eDiqfJeDHUD44yaonwbysj9ZDwKBgQDL5VQfTppVaJk2PXNwhAkRQklZ8RFmt/7p
yA3dddQNdwMk4Fl8F7QuO1gBxDiHdnwIrlEOz6fTsM3LwIS+Q12P1vYFIhpo7HDB
wvjfMwCPxBIS4jI28RgcAf0VbZ/+CHAm6bb9iDwsjXhh1J5oOm5VKnju6/rPH/QY
+md40pnSWwKBgBnKPbdNquafNUG4XjmkcHEZa6wGuU20CAGZLYnfuP+WLdM2wET7
7cc6ElDyVnHTL/twXKPF/85rcBm9lH7zzgZ9wqVcKoh+gqQDDjSNNLKv3Hc6cojK
i1E5vzb/Vz/290q5/PGdhv6U7+6GOpWSGwfxoGPMjY8OT5o3rkeP0XaTAoGBALLR
GQmr4eZtqZDMK+XNpjYgsDvVE7HGRCW7cY17vNFiQruglloiX778BJ7n+7uxye3D
EwuuSj15ncLHwKMsaW2w1GqEEi1azzjfSWxWSnPLPR6aifdtUfueMtsMHXio5dL6
vaV0SXG5UI5b7eDy/bhrW0wOYRQtreIKGZz49jZpAoGBAIvxYngkLwmq6g6MmnAc
YK4oT6YAm2wfSy2mzpEQP5r1igp1rN7T46o7FMUPDLS9wK3ESAaIYe01qT6Yftcc
5qF+yiOGDTr9XQiHwe4BcyrNEMfUjDhDU5ao2gH8+t1VGr1KspLsUNbedrJwZsY4
UCZVKEEDHzKfLO/iBgKjJQF7
-----END PRIVATE KEY-----`)

	rsaSigner  = mustMakeSigner(jose.RS256, testPrivRSAKey1)
	hmacSigner = mustMakeSigner(jose.HS256, sharedKey)
)
