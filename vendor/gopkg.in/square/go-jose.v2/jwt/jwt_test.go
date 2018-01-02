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
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	hmacSignedToken                = `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzdWJqZWN0IiwiaXNzIjoiaXNzdWVyIiwic2NvcGVzIjpbInMxIiwiczIiXX0.Y6_PfQHrzRJ_Vlxij5VI07-pgDIuJNN3Z_g5sSaGQ0c`
	rsaSignedToken                 = `eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpc3N1ZXIiLCJzY29wZXMiOlsiczEiLCJzMiJdLCJzdWIiOiJzdWJqZWN0In0.UDDtyK9gC9kyHltcP7E_XODsnqcJWZIiXeGmSAH7SE9YKy3N0KSfFIN85dCNjTfs6zvy4rkrCHzLB7uKAtzMearh3q7jL4nxbhUMhlUcs_9QDVoN4q_j58XmRqBqRnBk-RmDu9TgcV8RbErP4awpIhwWb5UU-hR__4_iNbHdKqwSUPDKYGlf5eicuiYrPxH8mxivk4LRD-vyRdBZZKBt0XIDnEU4TdcNCzAXojkftqcFWYsczwS8R4JHd1qYsMyiaWl4trdHZkO4QkeLe34z4ZAaPMt3wE-gcU-VoqYTGxz-K3Le2VaZ0r3j_z6bOInsv0yngC_cD1dCXMyQJWnWjQ`
	invalidPayloadSignedToken      = `eyJhbGciOiJIUzI1NiJ9.aW52YWxpZC1wYXlsb2Fk.ScBKKm18jcaMLGYDNRUqB5gVMRZl4DM6dh3ShcxeNgY`
	invalidPartsSignedToken        = `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzdWJqZWN0IiwiaXNzIjoiaXNzdWVyIiwic2NvcGVzIjpbInMxIiwiczIiXX0`
	hmacEncryptedToken             = `eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4R0NNIn0..NZrU98U4QNO0y-u6.HSq5CvlmkUT1BPqLGZ4.1-zuiZ4RbHrTTUoA8Dvfhg`
	rsaEncryptedToken              = `eyJhbGciOiJSU0ExXzUiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0.IvkVHHiI8JwwavvTR80xGjYvkzubMrZ-TDDx8k8SNJMEylfFfNUc7F2rC3WAABF_xmJ3SW2A6on-S6EAG97k0RsjqHHNqZuaFpDvjeuLqZFfYKzI45aCtkGG4C2ij2GbeySqJ784CcvFJPUWJ-6VPN2Ho2nhefUSqig0jE2IvOKy1ywTj_VBVBxF_dyXFnXwxPKGUQr3apxrWeRJfDh2Cf8YPBlLiRznjfBfwgePB1jP7WCZNwItj10L7hsT_YWEx01XJcbxHaXFLwKyVzwWaDhreFyaWMRbGqEfqVuOT34zfmhLDhQlgLLwkXrvYqX90NsQ9Ftg0LLIfRMbsfdgug.BFy2Tj1RZN8yq2Lk-kMiZQ.9Z0eOyPiv5cEzmXh64RlAQ36Uvz0WpZgqRcc2_69zHTmUOv0Vnl1I6ks8sTraUEvukAilolNBjBj47s0b4b-Og.VM8-eJg5ZsqnTqs0LtGX_Q`
	invalidPayloadEncryptedToken   = `eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4R0NNIn0..T4jCS4Yyw1GCH0aW.y4gFaMITdBs_QZM8RKrL.6MPyk1cMVaOJFoNGlEuaRQ`
	invalidPartsEncryptedToken     = `eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4R0NNIn0..NZrU98U4QNO0y-u6.HSq5CvlmkUT1BPqLGZ4`
	signedAndEncryptedToken        = `eyJhbGciOiJSU0ExXzUiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwiY3R5IjoiSldUIn0.icnR7M1HSgMDaUnJhfzT5nLmT0eRPeNsKPkioNcyq9TZsm-LgbE7wZkNFGfQqYwvbmrZ3UpOhNkrq4n2KN3N1dtjH9TVxzfMxz2OMh0dRWUNMi58EMadhmIpH3PLyyaeDyd0dyHpOIRPFTAoOdn2GoO_flV5CvPMhgdVKYB3h3vQW-ZZDu4cOZwXAjTuThdoUZCNWFhJhXyj-PrKLyVpX6rE1o4X05IS8008SLZyx-PZlsUPyLs6CJi7Z4PzZRzOJTV00a-7UOi-fBKBZV5V8eRpWuzJ673pMALlRCBzrRin-JeEA_QnAejtMAHG7RSGP60easQN4I-0jLTQNNNynw.oFrO-5ZgRrnWmbkPsbyMiQ.BVaWUzlrdfhe0otPJpb3DGoDCT6-BOmN_Pgq5NOqVFYIAwG5pM4pf7TaiPUJeQLf0phbLgpT4RfJ20Zhwfc2MH5unCqc8TZEP2dOrYRhb8o-X57x6IQppIDbjK2i_CAWf3yF5JUB7qRqOizpKZTh3HFTVEglY3WF8tAJ8KpnatTUmwcnqlyjdBFvYu4usiyvc_u9wNbXx5-lFt0slQYleHQMUirBprKyswIBjMoFJEe7kDvU_MCKI4NI9_fSfWJpaUdNxQEvRYR1PV4ZQdwBY0X9u2n2QH5iVQMrmgmQ5hPbWxwRv1-7jXBMPBpGeFQZHeEtSwif1_Umwyt8cDyRChb3OM7XQ3eY0UJRrbmvhcLWIcMp8FpblDaBinbjD6qIVXZVmaAdIbi2a_HblfoeL3-UABb82AAxOqQcAFjDEDTR2TFalDXSwgPZrAaQ_Mql3eFe9r2y0UVkgG7XYF4ik8sSK48CkZPUvkZFr-K9QMq-RZLzT3Zw0edxNaKgje27S26H9qClh6CCr9nk38AZZ76_Xz7f-Fil5xI0Dq95UzvwW__U3JJWE6OVUVx_RVJgdOJn8_B7hluckwBLUblscA.83pPXNnH0sKgHvFboiJVDA`
	invalidSignedAndEncryptedToken = `eyJhbGciOiJSU0ExXzUiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0.QKYu3DkFEXBUa2U0Sgtm-e44BMuaFVbMu2T-GB3qEGONrmOuaB5BtNCvBUnuj6HR0v6u-tvawToRSzExQQLFTvPcLiQR8iclWirqAFUqLrg8kRU3qIRLkmErYeGIfCML1jq9ofKg0DI5-YrU5RSyUg9cwfXKEx8KNwFcjeVeDZwWEACdU8xBnQp57rNfr0Tj-dPnGKID7LU5ZV0vhK90FpEG7UqOeSHFmvONQyz6Ca-ZkE8X2swqGad-q5xl8f9pApdFqHzADox5OlgtxPkr-Khkm6WGfvf1K_e-iW5LYtvWIAjNByft2TexsNcYpdAO2oNAgh2nkhoohl-zCWU-og.UAU65JWKqvHZ_Z0V-xLyjQ.M6sQ4lAzKFelSmL6C6uoK00rB8IFCAK-eJ0iByGhtg8eYtmSBFsP_oUySfKPtxcPRkQ7YxnEX5D-DOo20wCV7il2Be9No__0R6_5heISOMXcKmKP3D6pFusaPisNGOgLw8SKXBuVpe20PvOJ9RgOXRKucSR2UMINXtqIn9RdxbKOlBBmMJhnX4TeQ00fRILng2sMbUHsWExSthQODHGx6VcwLFp-Aqmsnv2q2KkLpA8sEm48AHHFQXSGtlVGVgWKi3dOQYUnDJW4P64Xxr1Uq3yT7w_dRwK4BA7l3Biecj5dwkKrFMJ_RaCt-ED_R15zpxg6PmnXeeJnif58Fai40ZWOsGvLZNYwL1jbi-TrsargpdUQedfzuTk8Na2NkCzFNg2BYXVDHJ_WAX1daVyhvunaURwAlBatAcmnOGxWebwV1xQoQ7iHg6ZGohCannn_pqGwJlMHMgnCcnCIhwfj9uL9Ejz_TVceZNMlT1KvLRafVfxGhkp48bdnd8OcXmjT9pQzZUB3OqrstWKhbItZ1xMpy6dZ54ldWvtTTyQ4tQJaVWgXERUM1erDT6Ypyl15-fumOB9MRcgMG3NDblKowA.P9WTBITvVUgrLjX6bS0opQ`
)

type customClaims struct {
	Scopes []string `json:"scopes,omitempty"`
}

func TestDecodeToken(t *testing.T) {
	tok, err := ParseSigned(hmacSignedToken)
	if assert.NoError(t, err, "Error parsing signed token.") {
		c := &Claims{}
		c2 := &customClaims{}
		if assert.NoError(t, tok.Claims(sharedKey, c, c2)) {
			assert.Equal(t, "subject", c.Subject)
			assert.Equal(t, "issuer", c.Issuer)
			assert.Equal(t, []string{"s1", "s2"}, c2.Scopes)
		}
	}
	assert.EqualError(t, tok.Claims([]byte("invalid-secret")), "square/go-jose: error in cryptographic primitive")

	tok2, err := ParseSigned(rsaSignedToken)
	if assert.NoError(t, err, "Error parsing encrypted token.") {
		c := make(map[string]interface{})
		if assert.NoError(t, tok2.Claims(&testPrivRSAKey1.PublicKey, &c)) {
			assert.Equal(t, map[string]interface{}{
				"sub":    "subject",
				"iss":    "issuer",
				"scopes": []interface{}{"s1", "s2"},
			}, c)
		}
	}
	assert.EqualError(t, tok.Claims(&testPrivRSAKey2.PublicKey), "square/go-jose: error in cryptographic primitive")

	tok3, err := ParseSigned(invalidPayloadSignedToken)
	if assert.NoError(t, err, "Error parsing signed token.") {
		assert.Error(t, tok3.Claims(sharedKey, &Claims{}), "Expected unmarshaling claims to fail.")
	}

	_, err = ParseSigned(invalidPartsSignedToken)
	assert.EqualError(t, err, "square/go-jose: compact JWS format must have three parts")

	tok4, err := ParseEncrypted(hmacEncryptedToken)
	if assert.NoError(t, err, "Error parsing encrypted token.") {
		c := Claims{}
		if assert.NoError(t, tok4.Claims(sharedEncryptionKey, &c)) {
			assert.Equal(t, "foo", c.Subject)
		}
	}
	assert.EqualError(t, tok4.Claims([]byte("invalid-secret-key")), "square/go-jose: error in cryptographic primitive")

	tok5, err := ParseEncrypted(rsaEncryptedToken)
	if assert.NoError(t, err, "Error parsing encrypted token.") {
		c := make(map[string]interface{})
		if assert.NoError(t, tok5.Claims(testPrivRSAKey1, &c)) {
			assert.Equal(t, map[string]interface{}{
				"sub":    "subject",
				"iss":    "issuer",
				"scopes": []interface{}{"s1", "s2"},
			}, c)
		}
	}
	assert.EqualError(t, tok5.Claims(testPrivRSAKey2), "square/go-jose: error in cryptographic primitive")

	tok6, err := ParseEncrypted(invalidPayloadEncryptedToken)
	if assert.NoError(t, err, "Error parsing encrypted token.") {
		assert.Error(t, tok6.Claims(sharedEncryptionKey, &Claims{}))
	}

	_, err = ParseEncrypted(invalidPartsEncryptedToken)
	assert.EqualError(t, err, "square/go-jose: compact JWE format must have five parts")

	tok7, err := ParseSignedAndEncrypted(signedAndEncryptedToken)
	if assert.NoError(t, err, "Error parsing signed-then-encrypted token.") {
		c := make(map[string]interface{})
		if nested, err := tok7.Decrypt(testPrivRSAKey1); assert.NoError(t, err) {
			assert.NoError(t, nested.Claims(testPrivRSAKey1.Public(), &c))
			assert.Equal(t, map[string]interface{}{
				"sub":    "subject",
				"iss":    "issuer",
				"scopes": []interface{}{"s1", "s2"},
			}, c)
			assert.EqualError(t, nested.Claims(testPrivRSAKey2.Public()), "square/go-jose: error in cryptographic primitive")
		}
	}
	_, err = tok7.Decrypt(testPrivRSAKey2)
	assert.EqualError(t, err, "square/go-jose: error in cryptographic primitive")

	_, err = ParseSignedAndEncrypted(invalidSignedAndEncryptedToken)
	assert.EqualError(t, err, "square/go-jose/jwt: expected content type to be JWT (cty header)")
}

func BenchmarkDecodeSignedToken(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseSigned(hmacSignedToken)
	}
}

func BenchmarkDecodeEncryptedHMACToken(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseEncrypted(hmacEncryptedToken)
	}
}
