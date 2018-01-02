/*-
 * Copyright 2014 Square Inc.
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

package jose

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/x509"
	"math/big"
	"testing"
)

func TestCompactParseJWE(t *testing.T) {
	// Should parse
	msg := "eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkExMjhHQ00ifQ.dGVzdA.dGVzdA.dGVzdA.dGVzdA"
	_, err := ParseEncrypted(msg)
	if err != nil {
		t.Error("Unable to parse valid message:", err)
	}

	// Messages that should fail to parse
	failures := []string{
		// Too many parts
		"eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkExMjhHQ00ifQ.dGVzdA.dGVzdA.dGVzdA.dGVzdA.dGVzdA",
		// Not enough parts
		"eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkExMjhHQ00ifQ.dGVzdA.dGVzdA.dGVzdA",
		// Invalid encrypted key
		"eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkExMjhHQ00ifQ.//////.dGVzdA.dGVzdA.dGVzdA",
		// Invalid IV
		"eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkExMjhHQ00ifQ.dGVzdA.//////.dGVzdA.dGVzdA",
		// Invalid ciphertext
		"eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkExMjhHQ00ifQ.dGVzdA.dGVzdA.//////.dGVzdA",
		// Invalid tag
		"eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkExMjhHQ00ifQ.dGVzdA.dGVzdA.dGVzdA.//////",
		// Invalid header
		"W10.dGVzdA.dGVzdA.dGVzdA.dGVzdA",
		// Invalid header
		"######.dGVzdA.dGVzdA.dGVzdA.dGVzdA",
		// Missing alc/enc params
		"e30.dGVzdA.dGVzdA.dGVzdA.dGVzdA",
	}

	for _, msg := range failures {
		_, err = ParseEncrypted(msg)
		if err == nil {
			t.Error("Able to parse invalid message", msg)
		}
	}
}

func TestFullParseJWE(t *testing.T) {
	// Messages that should succeed to parse
	successes := []string{
		// Flattened serialization, single recipient
		"{\"protected\":\"eyJhbGciOiJYWVoiLCJlbmMiOiJYWVoifQo\",\"encrypted_key\":\"QUJD\",\"iv\":\"QUJD\",\"ciphertext\":\"QUJD\",\"tag\":\"QUJD\"}",
		// Unflattened serialization, single recipient
		"{\"protected\":\"\",\"unprotected\":{\"enc\":\"XYZ\"},\"recipients\":[{\"header\":{\"alg\":\"XYZ\"},\"encrypted_key\":\"QUJD\"}],\"iv\":\"QUJD\",\"ciphertext\":\"QUJD\",\"tag\":\"QUJD\"}",
	}

	for i := range successes {
		_, err := ParseEncrypted(successes[i])
		if err != nil {
			t.Error("Unble to parse valid message", err, successes[i])
		}
	}

	// Messages that should fail to parse
	failures := []string{
		// Empty
		"{}",
		// Invalid JSON
		"{XX",
		// Invalid protected header
		"{\"protected\":\"###\"}",
		// Invalid protected header
		"{\"protected\":\"e1gK\"}",
		// Invalid encrypted key
		"{\"protected\":\"e30\",\"encrypted_key\":\"###\"}",
		// Invalid IV
		"{\"protected\":\"e30\",\"encrypted_key\":\"QUJD\",\"iv\":\"###\"}",
		// Invalid ciphertext
		"{\"protected\":\"e30\",\"encrypted_key\":\"QUJD\",\"iv\":\"QUJD\",\"ciphertext\":\"###\"}",
		// Invalid tag
		"{\"protected\":\"e30\",\"encrypted_key\":\"QUJD\",\"iv\":\"QUJD\",\"ciphertext\":\"QUJD\",\"tag\":\"###\"}",
		// Invalid AAD
		"{\"protected\":\"e30\",\"encrypted_key\":\"QUJD\",\"iv\":\"QUJD\",\"ciphertext\":\"QUJD\",\"tag\":\"QUJD\",\"aad\":\"###\"}",
		// Missing alg/enc headers
		"{\"protected\":\"e30\",\"encrypted_key\":\"QUJD\",\"iv\":\"QUJD\",\"ciphertext\":\"QUJD\",\"tag\":\"QUJD\"}",
		// Missing enc header
		"{\"protected\":\"eyJhbGciOiJYWVoifQ\",\"encrypted_key\":\"QUJD\",\"iv\":\"QUJD\",\"ciphertext\":\"QUJD\",\"tag\":\"QUJD\"}",
		// Missing alg header
		"{\"protected\":\"eyJlbmMiOiJYWVoifQ\",\"encrypted_key\":\"QUJD\",\"iv\":\"QUJD\",\"ciphertext\":\"QUJD\",\"tag\":\"QUJD\"}",
		// Unflattened serialization, single recipient, invalid encrypted_key
		"{\"protected\":\"\",\"recipients\":[{\"header\":{\"alg\":\"XYZ\", \"enc\":\"XYZ\"},\"encrypted_key\":\"###\"}],\"iv\":\"QUJD\",\"ciphertext\":\"QUJD\",\"tag\":\"QUJD\"}",
		// Unflattened serialization, single recipient, missing alg
		"{\"protected\":\"eyJhbGciOiJYWVoifQ\",\"recipients\":[{\"encrypted_key\":\"QUJD\"}],\"iv\":\"QUJD\",\"ciphertext\":\"QUJD\",\"tag\":\"QUJD\"}",
	}

	for i := range failures {
		_, err := ParseEncrypted(failures[i])
		if err == nil {
			t.Error("Able to parse invalid message", err, failures[i])
		}
	}
}

func TestMissingInvalidHeaders(t *testing.T) {
	protected := &rawHeader{}
	protected.set(headerEncryption, A128GCM)

	obj := &JSONWebEncryption{
		protected:   protected,
		unprotected: &rawHeader{},
		recipients: []recipientInfo{
			{},
		},
	}

	_, err := obj.Decrypt(nil)
	if err != ErrUnsupportedKeyType {
		t.Error("should detect invalid key")
	}

	obj.unprotected.set(headerCritical, []string{"1", "2"})

	_, err = obj.Decrypt(nil)
	if err == nil {
		t.Error("should reject message with crit header")
	}

	obj.unprotected.set(headerCritical, nil)
	obj.protected = &rawHeader{}
	obj.protected.set(headerAlgorithm, RSA1_5)

	_, err = obj.Decrypt(rsaTestKey)
	if err == nil || err == ErrCryptoFailure {
		t.Error("should detect missing enc header")
	}
}

func TestRejectUnprotectedJWENonce(t *testing.T) {
	// No need to test compact, since that's always protected

	// Flattened JSON
	input := `{
	"header":  {
		"alg": "XYZ", "enc": "XYZ",
		"nonce": "should-cause-an-error"
	},
	"encrypted_key": "does-not-matter",
	"aad": "does-not-matter",
	"iv": "does-not-matter",
	"ciphertext": "does-not-matter",
	"tag": "does-not-matter"
	}`
	_, err := ParseEncrypted(input)
	if err == nil {
		t.Error("JWE with an unprotected nonce parsed as valid.")
	} else if err.Error() != "square/go-jose: Nonce parameter included in unprotected header" {
		t.Errorf("Improper error for unprotected nonce: %v", err)
	}

	input = `{
		"unprotected":  {
			"alg": "XYZ", "enc": "XYZ",
			"nonce": "should-cause-an-error"
		},
		"encrypted_key": "does-not-matter",
		"aad": "does-not-matter",
		"iv": "does-not-matter",
		"ciphertext": "does-not-matter",
		"tag": "does-not-matter"
	}`
	_, err = ParseEncrypted(input)
	if err == nil {
		t.Error("JWE with an unprotected nonce parsed as valid.")
	} else if err.Error() != "square/go-jose: Nonce parameter included in unprotected header" {
		t.Errorf("Improper error for unprotected nonce: %v", err)
	}

	// Full JSON
	input = `{
		"header":  { "alg": "XYZ", "enc": "XYZ" },
		"aad": "does-not-matter",
		"iv": "does-not-matter",
		"ciphertext": "does-not-matter",
		"tag": "does-not-matter",
		"recipients": [{
			"header": { "nonce": "should-cause-an-error" },
			"encrypted_key": "does-not-matter"
		}]
	}`
	_, err = ParseEncrypted(input)
	if err == nil {
		t.Error("JWS with an unprotected nonce parsed as valid.")
	} else if err.Error() != "square/go-jose: Nonce parameter included in unprotected header" {
		t.Errorf("Improper error for unprotected nonce: %v", err)
	}
}

func TestCompactSerialize(t *testing.T) {
	// Compact serialization must fail if we have unprotected headers
	obj := &JSONWebEncryption{
		unprotected: &rawHeader{},
	}
	obj.unprotected.set(headerAlgorithm, "XYZ")

	_, err := obj.CompactSerialize()
	if err == nil {
		t.Error("Object with unprotected headers can't be compact serialized")
	}
}

func TestVectorsJWE(t *testing.T) {
	plaintext := []byte("The true sign of intelligence is not knowledge but imagination.")

	publicKey := &rsa.PublicKey{
		N: fromBase64Int(`
			oahUIoWw0K0usKNuOR6H4wkf4oBUXHTxRvgb48E-BVvxkeDNjbC4he8rUW
			cJoZmds2h7M70imEVhRU5djINXtqllXI4DFqcI1DgjT9LewND8MW2Krf3S
			psk_ZkoFnilakGygTwpZ3uesH-PFABNIUYpOiN15dsQRkgr0vEhxN92i2a
			sbOenSZeyaxziK72UwxrrKoExv6kc5twXTq4h-QChLOln0_mtUZwfsRaMS
			tPs6mS6XrgxnxbWhojf663tuEQueGC-FCMfra36C9knDFGzKsNa7LZK2dj
			YgyD3JR_MB_4NUJW_TqOQtwHYbxevoJArm-L5StowjzGy-_bq6Gw`),
		E: 65537,
	}

	expectedCompact := stripWhitespace(`
		eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkEyNTZHQ00ifQ.ROQCfge4JPm_
		yACxv1C1NSXmwNbL6kvmCuyxBRGpW57DvlwByjyjsb6g8m7wtLMqKEyhFCn
		tV7sjippEePIlKln6BvVnz5ZLXHNYQgmubuNq8MC0KTwcaGJ_C0z_T8j4PZ
		a1nfpbhSe-ePYaALrf_nIsSRKu7cWsrwOSlaRPecRnYeDd_ytAxEQWYEKFi
		Pszc70fP9geZOB_09y9jq0vaOF0jGmpIAmgk71lCcUpSdrhNokTKo5y8MH8
		3NcbIvmuZ51cjXQj1f0_AwM9RW3oCh2Hu0z0C5l4BujZVsDuGgMsGZsjUhS
		RZsAQSXHCAmlJ2NlnN60U7y4SPJhKv5tKYw.48V1_ALb6US04U3b.5eym8T
		W_c8SuK0ltJ3rpYIzOeDQz7TALvtu6UG9oMo4vpzs9tX_EFShS8iB7j6jiS
		diwkIr3ajwQzaBtQD_A.XFBoMYUZodetZdvTiFvSkQ`)

	expectedFull := stripWhitespace(`
		{ "protected":"eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkEyNTZHQ00ifQ",
		"encrypted_key":
			"ROQCfge4JPm_yACxv1C1NSXmwNbL6kvmCuyxBRGpW57DvlwByjyjsb
			 6g8m7wtLMqKEyhFCntV7sjippEePIlKln6BvVnz5ZLXHNYQgmubuNq
			 8MC0KTwcaGJ_C0z_T8j4PZa1nfpbhSe-ePYaALrf_nIsSRKu7cWsrw
			 OSlaRPecRnYeDd_ytAxEQWYEKFiPszc70fP9geZOB_09y9jq0vaOF0
			 jGmpIAmgk71lCcUpSdrhNokTKo5y8MH83NcbIvmuZ51cjXQj1f0_Aw
			 M9RW3oCh2Hu0z0C5l4BujZVsDuGgMsGZsjUhSRZsAQSXHCAmlJ2Nln
			 N60U7y4SPJhKv5tKYw",
		"iv": "48V1_ALb6US04U3b",
		"ciphertext":
			"5eym8TW_c8SuK0ltJ3rpYIzOeDQz7TALvtu6UG9oMo4vpzs9tX_EFS
			 hS8iB7j6jiSdiwkIr3ajwQzaBtQD_A",
		"tag":"XFBoMYUZodetZdvTiFvSkQ" }`)

	// Mock random reader
	randReader = bytes.NewReader([]byte{
		// Encryption key
		177, 161, 244, 128, 84, 143, 225, 115, 63, 180, 3, 255, 107, 154,
		212, 246, 138, 7, 110, 91, 112, 46, 34, 105, 47, 130, 203, 46, 122,
		234, 64, 252,
		// Randomness for RSA-OAEP
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// Initialization vector
		227, 197, 117, 252, 2, 219, 233, 68, 180, 225, 77, 219})
	defer resetRandReader()

	// Encrypt with a dummy key
	encrypter, err := NewEncrypter(A256GCM, Recipient{Algorithm: RSA_OAEP, Key: publicKey}, nil)
	if err != nil {
		panic(err)
	}

	object, err := encrypter.Encrypt(plaintext)
	if err != nil {
		panic(err)
	}

	serialized, err := object.CompactSerialize()
	if serialized != expectedCompact {
		t.Error("Compact serialization is not what we expected", serialized, expectedCompact)
	}

	serialized = object.FullSerialize()
	if serialized != expectedFull {
		t.Error("Full serialization is not what we expected")
	}
}

func TestVectorsJWECorrupt(t *testing.T) {
	priv := &rsa.PrivateKey{
		PublicKey: rsa.PublicKey{
			N: fromHexInt(`
				a8b3b284af8eb50b387034a860f146c4919f318763cd6c5598c8
				ae4811a1e0abc4c7e0b082d693a5e7fced675cf4668512772c0c
				bc64a742c6c630f533c8cc72f62ae833c40bf25842e984bb78bd
				bf97c0107d55bdb662f5c4e0fab9845cb5148ef7392dd3aaff93
				ae1e6b667bb3d4247616d4f5ba10d4cfd226de88d39f16fb`),
			E: 65537,
		},
		D: fromHexInt(`
				53339cfdb79fc8466a655c7316aca85c55fd8f6dd898fdaf1195
				17ef4f52e8fd8e258df93fee180fa0e4ab29693cd83b152a553d
				4ac4d1812b8b9fa5af0e7f55fe7304df41570926f3311f15c4d6
				5a732c483116ee3d3d2d0af3549ad9bf7cbfb78ad884f84d5beb
				04724dc7369b31def37d0cf539e9cfcdd3de653729ead5d1`),
		Primes: []*big.Int{
			fromHexInt(`
				d32737e7267ffe1341b2d5c0d150a81b586fb3132bed2f8d5262
				864a9cb9f30af38be448598d413a172efb802c21acf1c11c520c
				2f26a471dcad212eac7ca39d`),
			fromHexInt(`
				cc8853d1d54da630fac004f471f281c7b8982d8224a490edbeb3
				3d3e3d5cc93c4765703d1dd791642f1f116a0dd852be2419b2af
				72bfe9a030e860b0288b5d77`),
		},
	}

	corruptCiphertext := stripWhitespace(`
		eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkExMjhHQ00ifQ.NFl09dehy
		IR2Oh5iSsvEa82Ps7DLjRHeo0RnuTuSR45OsaIP6U8yu7vLlWaZKSZMy
		B2qRBSujf-5XIRoNhtyIyjk81eJRXGa_Bxaor1XBCMyyhGchW2H2P71f
		PhDO6ufSC7kV4bNqgHR-4ziS7KXwzN83_5kogXqxUpymUoJDNc.tk-GT
		W_VVhiTIKFF.D_BE6ImZUl9F.52a-zFnRb3YQwIC7UrhVyQ`)

	corruptAuthtag := stripWhitespace(`
		eyJhbGciOiJSU0EtT0FFUCIsImVuYyI6IkExMjhHQ00ifQ.NFl09dehy
		IR2Oh5iSsvEa82Ps7DLjRHeo0RnuTuSR45OsaIP6U8yu7vLlWaZKSZMy
		B2qRBSujf-5XIRoNhtyIyjk81eJRXGa_Bxaor1XBCMyyhGchW2H2P71f
		PhDO6ufSC7kV4bNqgHR-4ziS7KNwzN83_5kogXqxUpymUoJDNc.tk-GT
		W_VVhiTIKFF.D_BE6ImZUl9F.52a-zFnRb3YQwiC7UrhVyQ`)

	msg, _ := ParseEncrypted(corruptCiphertext)
	_, err := msg.Decrypt(priv)
	if err != ErrCryptoFailure {
		t.Error("should detect corrupt ciphertext")
	}

	msg, _ = ParseEncrypted(corruptAuthtag)
	_, err = msg.Decrypt(priv)
	if err != ErrCryptoFailure {
		t.Error("should detect corrupt auth tag")
	}
}

// Test vectors generated with nimbus-jose-jwt
func TestSampleNimbusJWEMessagesRSA(t *testing.T) {
	rsaPrivateKey, err := x509.ParsePKCS8PrivateKey(fromBase64Bytes(`
		MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCNRCEmf5PlbXKuT4uwnb
		wGKvFrtpi+bDYxOZxxqxdVkZM/bYATAnD1fg9pNvLMKeF+MWJ9kPIMmDgOh9RdnRdLvQGb
		BzhLmxwhhcua2QYiHEZizXmiaXvNP12bzEBhebdX7ObW8izMVW0p0lqHPNzkK3K75B0Sxo
		FMVKkZ7KtBHgepBT5yPhPPcNe5lXQeTne5bo3I60DRcN9jTBgMJOXdq0I9o4y6ZmoXdNTm
		0EyLzn9/EYiHqBxtKFh791EHR7wYgyi/t+nOKr4sO74NbEByP0mHDil+mPvZSzFW4l7fPx
		OclRZvpRIKIub2TroZA9s2WsshGf79eqqXYbBB9NNRAgMBAAECggEAIExbZ/nzTplfhwsY
		3SCzRJW87OuqsJ79JPQPGM4NX7sQ94eJqM7+FKLl0yCFErjgnYGdCyiArvB+oJPdsimgke
		h83X0hGeg03lVA3/6OsG3WifCAxulnLN44AM8KST8S9D9t5+cm5vEBLHazzAfWWTS13s+g
		9hH8rf8NSqgZ36EutjKlvLdHx1mWcKX7SREFVHT8FWPAbdhTLEHUjoWHrfSektnczaSHnt
		q8fFJy6Ld13QkF1ZJRUhtA24XrD+qLTc+M36IuedjeZaLHFB+KyhYR3YvXEtrbCug7dCRd
		uG6uTlDCSaSy7xHeTPolWtWo9F202jal54otxiAJFGUHgQKBgQDRAT0s6YQZUfwE0wluXV
		k0JdhDdCo8sC1aMmKlRKWUkBAqrDl7BI3MF56VOr4ybr90buuscshFf9TtrtBOjHSGcfDI
		tSKfhhkW5ewQKB0YqyHzoD6UKT0/XAshFY3esc3uCxuJ/6vOiXV0og9o7eFvr51O0TfDFh
		mcTvW4wirKlQKBgQCtB7UAu8I9Nn8czkd6oXLDRyTWYviuiqFmxR+PM9klgZtsumkeSxO1
		lkfFoj9+G8nFaqYEBA9sPeNtJVTSROCvj/iQtoqpV2NiI/wWeVszpBwsswx2mlks4LJa8a
		Yz9xrsfNoroKYVppefc/MCoSx4M+99RSm3FSpLGZQHAUGyzQKBgQDMQmq4JuuMF1y2lk0E
		SESyuz21BqV0tDVOjilsHT+5hmXWXoS6nkO6L2czrrpM7YE82F6JJZBmo7zEIXHBInGLJ3
		XLoYLZ5qNEhqYDUEDHaBCBWZ1vDTKnZlwWFEuXVavNNZvPbUhKTHq25t8qjDki/r09Vykp
		BsM2yNBKpbBOVQKBgCJyUVd3CaFUExQyAMrqD0XPCQdhJq7gzGcAQVsp8EXmOoH3zmuIeM
		ECzQEMXuWFNLMHm0tbX5Kl83vMHcnKioyI9ewhWxOBYTitf0ceG8j5F97SOl32NmCXzwoJ
		55Oa0xJXfLuIvOe8hZzp4WwZmBfKBxiCR166aPQQgIawelrVAoGAEJsHomfCI4epxH4oMw
		qYJMCGy95zloB+2+c86BZCOJAGwnfzbtc2eutWZw61/9sSO8sQCfzA8oX+5HwAgnFVzwW4
		lNMZohppYcpwN9EyjkPaCXuALC7p5rF2o63wY7JLvnjS2aYZliknh2yW6X6fSB0PK0Cpvd
		lAIyRw6Kud0zI=`))
	if err != nil {
		panic(err)
	}

	rsaSampleMessages := []string{
		"eyJlbmMiOiJBMTI4R0NNIiwiYWxnIjoiUlNBMV81In0.EW0KOhHeoAxTBnLjYhh2T6HjwI-srNs6RpcSdZvE-GJ5iww3EYWBCmeGGj1UVz6OcBfwW3wllZ6GPOHU-hxVQH5KYpVOjkmrFIYU6-8BHhxBP_PjSJEBCZzjOgsCm9Th4-zmlO7UWTdK_UtwE7nk4X-kkmEy-aZBCShA8nFe2MVvqD5F7nvEWNFBOHh8ae_juo-kvycoIzvxLV9g1B0Zn8K9FAlu8YF1KiL5NFekn76f3jvAwlExuRbFPUx4gJN6CeBDK_D57ABsY2aBVDSiQceuYZxvCIAajqSS6dMT382FNJzAiQhToOpo_1w5FnnBjzJLLEKDk_I-Eo2YCWxxsQ.5mCMuxJqLRuPXGAr.Ghe4INeBhP3MDWGvyNko7qanKdZIzKjfeiU.ja3UlVWJXKNFJ-rZsJWycw",
		"eyJlbmMiOiJBMTkyR0NNIiwiYWxnIjoiUlNBMV81In0.JsJeYoP0St1bRYNUaAmA34DAA27usE7RNuC2grGikBRmh1xrwUOpnEIXXpwr7fjVmNi52zzWkNHC8JkkRTrLcCh2VXvnOnarpH8DCr9qM6440bSrahzbxIvDds8z8q0wT1W4kjVnq1mGwGxg8RQNBWTV6Sp2FLQkZyjzt_aXsgYzr3zEmLZxB-d41lBS81Mguk_hdFJIg_WO4ao54lozvxkCn_uMiIZ8eLb8qHy0h-N21tiHGCaiC2vV8KXomwoqbJ0SXrEH4r9_R2J844H80TBZdbvNBd8whvoQNHvOX659LNs9EQ9xxvHU2kqGZekXBu7sDXXTjctMkMITobGSzw.1v5govaDvanP3LGp.llwYNBDrD7MwVLaFHesljlratfmndWs4XPQ.ZGT1zk9_yIKi2GzW6CuAyA",
		"eyJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiUlNBMV81In0.fBv3fA3TMS3ML8vlsCuvwdsKvB0ym8R30jJrlOiqkWKk7WVUkjDInFzr1zw3Owla6c5BqOJNoACXt4IWbkLbkoWV3tweXlWwpafuaWPkjLOUH_K31rS2fCX5x-MTj8_hScquVQXpbz3vk2EfulRmGXZc_8JU2NqQCAsYy3a28houqP3rDe5jEAvZS2SOFvJkKW--f5S-z39t1D7fNz1N8Btd9SmXWQzjbul5YNxI9ctqxhJpkKYpxOLlvrzdA6YdJjOlDx3n6S-HnSZGM6kQd_xKtAf8l1EGwhQmhbXhMhjVxMvGwE5BX7PAb8Ccde5bzOCJx-PVbVetuLb169ZYqQ._jiZbOPRR82FEWMZ.88j68LI-K2KT6FMBEdlz6amG5nvaJU8a-90.EnEbUTJsWNqJYKzfO0x4Yw",
		"eyJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwiYWxnIjoiUlNBMV81In0.bN6FN0qmGxhkESiVukrCaDVG3woL0xE-0bHN_Mu0WZXTQWbzzT-7jOvaN1xhGK8nzi8qpCSRgE5onONNB9i8OnJm3MMIxF7bUUEAXO9SUAFn2v--wNc4drPc5OjIu0RiJrDVDkkGjNrBDIuBaEQcke7A0v91PH58dXE7o4TLPzC8UJmRtXWhUSwjXVF3-UmYRMht2rjHJlvRbtm6Tu2LMBIopRL0zj6tlPP4Dm7I7sz9OEB3VahYAhpXnFR7D_f8RjLSXQmBvB1FiI5l_vMz2NFt2hYUmQF3EJMLIEdHvvPp3iHDGiXC1obJrDID_CCf3qs9UY7DMYL622KLvP2NIg.qb72oxECzxd_aNuHVR0aNg.Gwet9Ms8hB8rKEb0h4RGdFNRq97Qs2LQaJM0HWrCqoI.03ljVThOFvgXzMmQJ79VjQ",
		"eyJlbmMiOiJBMTkyQ0JDLUhTMzg0IiwiYWxnIjoiUlNBMV81In0.ZbEOP6rqdiIP4g7Nl1PL5gwhgDwv9RinyiUQxZXPOmD7kwEZrZ093dJnhqI9kEd3QGFlHDpB7HgNz53d27z2zmEj1-27v6miizq6tH4sN2MoeZLwSyk16O1_n3bVdDmROawsTYYFJfHsuLwyVJxPd37duIYnbUCFO9J8lLIv-2VI50KJ1t47YfE4P-Wt9jVzxP2CVUQaJwTlcwfiDLJTagYmfyrDjf525WlQFlgfJGqsJKp8BX9gmKvAo-1iCBAM8VpEjS0u0_hW9VSye36yh8BthVV-VJkhJ-0tMpto3bbBmj7M25Xf4gbTrrVU7Nz6wb18YZuhHZWmj2Y2nHV6Jg.AjnS44blTrIIfFlqVw0_Mg.muCRgaEXNKKpW8rMfW7jf7Zpn3VwSYDz-JTRg16jZxY.qjc9OGlMaaWKDWQSIwVpR4K556Pp6SF9",
		"eyJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiUlNBMV81In0.c7_F1lMlRHQQE3WbKmtHBYTosdZrG9hPfs-F9gNQYet61zKG8NXVkSy0Zf2UFHt0vhcO8hP2qrqOFsy7vmRj20xnGHQ2EE29HH6hwX5bx1Jj3uE5WT9Gvh0OewpvF9VubbwWTIObBpdEG7XdJsMAQlIxtXUmQYAtLTWcy2ZJipyJtVlWQLaPuE8BKfZH-XAsp2CpQNiRPI8Ftza3EAspiyRfVQbjKt7nF8nuZ2sESjt7Y50q4CSiiCuGT28T3diMN0_rWrH-I-xx7OQvJlrQaNGglGtu3jKUcrJDcvxW2e1OxriaTeuQ848ayuRvGUNeSv6WoVYmkiK1x_gNwUAAbw.7XtSqHJA7kjt6JrfxJMwiA.Yvi4qukAbdT-k-Fd2s4G8xzL4VFxaFC0ZIzgFDAI6n0.JSWPJ-HjOE3SK9Lm0yHclmjS7Z1ahtQga9FHGCWVRcc",
		"eyJlbmMiOiJBMTI4R0NNIiwiYWxnIjoiUlNBLU9BRVAifQ.SYVxJbCnJ_tcR13LJpaqHQj-nGNkMxre4A1FmnUdxnvzeJwuvyrLiUdRsZR1IkP4fqLtDON2mumx39QeJQf0WIObPBYlIxycRLkwxDHRVlyTmPvdZHAxN26jPrk09wa5SgK1UF1W1VSQIPm-Tek8jNAmarF1Yxzxl-t54wZFlQiHP4TuaczugO5f-J4nlWenfla2mU1snDgdUMlEZGOAQ_gTEtwSgd1MqXmK_7LZBkoDqqoCujMZhziafJPXPDaUUqBLW3hHkkDA7GpVec3XcTtNUWQJqOpMyQhqo1KQMc8jg3fuirILp-hjvvNVtBnCRBvbrKUCPzu2_yH3HM_agA.2VsdijtonAxShNIW.QzzB3P9CxYP3foNKN0Ma1Z9tMwijAlkWo08.ZdQkIPDY_M-hxqi5fD4NGw",
		"eyJlbmMiOiJBMTkyR0NNIiwiYWxnIjoiUlNBLU9BRVAifQ.Z2oTJXXib1u-S38Vn3DRKE3JnhnwgUa92UhsefzY2Wpdn0dmxMfYt9iRoJGFfSAcA97MOfjyvXVRCKWXGrG5AZCMAXEqU8SNQwKPRjlcqojcVzQyMucXI0ikLC4mUgeRlfKTwsBicq6JZZylzRoLGGSNJQbni3_BLsf7H3Qor0BYg0FPCLG9Z2OVvrFzvjTLmZtV6gFlVrMHBxJub_aUet9gAkxiu1Wx_Kx46TlLX2tkumXIpTGlzX6pef6jLeZ5EIg_K-Uz4tkWgWQIEkLD7qmTyk5pAGmzukHa_08jIh5-U-Sd8XGZdx4J1pVPJ5CPg0qDJGZ_cfgkgpWbP_wB6A.4qgKfokK1EwYxz20._Md82bv_KH2Vru0Ue2Eb6oAqHP2xBBP5jF8.WFRojvQpD5VmZlOr_dN0rQ",
		"eyJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiUlNBLU9BRVAifQ.JzCUgJcBJmBgByp4PBAABUfhezPvndxBIVzaoZ96DAS0HPni0OjMbsOGsz6JwNsiTr1gSn_S6R1WpZM8GJc9R2z0EKKVP67TR62ZSG0MEWyLpHmG_4ug0fAp1HWWMa9bT4ApSaOLgwlpVAb_-BPZZgIu6c8cREuMon6UBHDqW1euTBbzk8zix3-FTZ6p5b_3soDL1wXfRiRBEsxxUGMnpryx1OFb8Od0JdyGF0GgfLt6OoaujDJpo-XtLRawu1Xlg6GqRs0NQwSHZ5jXgQ6-zgCufXonAmYTiIyBXY2no9XmECTexjwrS_05nA7H-UyIZEBOCp3Yhz2zxrt5j_0pvQ.SJR-ghhaUKP4zXtZ.muiuzLfZA0y0BDNsroGTw2r2-l73SLf9lK8.XFMH1oHr1G6ByP3dWSUUPA",
		"eyJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwiYWxnIjoiUlNBLU9BRVAifQ.U946MVfIm4Dpk_86HrnIA-QXyiUu0LZ67PL93CMLmEtJemMNDqmRd9fXyenCIhAC7jPIV1aaqW7gS194xyrrnUpBoJBdbegiPqOfquy493Iq_GQ8OXnFxFibPNQ6rU0l8BwIfh28ei_VIF2jqN6bhxFURCVW7fG6n6zkCCuEyc7IcxWafSHjH2FNttREuVj-jS-4LYDZsFzSKbpqoYF6mHt8H3btNEZDTSmy_6v0fV1foNtUKNfWopCp-iE4hNh4EzJfDuU8eXLhDb03aoOockrUiUCh-E0tQx9su4rOv-mDEOHHAQK7swm5etxoa7__9PC3Hg97_p4GM9gC9ykNgw.pnXwvoSPi0kMQP54of-HGg.RPJt1CMWs1nyotx1fOIfZ8760mYQ69HlyDp3XmdVsZ8.Yxw2iPVWaBROFE_FGbvodA",
		"eyJlbmMiOiJBMTkyQ0JDLUhTMzg0IiwiYWxnIjoiUlNBLU9BRVAifQ.eKEOIJUJpXmO_ghH_nGCJmoEspqKyiy3D5l0P8lKutlo8AuYHPQlgOsaFYnDkypyUVWd9zi-JaQuCeo7dzoBiS1L71nAZo-SUoN0anQBkVuyuRjr-deJMhPPfq1H86tTk-4rKzPr1Ivd2RGXMtWsrUpNGk81r1v8DdMntLE7UxZQqT34ONuZg1IXnD_U6di7k07unI29zuU1ySeUr6w1YPw5aUDErMlpZcEJWrgOEYWaS2nuC8sWGlPGYEjqkACMFGn-y40UoS_JatNZO6gHK3SKZnXD7vN5NAaMo_mFNbh50e1t_zO8DaUdLtXPOBLcx_ULoteNd9H8HyDGWqwAPw.0xmtzJfeVMoIT1Cp68QrXA.841l1aA4c3uvSYfw6l180gn5JZQjL53WQ5fr8ejtvoI.lojzeWql_3gDq-AoaIbl_aGQRH_54w_f",
		"eyJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiUlNBLU9BRVAifQ.D0QkvIXR1TL7dIHWuPNMybmmD8UPyQd1bRKjRDNbA2HmKGpamCtcJmpNB_EetNFe-LDmhe44BYI_XN2wIBbYURKgDK_WG9BH0LQw_nCVqQ-sKqjtj3yQeytXhLHYTDmiF0TO-uW-RFR7GbPAdARBfuf4zj82r_wDD9sD5WSCGx89iPfozDOYQ_OLwdL2WD99VvDyfwS3ZhxA-9IMSYv5pwqPkxj4C0JdjCqrN0YNrZn_1ORgjtsVmcWXsmusObTozUGA7n5GeVepfZdU1vrMulAwdRYqOYtlqKaOpFowe9xFN3ncBG7wb4f9pmzbS_Dgt-1_Ii_4SEB9GQ4NiuBZ0w.N4AZeCxMGUv52A0UVJsaZw.5eHOGbZdtahnp3l_PDY-YojYib4ft4SRmdsQ2kggrTs.WsmGH8ZDv4ctBFs7qsQvw2obe4dVToRcAQaZ3PYL34E",
		"eyJlbmMiOiJBMTI4R0NNIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.fDTxO_ZzZ3Jdrdw-bxvg7u-xWB2q1tp3kI5zH6JfhLUm4h6rt9qDA_wZlRym8-GzEtkUjkTtQGs6HgQx_qlyy8ylCakY5GHsNhCG4m0UNhRiNfcasAs03JSXfON9-tfTJimWD9n4k5OHHhvcrsCW1G3jYeLsK9WHCGRIhNz5ULbo8HBrCTbmZ6bOEQ9mqhdssLpdV24HDpebotf3bgPJqoaTfWU6Uy7tLmPiNuuNRLQ-iTpLyNMTVvGqqZhpcV3lAEN5l77QabI5xLJYucvYjrXQhAEZ7YXO8oRYhGkdG2XXIRcwr87rBeRH-47HAyhZgF_PBPBhhrJNS9UNMqdfBw.FvU4_s7Md6vxnXWd.fw29Q4_gHt4f026DPPV-CNebQ8plJ6IVLX8._apBZrw7WsT8HOmxgCrTwA",
		"eyJlbmMiOiJBMTkyR0NNIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.bYuorK-rHMbO4c2CRWtvyOEaM1EN-o-wLRZ0wFWRX9mCXQ-iTNarZn7ksYM1XnGmZ4u3CSowX1Hpca9Rg72_VJCmKapqCT7r3YfasN4_oeLwuSKI_gT-uVOznod97tn3Gf_EDv0y1V4H0k9BEIFGbajAcG1znTD_ODY3j2KZJxisfrsBoslc6N-HI0kKZMC2hSGuHOcOf8HN1sTE-BLqZCtoj-zxQECJK8Wh14Ih4jzzdmmiu_qmSR780K6su-4PRt3j8uY7oCiLBfwpCsCmhJgp8rKd91zoedZmamfvX38mJIfE52j4fG6HmIYw9Ov814fk9OffV6tzixjcg54Q2g.yeVJz4aSh2s-GUr9.TBzzWP5llEiDdugpP2SmPf2U4MEGG9EoPWk.g25UoWpsBaOd45J__FX7mA",
		"eyJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.h9tFtmh762JuffBxlSQbJujCyI4Zs9yc3IOb1yR8g65W4ZHosIvzVGHWbShj4EY9MNrz-RbKtHfqQGGzDeo3Xb4-HcQ2ZDHyWoUg7VfA8JafJ5zIKL1npz8eUExOVMLsAaRfHg8qNfczodg3egoSmX5Q-nrx4DeidDSXYZaZjV0C72stLTPcuQ7XPV7z1tvERAkqpvcsRmJn_PiRNxIbAgoyHMJ4Gijuzt1bWZwezlxYmw0TEuwCTVC2fl9NJTZyxOntS1Lcm-WQGlPkVYeVgYTOQXLlp7tF9t-aAvYpth2oWGT6Y-hbPrjx_19WaKD0XyWCR46V32DlXEVDP3Xl2A.NUgfnzQyEaJjzt9r.k2To43B2YVWMeR-w3n4Pr2b5wYq2o87giHk.X8_QYCg0IGnn1pJqe8p_KA",
		"eyJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.EDq6cNP6Yp1sds5HZ4CkXYp7bs9plIYVZScKvuyxUy0H1VyBC_YWg0HvndPNb-vwh1LA6KMxRazlOwJ9iPR9YzHnYmGgPM3Je_ZzBfiPlRfq6hQBpGnNaypBI1XZ2tyFBhulsVLqyJe2SmM2Ud00kasOdMYgcN8FNFzq7IOE7E0FUQkIwLdUL1nrzepiYDp-5bGkxWRcL02cYfdqdm00G4m0GkUxAmdxa3oPNxZlt2NeBI_UVWQSgJE-DJVJQkDcyA0id27TV2RCDnmujYauNT_wYlyb0bFDx3pYzzNXfAXd4wHZxt75QaLZ5APJ0EVfiXJ0qki6kT-GRVmOimUbQA.vTULZL7LvS0WD8kR8ZUtLg.mb2f0StEmmkuuvsyz8UplMvF58FtZzlu8eEwzvPUvN0.hbhveEN40V-pgG2hSVgyKg",
		"eyJlbmMiOiJBMTkyQ0JDLUhTMzg0IiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.DuYk92p7u-YIN-JKn-XThmlVcnhU9x5TieQ2uhsLQVNlo0iWC9JJPP6bT6aI6u_1BIS3yE8_tSGGL7eM-zyEk6LuTqSWFRaZcZC06d0MnS9eYZcw1T2D17fL-ki-NtCaTahJD7jE2s0HevRVW49YtL-_V8whnO_EyVjvXIAQlPYqhH_o-0Nzcpng9ggdAnuF2rY1_6iRPYFJ3BLQvG1oWhyJ9s6SBttlOa0i6mmFCVLHx6sRpdGAB3lbCL3wfmHq4tpIv77gfoYUNP0SNff-zNmBXF_wp3dCntLZFTjbfMpGyHlruF_uoaLqwdjYpUGNUFVUoeSiMnSbMKm9NxiDgQ.6Mdgcqz7bMU1UeoAwFC8pg.W36QWOlBaJezakUX5FMZzbAgeAu_R14AYKZCQmuhguw.5OeyIJ03olxmJft8uBmjuOFQPWNZMYLI",
		"eyJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.ECulJArWFsPL2FlpCN0W8E7IseSjJg1cZqE3wz5jk9gvwgNForAUEv5KYZqhNI-p5IxkGV0f8K6Y2X8pWzbLwiPIjZe8_dVqHYJoINxqCSgWLBhz0V36qL9Nc_xARTBk4-ZteIu75NoXVeos9gNvFnkOCj4tm-jGo8z8EFO9XfODgjhiR4xv8VqUtvrkjo9GQConaga5zpV-J4JQlXbdqbDjnuwacnJAxYpFyuemqcgqsl6BnFX3tovGkmSUPqcvF1A6tiHqr-TEmcgVqo5C3xswknRBKTQRM00iAmJ92WlVdkoOCx6E6O7cVHFawZ14BLzWzm66Crb4tv0ucYvk_Q.mxolwUaoj5S5kHCfph0w8g.nFpgYdnYg3blHCCEi2XXQGkkKQBXs2OkZaH11m3PRvk.k8BAVT4EcyrUFVIKr-KOSPbF89xyL0Vri2rFTu2iIWM",
	}

	for _, msg := range rsaSampleMessages {
		obj, err := ParseEncrypted(msg)
		if err != nil {
			t.Error("unable to parse message", msg, err)
			continue
		}
		plaintext, err := obj.Decrypt(rsaPrivateKey)
		if err != nil {
			t.Error("unable to decrypt message", msg, err)
			continue
		}
		if string(plaintext) != "Lorem ipsum dolor sit amet" {
			t.Error("plaintext is not what we expected for msg", msg)
		}
	}
}

// Test vectors generated with nimbus-jose-jwt
func TestSampleNimbusJWEMessagesAESKW(t *testing.T) {
	aesTestKeys := [][]byte{
		fromHexBytes("DF1FA4F36FFA7FC42C81D4B3C033928D"),
		fromHexBytes("DF1FA4F36FFA7FC42C81D4B3C033928D95EC9CDC2D82233C"),
		fromHexBytes("DF1FA4F36FFA7FC42C81D4B3C033928D95EC9CDC2D82233C333C35BA29044E90"),
	}

	aesSampleMessages := [][]string{
		{
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4R0NNIiwidGFnIjoib2ZMd2Q5NGloVWFRckJ0T1pQUDdjUSIsImFsZyI6IkExMjhHQ01LVyIsIml2IjoiV2Z3TnN5cjEwWUFjY2p2diJ9.9x3RxdqIS6P9xjh93Eu1bQ.6fs3_fSGt2jull_5.YDlzr6sWACkFg_GU5MEc-ZEWxNLwI_JMKe_jFA.f-pq-V7rlSSg_q2e1gDygw",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyR0NNIiwidGFnIjoic2RneXB1ckFjTEFzTmZJU0lkZUNpUSIsImFsZyI6IkExMjhHQ01LVyIsIml2IjoieVFMR0dCdDJFZ0c1THdyViJ9.arslKo4aKlh6f4s0z1_-U-8JbmhAoZHN.Xw2Q-GX98YXwuc4i.halTEWMWAYZbv-qOD52G6bte4x6sxlh1_VpGEA.Z1spn016v58cW6Q2o0Qxag",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2R0NNIiwidGFnIjoicTNzejF5VUlhbVBDYXJfZ05kSVJqQSIsImFsZyI6IkExMjhHQ01LVyIsIml2IjoiM0ZRM0FsLWJWdWhmcEIyQyJ9.dhVipWbzIdsINttuZM4hnjpHvwEHf0VsVrOp4GAg01g.dk7dUyt1Qj13Pipw.5Tt70ONATF0BZAS8dBkYmCV7AQUrfb8qmKNLmw.A6ton9MQjZg0b3C0QcW-hg",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidGFnIjoiUHNpTGphZnJZNE16UlRmNlBPLTZfdyIsImFsZyI6IkExMjhHQ01LVyIsIml2IjoiSUFPbnd2ODR5YXFEaUxtbSJ9.swf92_LyCvjsvkynHTuMNXRl_MX2keU-fMDWIMezHG4.LOp9SVIXzs4yTnOtMyXZYQ.HUlXrzqJ1qXYl3vUA-ydezCg77WvJNtKdmZ3FPABoZw.8UYl1LOofQLAxHHvWqoTbg",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyQ0JDLUhTMzg0IiwidGFnIjoiWGRndHQ5dUVEMVlVeU1rVHl6M3lqZyIsImFsZyI6IkExMjhHQ01LVyIsIml2IjoiWF90V2RhSmh6X3J1SHJvQSJ9.JQ3dS1JSgzIFi5M9ig63FoFU1nHBTmPwXY_ovNE2m1JOSUvHtalmihIuraPDloCf.e920JVryUIWt7zJJQM-www.8DUrl4LmsxIEhRr9RLTHG9tBTOcwXqEbQHAJd_qMHzE.wHinoqGUhL4O7lx125kponpwNtlp8VGJ",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwidGFnIjoicGgyaTdoY0FWNlh3ZkQta1RHYlVXdyIsImFsZyI6IkExMjhHQ01LVyIsIml2IjoiaG41Smk4Wm1rUmRrSUxWVSJ9._bQlJXl22dhsBgYPhkxUyinBNi871teGWbviOueWj2PqG9OPxIc9SDS8a27YLSVDMircd5Q1Df28--vcXIABQA.DssmhrAg6w_f2VDaPpxTbQ.OGclEmqrxwvZqAfn7EgXlIfXgr0wiGvEbZz3zADnqJs.YZeP0uKVEiDl8VyC-s20YN-RbdyGNsbdtoGDP3eMof8",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4R0NNIiwiYWxnIjoiQTEyOEtXIn0.TEMcXEoY8WyqGjYs5GZgS-M_Niwu6wDY.i-26KtTt51Td6Iwd.wvhkagvPsLj3QxhPBbfH_th8OqxisUtme2UadQ.vlfvBPv3bw2Zk2H60JVNLQ",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyR0NNIiwiYWxnIjoiQTEyOEtXIn0.gPaR6mgQ9TUx05V6DRfgTQeZxl0ZSzBa5uQd-qw6yLs.MojplOD77FkMooS-.2yuD7dKR_C3sFbhgwiBccKKOF8DrSvNiwX7wPQ.qDKUbSvMnJv0qifjpWC14g",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiQTEyOEtXIn0.Fg-dgSkUW1KEaL5YDPoWHNL8fpX1WxWVLA9OOWsjIFhQVDKyUZI7BQ.mjRBpyJTZf7H-quf.YlNHezMadtaSKp23G-ozmYhHOeHwuJnvWGTtGg.YagnR7awBItUlMDo4uklvg",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwiYWxnIjoiQTEyOEtXIn0.x1vYzUE-E2XBWva9OPuwtqfQaf9rlJCIBAyAe6N2q2kWfJrkxGxFsQ.gAwe78dyODFaoP2IOityAA.Yh5YfovkWxGBNAs1sVhvXow_2izHHsBiYEc9JYD6kVg.mio1p3ncp2wLEaEaRa7P0w",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyQ0JDLUhTMzg0IiwiYWxnIjoiQTEyOEtXIn0.szGrdnmF7D5put2aRBvSSFfp0vRgkRGYaafijJIqAF6PWd1IxsysZRV8aQkQOW1cB6d0fXsTfYM.Ru25LVOOk4xhaK-cIZ0ThA.pF9Ok5zot7elVqXFW5YYHV8MuF9gVGzpQnG1XDs_g_w.-7la0uwcNPpteev185pMHZjbVDXlrec8",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiQTEyOEtXIn0.cz-hRv0xR5CnOcnoRWNK8Q9poyVYzRCVTjfmEXQN6xPOZUkJ3zKNqb8Pir_FS0o2TVvxmIbuxeISeATTR2Ttx_YGCNgMkc93.SF5rEQT94lZR-UORcMKqGw.xphygoU7zE0ZggOczXCi_ytt-Evln8CL-7WLDlWcUHg.5h99r8xCCwP2PgDbZqzCJ13oFfB2vZWetD5qZjmmVho",
		},
		{
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4R0NNIiwidGFnIjoiVWR5WUVKdEJ5ZTA5dzdjclY0cXI1QSIsImFsZyI6IkExOTJHQ01LVyIsIml2IjoiZlBBV0QwUmdSbHlFdktQcCJ9.P1uTfTuH-imL-NJJMpuTRA.22yqZ1NIfx3KNPgc.hORWZaTSgni1FS-JT90vJly-cU37qTn-tWSqTg.gMN0ufXF92rSXupTtBNkhA",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyR0NNIiwidGFnIjoiOU9qX3B2LTJSNW5lZl9YbWVkUWltUSIsImFsZyI6IkExOTJHQ01LVyIsIml2IjoiY3BybGEwYUYzREVQNmFJTSJ9.6NVpAm_APiC7km2v-oNR8g23K9U_kf1-.jIg-p8tNwSvwxch0.1i-GPaxS4qR6Gy4tzeVtSdRFRSKQSMpmn-VhzA.qhFWPqtA6vVPl7OM3DThsA",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2R0NNIiwidGFnIjoiOVc3THg3MVhGQVJCb3NaLVZ5dXc4ZyIsImFsZyI6IkExOTJHQ01LVyIsIml2IjoiZ1N4ZE5heFdBSVBRR0tHYiJ9.3YjPz6dVQwAtCekvtXiHZrooOUlmCsMSvyfwmGwdrOA.hA_C0IDJmGaRzsB0.W4l7OPqpFxiVOZTGfAlRktquyRTo4cEOk9KurQ.l4bGxOkO_ql_jlPo3Oz3TQ",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidGFnIjoiOHJYbWl2WXFWZjNfbHhhd2NUbHJoUSIsImFsZyI6IkExOTJHQ01LVyIsIml2IjoiVXBWeXprVTNKcjEwYXRqYyJ9.8qft-Q_xqUbo5j_aVrVNHchooeLttR4Kb6j01O8k98M.hXO-5IKBYCL9UdwBFVm0tg.EBM4lCZX_K6tfqYmfoDxVPHcf6cT--AegXTTjfSqsIw.Of8xUvEQSh3xgFT3uENnAg",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyQ0JDLUhTMzg0IiwidGFnIjoiVnItSnVaX0tqV2hSWWMzdzFwZ3cwdyIsImFsZyI6IkExOTJHQ01LVyIsIml2IjoiRGg2R3dISVBVS3ljZGNZeCJ9.YSEDjCnGWr_n9H94AvLoRnwm6bdU9w6-Q67k-QQRVcKRd6673pgH9zEF9A9Dt6o1.gcmVN4kxqBuMq6c7GrK3UQ.vWzJb0He6OY1lhYYjYS7CLh55REAAq1O7yNN-ND4R5Q.OD0B6nwyFaDr_92ysDOtlVnJaeoIqhGw",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwidGFnIjoieEtad1BGYURpQ3NqUnBqZUprZHhmZyIsImFsZyI6IkExOTJHQ01LVyIsIml2IjoieTVHRFdteXdkb2R1SDJlYyJ9.AW0gbhWqlptOQ1y9aoNVwrTIIkBfrp33C2OWJsbrDRk6lhxg_IgFhMDTE37moReySGUtttC4CXQD_7etHmd3Hw.OvKXK-aRKlXHOpJQ9ZY_YQ.Ngv7WarDDvR2uBj_DavPAR3DYuIaygvSSdcHrc8-ZqM.MJ6ElitzFCKf_0h5fIJw8uOLC6ps7dKZPozF8juQmUY",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4R0NNIiwiYWxnIjoiQTE5MktXIn0.8qu63pppcSvp1vv37WrZ44qcCTg7dQMA.cDp-f8dJTrDEpZW4.H6OBJYs4UvFR_IZHLYQZxB6u9a0wOdAif2LNfQ.1dB-id0UIwRSlmwHx5BJCg",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyR0NNIiwiYWxnIjoiQTE5MktXIn0._FdoKQvC8qUs7K0upriEihUwztK8gOwonXpOxdIwrfs.UO38ok8gDdpLVa1T.x1GvHdVCy4fxoQRg-OQK4Ez3jDOvu9gllLPeEA.3dLeZGIprh_nHizOTVi1xw",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiQTE5MktXIn0.uzCJskgSIK6VkjJIu-dQi18biqaY0INc_A1Ehx0oESafgtR99_n4IA.W2eKK8Y14WwTowI_.J2cJC7R6Bz6maR0s1UBMPyRi5BebNUAmof4pvw.-7w6htAlc4iUsOJ6I04rFg",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwiYWxnIjoiQTE5MktXIn0.gImQeQETp_6dfJypFDPLlv7c5pCzuq86U16gzrLiCXth6X9XfxJpvQ.YlC4MxjtLWrsyEvlFhvsqw.Vlpvmg9F3gkz4e1xG01Yl2RXx-jG99rF5UvCxOBXSLc.RZUrU_FoR5bG3M-j3GY0Dw",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyQ0JDLUhTMzg0IiwiYWxnIjoiQTE5MktXIn0.T2EfQ6Tu2wJyRMgZzfvBYmQNCCfdMudMrg86ibEMVAOUKJPtR3WMPEb_Syy9p2VjrLKRlv7nebo.GPc8VbarPPRtzIRATB8NsA.ugPCqLvVLwh55bWlwjsFkmWzJ31z5z-wuih2oJqmG_U.m7FY3EjvV6mKosEYJ5cY7ezFoVQoJS8X",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiQTE5MktXIn0.OgLMhZ-2ZhslQyHfzOfyC-qmT6bNg9AdpP59B4jtyxWkQu3eW475WCdiAjojjeyBtVRGQ5vOomwaOIFejY_IekzH6I_taii3.U9x44MF6Wyz5TIwIzwhoxQ.vK7yvSF2beKdNxNY_7n4XdF7JluCGZoxdFJyTJVkSmI.bXRlI8KL-g7gpprQxGmXjVYjYghhWJq7mlCfWI8q2uA",
		},
		{
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4R0NNIiwidGFnIjoiR3BjX3pfbjduZjJVZlEtWGdsaTBaQSIsImFsZyI6IkEyNTZHQ01LVyIsIml2IjoiUk40eUdhOVlvYlFhUmZ1TCJ9.Q4ukD6_hZpmASAVcqWJ9Wg.Zfhny_1WNdlp4fH-.3sekDCjkExQCcv28ZW4yrcFnz0vma3vgoenSXA.g8_Ird2Y0itTCDP61du-Yg",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyR0NNIiwidGFnIjoiWC05UkNVWVh4U3NRelcwelVJS01VUSIsImFsZyI6IkEyNTZHQ01LVyIsIml2IjoiY3JNMnJfa3RrdWpyQ1h5OSJ9.c0q2jCxxV4y1h9u_Xvn7FqUDnbkmNEG4.S_noOTZKuUo9z1l6.ez0RdA25vXMUGH96iXmj3DEVox0J7TasJMnzgg.RbuSPTte_NzTtEEokbc5Ig",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2R0NNIiwidGFnIjoiWmwyaDFpUW11QWZWd2lJeVp5RHloZyIsImFsZyI6IkEyNTZHQ01LVyIsIml2Ijoib19xZmljb0N0NzNzRWo1QyJ9.NpJxRJ0aqcpekD6HU2u9e6_pL_11JXjWvjfeQnAKkZU.4c5qBcBBrMWi27Lf.NKwNIb4b6cRDJ1TwMKsPrjs7ADn6aNoBdQClVw.yNWmSSRBqQfIQObzj8zDqw",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidGFnIjoiMXdwVEI3LWhjdzZUVXhCbVh2UzdhUSIsImFsZyI6IkEyNTZHQ01LVyIsIml2IjoiOUdIVnZJaDZ0a09vX2pHUSJ9.MFgIhp9mzlq9hoPqqKVKHJ3HL79EBYtV4iNhD63yqiU.UzW5iq8ou21VpZYJgKEN8A.1gOEzA4uAPvHP76GMfs9uLloAV10mKaxiZVAeL7iQA0.i1X_2i0bCAz-soXF9bI_zw",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyQ0JDLUhTMzg0IiwidGFnIjoiNThocUtsSk15Y1BFUEFRUlNfSzlNUSIsImFsZyI6IkEyNTZHQ01LVyIsIml2IjoiUDh3aTBWMTluVnZqNXpkOSJ9.FXidOWHNFJODO74Thq3J2cC-Z2B8UZkn7SikeosU0bUK6Jx_lzzmUZ-Lafadpdpj.iLfcDbpuBKFiSfiBzUQc7Q.VZK-aD7BFspqfvbwa0wE2wwWxdomzk2IKMetFe8bI44.7wC6rJRGa4x48xbYMd6NH9VzK8uNn4Cb",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwidGFnIjoicGcwOEpUcXdzMXdEaXBaRUlpVExoQSIsImFsZyI6IkEyNTZHQ01LVyIsIml2IjoiSlpodk9CdU1RUDFFZTZTNSJ9.wqVgTPm6TcYCTkpbwmn9sW4mgJROH2A3dIdSXo5oKIQUIVbQsmy7KXH8UYO2RS9slMGtb869C8o0My67GKg9dQ.ogrRiLlqjB1S5j-7a05OwA.2Y_LyqhU4S_RXMsB74bxcBacd23J2Sp5Lblw-sOkaUY.XGMiYoU-f3GaEzSvG41vpJP2DMGbeDFoWmkUGLUjc4M",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4R0NNIiwiYWxnIjoiQTI1NktXIn0.QiIZm9NYfahqYFIbiaoUhCCHjotHMkup.EsU0XLn4FjzzCILn.WuCoQkm9vzo95E7hxBtfYpt-Mooc_vmSTyzj6Q.NbeeYVy6gQPlmhoWDrZwaQ",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyR0NNIiwiYWxnIjoiQTI1NktXIn0.1ol3j_Lt0Os3UMe2Gypj0o8b77k0FSmqD7kNRNoMa9U.vZ2HMTgN2dgUd42h.JvNcy8-c8sYzOC089VtFSg2BOQx3YF8CqSTuJw.t03LRioWWKN3d7SjinU6SQ",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiQTI1NktXIn0.gbkk03l1gyrE9qGEMVtORiyyUqKsgzbqjLd8lw0RQ07WWn--TV4BgA.J8ThH4ac2UhSsMIP.g-W1piEGrdi3tNwQDJXpYm3fQjTf82mtVCrCOg.-vY05P4kiB9FgF2vwrSeXQ",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwiYWxnIjoiQTI1NktXIn0.k86pQs7gmQIzuIWRFwesF32XY2xi1WbYxi7XUf_CYlOlehwGCTINHg.3NcC9VzfQgsECISKf4xy-g.v2amdo-rgeGsg-II_tvPukX9D-KAP27xxf2uQJ277Ws.E4LIE3fte3glAnPpnd8D9Q",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMTkyQ0JDLUhTMzg0IiwiYWxnIjoiQTI1NktXIn0.b8iN0Am3fCUvj7sBd7Z0lpfzBjh1MOgojV7J5rDfrcTU3b35RGYgEV1RdcrtUTBgUwITDjmU7jM.wsSDBFghDga_ERv36I2AOg.6uJsucCb2YReFOJGBdo4zidTIKLUmZBIXfm_M0AJpKk.YwdAfXI3HHcw2wLSnfCRtw4huZQtSKhz",
			"eyJ6aXAiOiJERUYiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiQTI1NktXIn0.akY9pHCbkHPh5VpXIrX0At41XnJIKBR9iMMkf301vKeJNAZYJTxWzeJhFd-DhQ47tMctc3YYkwZkQ5I_9fGYb_f0oBcw4esh.JNwuuHud78h6S99NO1oBQQ.0RwckPYATBgvw67upkAQ1AezETHc-gh3rryz19i5ryc.3XClRTScgzfMgLCHxHHoRF8mm9VVGXv_Ahtx65PskKQ",
		},
	}

	for i, msgs := range aesSampleMessages {
		for _, msg := range msgs {
			obj, err := ParseEncrypted(msg)
			if err != nil {
				t.Error("unable to parse message", msg, err)
				continue
			}
			plaintext, err := obj.Decrypt(aesTestKeys[i])
			if err != nil {
				t.Error("unable to decrypt message", msg, err)
				continue
			}
			if string(plaintext) != "Lorem ipsum dolor sit amet" {
				t.Error("plaintext is not what we expected for msg", msg)
			}
		}
	}
}

// Test vectors generated with jose4j
func TestSampleJose4jJWEMessagesECDH(t *testing.T) {
	ecTestKey := &ecdsa.PrivateKey{
		PublicKey: ecdsa.PublicKey{
			Curve: elliptic.P256(),
			X:     fromBase64Int("weNJy2HscCSM6AEDTDg04biOvhFhyyWvOHQfeF_PxMQ"),
			Y:     fromBase64Int("e8lnCO-AlStT-NJVX-crhB7QRYhiix03illJOVAOyck"),
		},
		D: fromBase64Int("VEmDZpDXXK8p8N0Cndsxs924q6nS1RXFASRl6BfUqdw"),
	}

	ecSampleMessages := []string{
		"eyJhbGciOiJFQ0RILUVTIiwiZW5jIjoiQTEyOENCQy1IUzI1NiIsImVwayI6eyJrdHkiOiJFQyIsIngiOiJTQzAtRnJHUkVvVkpKSmg1TGhORmZqZnFXMC1XSUFyd3RZMzJzQmFQVVh3IiwieSI6ImFQMWlPRENveU9laTVyS1l2VENMNlRMZFN5UEdUN0djMnFsRnBwNXdiWFEiLCJjcnYiOiJQLTI1NiJ9fQ..3mifklTnTTGuA_etSUBBCw.dj8KFM8OlrQ3rT35nHcHZ7A5p84VB2OZb054ghSjS-M.KOIgnJjz87LGqMtikXGxXw",
		"eyJhbGciOiJFQ0RILUVTIiwiZW5jIjoiQTE5MkNCQy1IUzM4NCIsImVwayI6eyJrdHkiOiJFQyIsIngiOiJUaHRGc0lRZ1E5MkZOYWFMbUFDQURLbE93dmNGVlRORHc4ampfWlJidUxjIiwieSI6IjJmRDZ3UXc3YmpYTm1nVThXMGpFbnl5ZUZkX3Y4ZmpDa3l1R29vTFhGM0EiLCJjcnYiOiJQLTI1NiJ9fQ..90zFayMkKc-fQC_19f6P3A.P1Y_7lMnfkUQOXW_en31lKZ3zAn1nEYn6fXLjmyVPrQ.hrgwy1cePVfhMWT0h-crKTXldglHZ-4g",
		"eyJhbGciOiJFQ0RILUVTIiwiZW5jIjoiQTI1NkNCQy1IUzUxMiIsImVwayI6eyJrdHkiOiJFQyIsIngiOiI5R1Z6c3VKNWgySl96UURVUFR3WU5zUkFzVzZfY2RzN0pELVQ2RDREQ1ZVIiwieSI6InFZVGl1dVU4aTB1WFpoaS14VGlRNlZJQm5vanFoWENPVnpmWm1pR2lRTEUiLCJjcnYiOiJQLTI1NiJ9fQ..v2reRlDkIsw3eWEsTCc1NA.0qakrFdbhtBCTSl7EREf9sxgHBP9I-Xw29OTJYnrqP8.54ozViEBYYmRkcKp7d2Ztt4hzjQ9Vb5zCeijN_RQrcI",
		"eyJhbGciOiJFQ0RILUVTK0EyNTZLVyIsImVuYyI6IkExMjhDQkMtSFMyNTYiLCJlcGsiOnsia3R5IjoiRUMiLCJ4IjoiOElUemg3VVFaaUthTWtfME9qX1hFaHZENXpUWjE2Ti13WVdjeTJYUC1tdyIsInkiOiJPNUJiVEk0bUFpU005ZmpCejBRU3pXaU5vbnl3cWlQLUN0RGgwdnNGYXNRIiwiY3J2IjoiUC0yNTYifX0.D3DP3wqPvJv4TYYfhnfrOG6nsM-MMH_CqGfnOGjgdXHNF7xRwEJBOA.WL9Kz3gNYA7S5Rs5mKcXmA.EmQkXhO_nFqAwxJWaM0DH4s3pmCscZovB8YWJ3Ru4N8.Bf88uzwfxiyTjpejU5B0Ng",
		"eyJhbGciOiJFQ0RILUVTK0EyNTZLVyIsImVuYyI6IkExOTJDQkMtSFMzODQiLCJlcGsiOnsia3R5IjoiRUMiLCJ4IjoiMjlJMk4zRkF0UlBlNGhzYjRLWlhTbmVyV0wyTVhtSUN1LXJJaXhNSHpJQSIsInkiOiJvMjY1bzFReEdmbDhzMHQ0U1JROS00RGNpc3otbXh4NlJ6WVF4SktyeWpJIiwiY3J2IjoiUC0yNTYifX0.DRmsmXz6fCnLc_njDIKdpM7Oc4jTqd_yd9J94TOUksAstEUkAl9Ie3Wg-Ji_LzbdX2xRLXIimcw.FwJOHPQhnqKJCfxt1_qRnQ.ssx3q1ZYILsMTln5q-K8HVn93BVPI5ViusstKMxZzRs.zzcfzWNYSdNDdQ4CiHfymj0bePaAbVaT",
		"eyJhbGciOiJFQ0RILUVTK0EyNTZLVyIsImVuYyI6IkEyNTZDQkMtSFM1MTIiLCJlcGsiOnsia3R5IjoiRUMiLCJ4IjoiRUp6bTViQnRzVXJNYTl2Y1Q2d1hZRXI3ZjNMcjB0N1V4SDZuZzdGcFF0VSIsInkiOiJRYTNDSDllVTFXYjItdFdVSDN3Sk9fTDVMZXRsRUlMQWNkNE9XR2tFd0hZIiwiY3J2IjoiUC0yNTYifX0.5WxwluZpVWAOJdVrsnDIlEc4_wfRE1gXOaQyx_rKkElNz157Ykf-JsAD7aEvXfx--NKF4js5zYyjeCtxWBhRWPOoNNZJlqV_.Iuo82-qsP2S1SgQQklAnrw.H4wB6XoLKOKWCu6Y3LPAEuHkvyvr-xAh4IBm53uRF8g._fOLKq0bqDZ8KNjni_MJ4olHNaYz376dV9eNmp9O9PU",
		"eyJhbGciOiJFQ0RILUVTK0ExOTJLVyIsImVuYyI6IkExMjhDQkMtSFMyNTYiLCJlcGsiOnsia3R5IjoiRUMiLCJ4IjoiZktNSG5sRkoxajBTSnJ3WGtVWlpaX3BtWHdUQlJtcHhlaTkxdUpaczUycyIsInkiOiJLRkxKaXhEUTJQcjEybWp1aFdYb3pna2U1V3lhWnhmTWlxZkJ0OEJpbkRvIiwiY3J2IjoiUC0yNTYifX0.2LSD2Mw4tyYJyfsmpVmzBtJRd12jMEYGdlhFbaXIbKi5A33CGNQ1tg.s40aAjmZOvK8Us86FCBdHg.jpYSMAKp___oMCoWM495mTfbi_YC80ObeoCmGE3H_gs.A6V-jJJRY1yz24CaXGUbzg",
		"eyJhbGciOiJFQ0RILUVTK0ExOTJLVyIsImVuYyI6IkExOTJDQkMtSFMzODQiLCJlcGsiOnsia3R5IjoiRUMiLCJ4IjoiSDRxcFUzeWtuRktWRnV4SmxLa3NZSE5ieHF3aXM0WWtCVVFHVE1Td05JQSIsInkiOiJHb0lpRUZaUGRRSHJCbVR4ZTA3akJoZmxrdWNqUjVoX1QwNWVXc3Zib0prIiwiY3J2IjoiUC0yNTYifX0.KTrwwV2uzD--gf3PGG-kjEAGgi7u0eMqZPZfa4kpyFGm3x8t2m1NHdz3t9rfiqjuaqsxPKhF4gs.cu16fEOzYaSxhHu_Ht9w4g.BRJdxVBI9spVtY5KQ6gTR4CNcKvmLUMKZap0AO-RF2I.DZyUaa2p6YCIaYtjWOjC9GN_VIYgySlZ",
		"eyJhbGciOiJFQ0RILUVTK0ExOTJLVyIsImVuYyI6IkEyNTZDQkMtSFM1MTIiLCJlcGsiOnsia3R5IjoiRUMiLCJ4IjoieDBYSGRkSGM2Q0ktSnlfbUVMOEZZRExhWnV0UkVFczR4c3BMQmcwZk1jbyIsInkiOiJEa0xzOUJGTlBkTTVTNkpLYVJ3cnV1TWMwcUFzWW9yNW9fZWp6NXBNVXFrIiwiY3J2IjoiUC0yNTYifX0.mfCxJ7JYIqTMqcAh5Vp2USF0eF7OhOeluqda7YagOUJNwxA9wC9o23DSoLUylfrZUfanZrJJJcG69awlv-LY7anOLHlp3Ht5.ec48A_JWb4qa_PVHWZaTfQ.kDAjIDb3LzJpfxNh-DiAmAuaKMYaOGSTb0rkiJLuVeY.oxGCpPlii4pr89XMk4b9s084LucTqPGU6TLbOW2MZoc",
		"eyJhbGciOiJFQ0RILUVTK0ExMjhLVyIsImVuYyI6IkExMjhDQkMtSFMyNTYiLCJlcGsiOnsia3R5IjoiRUMiLCJ4IjoiQXB5TnlqU2d0bmRUcFg0eENYenNDRnZva1l3X18weXg2dGRUYzdPUUhIMCIsInkiOiJYUHdHMDVDaW1vOGlhWmxZbDNsMEp3ZllhY1FZWHFuM2RRZEJUWFpldDZBIiwiY3J2IjoiUC0yNTYifX0.yTA2PwK9IPqkaGPenZ9R-gOn9m9rvcSEfuX_Nm8AkuwHIYLzzYeAEA.ZW1F1iyHYKfo-YoanNaIVg.PouKQD94DlPA5lbpfGJXY-EJhidC7l4vSayVN2vVzvA.MexquqtGaXKUvX7WBmD4bA",
		"eyJhbGciOiJFQ0RILUVTK0ExMjhLVyIsImVuYyI6IkExOTJDQkMtSFMzODQiLCJlcGsiOnsia3R5IjoiRUMiLCJ4IjoiaDRWeGNzNVUzWk1fTlp4WmJxQ3hMTVB5UmEtR2ktSVNZa0xDTzE1RHJkZyIsInkiOiJFeVotS3dWNVE5OXlnWk5zU0lpSldpR3hqbXNLUk1WVE5sTTNSd1VYTFRvIiwiY3J2IjoiUC0yNTYifX0.wo56VISyL1QAbi2HLuVut5NGF2FvxKt7B8zHzJ3FpmavPozfbVZV08-GSYQ6jLQWJ4xsO80I4Kg.3_9Bo5ozvD96WHGhqp_tfQ.48UkJ6jk6WK70QItb2QZr0edKH7O-aMuVahTEeqyfW4.ulMlY2tbC341ct20YSmNdtc84FRz1I4g",
		"eyJhbGciOiJFQ0RILUVTK0ExMjhLVyIsImVuYyI6IkEyNTZDQkMtSFM1MTIiLCJlcGsiOnsia3R5IjoiRUMiLCJ4IjoiN0xZRzZZWTJkel9ZaGNvNnRCcG1IX0tPREQ2X2hwX05tajdEc1c2RXgxcyIsInkiOiI5Y2lPeDcwUkdGT0tpVnBRX0NHQXB5NVlyeThDazBmUkpwNHVrQ2tjNmQ0IiwiY3J2IjoiUC0yNTYifX0.bWwW3J80k46HG1fQAZxUroko2OO8OKkeRavr_o3AnhJDMvp78OR229x-fZUaBm4uWv27_Yjm0X9T2H2lhlIli2Rl9v1PNC77.1NmsJBDGI1fDjRzyc4mtyA.9KfCFynQj7LmJq08qxAG4c-6ZPz1Lh3h3nUbgVwB0TI.cqech0d8XHzWfkWqgKZq1SlAfmO0PUwOsNVkuByVGWk",
	}

	for _, msg := range ecSampleMessages {
		obj, err := ParseEncrypted(msg)
		if err != nil {
			t.Error("unable to parse message", msg, err)
			continue
		}
		plaintext, err := obj.Decrypt(ecTestKey)
		if err != nil {
			t.Error("unable to decrypt message", msg, err)
			continue
		}
		if string(plaintext) != "Lorem ipsum dolor sit amet." {
			t.Error("plaintext is not what we expected for msg", msg)
		}
	}
}
