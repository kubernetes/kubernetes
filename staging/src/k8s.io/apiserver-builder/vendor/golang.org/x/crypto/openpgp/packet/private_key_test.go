// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/hex"
	"hash"
	"io"
	"testing"
	"time"
)

var privateKeyTests = []struct {
	privateKeyHex string
	creationTime  time.Time
}{
	{
		privKeyRSAHex,
		time.Unix(0x4cc349a8, 0),
	},
	{
		privKeyElGamalHex,
		time.Unix(0x4df9ee1a, 0),
	},
}

func TestPrivateKeyRead(t *testing.T) {
	for i, test := range privateKeyTests {
		packet, err := Read(readerFromHex(test.privateKeyHex))
		if err != nil {
			t.Errorf("#%d: failed to parse: %s", i, err)
			continue
		}

		privKey := packet.(*PrivateKey)

		if !privKey.Encrypted {
			t.Errorf("#%d: private key isn't encrypted", i)
			continue
		}

		err = privKey.Decrypt([]byte("wrong password"))
		if err == nil {
			t.Errorf("#%d: decrypted with incorrect key", i)
			continue
		}

		err = privKey.Decrypt([]byte("testing"))
		if err != nil {
			t.Errorf("#%d: failed to decrypt: %s", i, err)
			continue
		}

		if !privKey.CreationTime.Equal(test.creationTime) || privKey.Encrypted {
			t.Errorf("#%d: bad result, got: %#v", i, privKey)
		}
	}
}

func populateHash(hashFunc crypto.Hash, msg []byte) (hash.Hash, error) {
	h := hashFunc.New()
	if _, err := h.Write(msg); err != nil {
		return nil, err
	}
	return h, nil
}

func TestRSAPrivateKey(t *testing.T) {
	privKeyDER, _ := hex.DecodeString(pkcs1PrivKeyHex)
	rsaPriv, err := x509.ParsePKCS1PrivateKey(privKeyDER)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := NewRSAPrivateKey(time.Now(), rsaPriv).Serialize(&buf); err != nil {
		t.Fatal(err)
	}

	p, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}

	priv, ok := p.(*PrivateKey)
	if !ok {
		t.Fatal("didn't parse private key")
	}

	sig := &Signature{
		PubKeyAlgo: PubKeyAlgoRSA,
		Hash:       crypto.SHA256,
	}
	msg := []byte("Hello World!")

	h, err := populateHash(sig.Hash, msg)
	if err != nil {
		t.Fatal(err)
	}
	if err := sig.Sign(h, priv, nil); err != nil {
		t.Fatal(err)
	}

	if h, err = populateHash(sig.Hash, msg); err != nil {
		t.Fatal(err)
	}
	if err := priv.VerifySignature(h, sig); err != nil {
		t.Fatal(err)
	}
}

func TestECDSAPrivateKey(t *testing.T) {
	ecdsaPriv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := NewECDSAPrivateKey(time.Now(), ecdsaPriv).Serialize(&buf); err != nil {
		t.Fatal(err)
	}

	p, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}

	priv, ok := p.(*PrivateKey)
	if !ok {
		t.Fatal("didn't parse private key")
	}

	sig := &Signature{
		PubKeyAlgo: PubKeyAlgoECDSA,
		Hash:       crypto.SHA256,
	}
	msg := []byte("Hello World!")

	h, err := populateHash(sig.Hash, msg)
	if err != nil {
		t.Fatal(err)
	}
	if err := sig.Sign(h, priv, nil); err != nil {
		t.Fatal(err)
	}

	if h, err = populateHash(sig.Hash, msg); err != nil {
		t.Fatal(err)
	}
	if err := priv.VerifySignature(h, sig); err != nil {
		t.Fatal(err)
	}
}

type rsaSigner struct {
	priv *rsa.PrivateKey
}

func (s *rsaSigner) Public() crypto.PublicKey {
	return s.priv.PublicKey
}

func (s *rsaSigner) Sign(rand io.Reader, msg []byte, opts crypto.SignerOpts) ([]byte, error) {
	return s.priv.Sign(rand, msg, opts)
}

func TestRSASignerPrivateKey(t *testing.T) {
	rsaPriv, err := rsa.GenerateKey(rand.Reader, 1024)
	if err != nil {
		t.Fatal(err)
	}

	priv := NewSignerPrivateKey(time.Now(), &rsaSigner{rsaPriv})

	if priv.PubKeyAlgo != PubKeyAlgoRSASignOnly {
		t.Fatal("NewSignerPrivateKey should have made a sign-only RSA private key")
	}

	sig := &Signature{
		PubKeyAlgo: PubKeyAlgoRSASignOnly,
		Hash:       crypto.SHA256,
	}
	msg := []byte("Hello World!")

	h, err := populateHash(sig.Hash, msg)
	if err != nil {
		t.Fatal(err)
	}
	if err := sig.Sign(h, priv, nil); err != nil {
		t.Fatal(err)
	}

	if h, err = populateHash(sig.Hash, msg); err != nil {
		t.Fatal(err)
	}
	if err := priv.VerifySignature(h, sig); err != nil {
		t.Fatal(err)
	}
}

type ecdsaSigner struct {
	priv *ecdsa.PrivateKey
}

func (s *ecdsaSigner) Public() crypto.PublicKey {
	return s.priv.PublicKey
}

func (s *ecdsaSigner) Sign(rand io.Reader, msg []byte, opts crypto.SignerOpts) ([]byte, error) {
	return s.priv.Sign(rand, msg, opts)
}

func TestECDSASignerPrivateKey(t *testing.T) {
	ecdsaPriv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	priv := NewSignerPrivateKey(time.Now(), &ecdsaSigner{ecdsaPriv})

	if priv.PubKeyAlgo != PubKeyAlgoECDSA {
		t.Fatal("NewSignerPrivateKey should have made a ECSDA private key")
	}

	sig := &Signature{
		PubKeyAlgo: PubKeyAlgoECDSA,
		Hash:       crypto.SHA256,
	}
	msg := []byte("Hello World!")

	h, err := populateHash(sig.Hash, msg)
	if err != nil {
		t.Fatal(err)
	}
	if err := sig.Sign(h, priv, nil); err != nil {
		t.Fatal(err)
	}

	if h, err = populateHash(sig.Hash, msg); err != nil {
		t.Fatal(err)
	}
	if err := priv.VerifySignature(h, sig); err != nil {
		t.Fatal(err)
	}
}

func TestIssue11505(t *testing.T) {
	// parsing a rsa private key with p or q == 1 used to panic due to a divide by zero
	_, _ = Read(readerFromHex("9c3004303030300100000011303030000000000000010130303030303030303030303030303030303030303030303030303030303030303030303030303030303030"))
}

// Generated with `gpg --export-secret-keys "Test Key 2"`
const privKeyRSAHex = "9501fe044cc349a8010400b70ca0010e98c090008d45d1ee8f9113bd5861fd57b88bacb7c68658747663f1e1a3b5a98f32fda6472373c024b97359cd2efc88ff60f77751adfbf6af5e615e6a1408cfad8bf0cea30b0d5f53aa27ad59089ba9b15b7ebc2777a25d7b436144027e3bcd203909f147d0e332b240cf63d3395f5dfe0df0a6c04e8655af7eacdf0011010001fe0303024a252e7d475fd445607de39a265472aa74a9320ba2dac395faa687e9e0336aeb7e9a7397e511b5afd9dc84557c80ac0f3d4d7bfec5ae16f20d41c8c84a04552a33870b930420e230e179564f6d19bb153145e76c33ae993886c388832b0fa042ddda7f133924f3854481533e0ede31d51278c0519b29abc3bf53da673e13e3e1214b52413d179d7f66deee35cac8eacb060f78379d70ef4af8607e68131ff529439668fc39c9ce6dfef8a5ac234d234802cbfb749a26107db26406213ae5c06d4673253a3cbee1fcbae58d6ab77e38d6e2c0e7c6317c48e054edadb5a40d0d48acb44643d998139a8a66bb820be1f3f80185bc777d14b5954b60effe2448a036d565c6bc0b915fcea518acdd20ab07bc1529f561c58cd044f723109b93f6fd99f876ff891d64306b5d08f48bab59f38695e9109c4dec34013ba3153488ce070268381ba923ee1eb77125b36afcb4347ec3478c8f2735b06ef17351d872e577fa95d0c397c88c71b59629a36aec"

// Generated by `gpg --export-secret-keys` followed by a manual extraction of
// the ElGamal subkey from the packets.
const privKeyElGamalHex = "9d0157044df9ee1a100400eb8e136a58ec39b582629cdadf830bc64e0a94ed8103ca8bb247b27b11b46d1d25297ef4bcc3071785ba0c0bedfe89eabc5287fcc0edf81ab5896c1c8e4b20d27d79813c7aede75320b33eaeeaa586edc00fd1036c10133e6ba0ff277245d0d59d04b2b3421b7244aca5f4a8d870c6f1c1fbff9e1c26699a860b9504f35ca1d700030503fd1ededd3b840795be6d9ccbe3c51ee42e2f39233c432b831ddd9c4e72b7025a819317e47bf94f9ee316d7273b05d5fcf2999c3a681f519b1234bbfa6d359b4752bd9c3f77d6b6456cde152464763414ca130f4e91d91041432f90620fec0e6d6b5116076c2985d5aeaae13be492b9b329efcaf7ee25120159a0a30cd976b42d7afe030302dae7eb80db744d4960c4df930d57e87fe81412eaace9f900e6c839817a614ddb75ba6603b9417c33ea7b6c93967dfa2bcff3fa3c74a5ce2c962db65b03aece14c96cbd0038fc"

// pkcs1PrivKeyHex is a PKCS#1, RSA private key.
// Generated by `openssl genrsa 1024 | openssl rsa -outform DER  | xxd -p`
const pkcs1PrivKeyHex = "3082025d02010002818100e98edfa1c3b35884a54d0b36a6a603b0290fa85e49e30fa23fc94fef9c6790bc4849928607aa48d809da326fb42a969d06ad756b98b9c1a90f5d4a2b6d0ac05953c97f4da3120164a21a679793ce181c906dc01d235cc085ddcdf6ea06c389b6ab8885dfd685959e693138856a68a7e5db263337ff82a088d583a897cf2d59e9020301000102818100b6d5c9eb70b02d5369b3ee5b520a14490b5bde8a317d36f7e4c74b7460141311d1e5067735f8f01d6f5908b2b96fbd881f7a1ab9a84d82753e39e19e2d36856be960d05ac9ef8e8782ea1b6d65aee28fdfe1d61451e8cff0adfe84322f12cf455028b581cf60eb9e0e140ba5d21aeba6c2634d7c65318b9a665fc01c3191ca21024100fa5e818da3705b0fa33278bb28d4b6f6050388af2d4b75ec9375dd91ccf2e7d7068086a8b82a8f6282e4fbbdb8a7f2622eb97295249d87acea7f5f816f54d347024100eecf9406d7dc49cdfb95ab1eff4064de84c7a30f64b2798936a0d2018ba9eb52e4b636f82e96c49cc63b80b675e91e40d1b2e4017d4b9adaf33ab3d9cf1c214f024100c173704ace742c082323066226a4655226819a85304c542b9dacbeacbf5d1881ee863485fcf6f59f3a604f9b42289282067447f2b13dfeed3eab7851fc81e0550240741fc41f3fc002b382eed8730e33c5d8de40256e4accee846667f536832f711ab1d4590e7db91a8a116ac5bff3be13d3f9243ff2e976662aa9b395d907f8e9c9024046a5696c9ef882363e06c9fa4e2f5b580906452befba03f4a99d0f873697ef1f851d2226ca7934b30b7c3e80cb634a67172bbbf4781735fe3e09263e2dd723e7"
