package openpgp

import (
	"bytes"
	"crypto"
	"strings"
	"testing"
	"time"

	"golang.org/x/crypto/openpgp/errors"
	"golang.org/x/crypto/openpgp/packet"
)

func TestKeyExpiry(t *testing.T) {
	kring, err := ReadKeyRing(readerFromHex(expiringKeyHex))
	if err != nil {
		t.Fatal(err)
	}
	entity := kring[0]

	const timeFormat = "2006-01-02"
	time1, _ := time.Parse(timeFormat, "2013-07-01")

	// The expiringKeyHex key is structured as:
	//
	// pub  1024R/5E237D8C  created: 2013-07-01                      expires: 2013-07-31  usage: SC
	// sub  1024R/1ABB25A0  created: 2013-07-01 23:11:07 +0200 CEST  expires: 2013-07-08  usage: E
	// sub  1024R/96A672F5  created: 2013-07-01 23:11:23 +0200 CEST  expires: 2013-07-31  usage: E
	//
	// So this should select the newest, non-expired encryption key.
	key, _ := entity.encryptionKey(time1)
	if id, expected := key.PublicKey.KeyIdShortString(), "96A672F5"; id != expected {
		t.Errorf("Expected key %s at time %s, but got key %s", expected, time1.Format(timeFormat), id)
	}

	// Once the first encryption subkey has expired, the second should be
	// selected.
	time2, _ := time.Parse(timeFormat, "2013-07-09")
	key, _ = entity.encryptionKey(time2)
	if id, expected := key.PublicKey.KeyIdShortString(), "96A672F5"; id != expected {
		t.Errorf("Expected key %s at time %s, but got key %s", expected, time2.Format(timeFormat), id)
	}

	// Once all the keys have expired, nothing should be returned.
	time3, _ := time.Parse(timeFormat, "2013-08-01")
	if key, ok := entity.encryptionKey(time3); ok {
		t.Errorf("Expected no key at time %s, but got key %s", time3.Format(timeFormat), key.PublicKey.KeyIdShortString())
	}
}

func TestMissingCrossSignature(t *testing.T) {
	// This public key has a signing subkey, but the subkey does not
	// contain a cross-signature.
	keys, err := ReadArmoredKeyRing(bytes.NewBufferString(missingCrossSignatureKey))
	if len(keys) != 0 {
		t.Errorf("Accepted key with missing cross signature")
	}
	if err == nil {
		t.Fatal("Failed to detect error in keyring with missing cross signature")
	}
	structural, ok := err.(errors.StructuralError)
	if !ok {
		t.Fatalf("Unexpected class of error: %T. Wanted StructuralError", err)
	}
	const expectedMsg = "signing subkey is missing cross-signature"
	if !strings.Contains(string(structural), expectedMsg) {
		t.Fatalf("Unexpected error: %q. Expected it to contain %q", err, expectedMsg)
	}
}

func TestInvalidCrossSignature(t *testing.T) {
	// This public key has a signing subkey, and the subkey has an
	// embedded cross-signature. However, the cross-signature does
	// not correctly validate over the primary and subkey.
	keys, err := ReadArmoredKeyRing(bytes.NewBufferString(invalidCrossSignatureKey))
	if len(keys) != 0 {
		t.Errorf("Accepted key with invalid cross signature")
	}
	if err == nil {
		t.Fatal("Failed to detect error in keyring with an invalid cross signature")
	}
	structural, ok := err.(errors.StructuralError)
	if !ok {
		t.Fatalf("Unexpected class of error: %T. Wanted StructuralError", err)
	}
	const expectedMsg = "subkey signature invalid"
	if !strings.Contains(string(structural), expectedMsg) {
		t.Fatalf("Unexpected error: %q. Expected it to contain %q", err, expectedMsg)
	}
}

func TestGoodCrossSignature(t *testing.T) {
	// This public key has a signing subkey, and the subkey has an
	// embedded cross-signature which correctly validates over the
	// primary and subkey.
	keys, err := ReadArmoredKeyRing(bytes.NewBufferString(goodCrossSignatureKey))
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 1 {
		t.Errorf("Failed to accept key with good cross signature, %d", len(keys))
	}
	if len(keys[0].Subkeys) != 1 {
		t.Errorf("Failed to accept good subkey, %d", len(keys[0].Subkeys))
	}
}

func TestRevokedUserID(t *testing.T) {
	// This key contains 2 UIDs, one of which is revoked:
	// [ultimate] (1)  Golang Gopher <no-reply@golang.com>
	// [ revoked] (2)  Golang Gopher <revoked@golang.com>
	keys, err := ReadArmoredKeyRing(bytes.NewBufferString(revokedUserIDKey))
	if err != nil {
		t.Fatal(err)
	}

	if len(keys) != 1 {
		t.Fatal("Failed to read key with a revoked user id")
	}

	var identities []*Identity
	for _, identity := range keys[0].Identities {
		identities = append(identities, identity)
	}

	if numIdentities, numExpected := len(identities), 1; numIdentities != numExpected {
		t.Errorf("obtained %d identities, expected %d", numIdentities, numExpected)
	}

	if identityName, expectedName := identities[0].Name, "Golang Gopher <no-reply@golang.com>"; identityName != expectedName {
		t.Errorf("obtained identity %s expected %s", identityName, expectedName)
	}
}

// TestExternallyRevokableKey attempts to load and parse a key with a third party revocation permission.
func TestExternallyRevocableKey(t *testing.T) {
	kring, err := ReadKeyRing(readerFromHex(subkeyUsageHex))
	if err != nil {
		t.Fatal(err)
	}

	// The 0xA42704B92866382A key can be revoked by 0xBE3893CB843D0FE70C
	// according to this signature that appears within the key:
	// :signature packet: algo 1, keyid A42704B92866382A
	//    version 4, created 1396409682, md5len 0, sigclass 0x1f
	//    digest algo 2, begin of digest a9 84
	//    hashed subpkt 2 len 4 (sig created 2014-04-02)
	//    hashed subpkt 12 len 22 (revocation key: c=80 a=1 f=CE094AA433F7040BB2DDF0BE3893CB843D0FE70C)
	//    hashed subpkt 7 len 1 (not revocable)
	//    subpkt 16 len 8 (issuer key ID A42704B92866382A)
	//    data: [1024 bits]

	id := uint64(0xA42704B92866382A)
	keys := kring.KeysById(id)
	if len(keys) != 1 {
		t.Errorf("Expected to find key id %X, but got %d matches", id, len(keys))
	}
}

func TestKeyRevocation(t *testing.T) {
	kring, err := ReadKeyRing(readerFromHex(revokedKeyHex))
	if err != nil {
		t.Fatal(err)
	}

	// revokedKeyHex contains these keys:
	// pub   1024R/9A34F7C0 2014-03-25 [revoked: 2014-03-25]
	// sub   1024R/1BA3CD60 2014-03-25 [revoked: 2014-03-25]
	ids := []uint64{0xA401D9F09A34F7C0, 0x5CD3BE0A1BA3CD60}

	for _, id := range ids {
		keys := kring.KeysById(id)
		if len(keys) != 1 {
			t.Errorf("Expected KeysById to find revoked key %X, but got %d matches", id, len(keys))
		}
		keys = kring.KeysByIdUsage(id, 0)
		if len(keys) != 0 {
			t.Errorf("Expected KeysByIdUsage to filter out revoked key %X, but got %d matches", id, len(keys))
		}
	}
}

func TestKeyWithRevokedSubKey(t *testing.T) {
	// This key contains a revoked sub key:
	//  pub   rsa1024/0x4CBD826C39074E38 2018-06-14 [SC]
	//        Key fingerprint = 3F95 169F 3FFA 7D3F 2B47  6F0C 4CBD 826C 3907 4E38
	//  uid   Golang Gopher <no-reply@golang.com>
	//  sub   rsa1024/0x945DB1AF61D85727 2018-06-14 [S] [revoked: 2018-06-14]

	keys, err := ReadArmoredKeyRing(bytes.NewBufferString(keyWithSubKey))
	if err != nil {
		t.Fatal(err)
	}

	if len(keys) != 1 {
		t.Fatal("Failed to read key with a sub key")
	}

	identity := keys[0].Identities["Golang Gopher <no-reply@golang.com>"]

	// Test for an issue where Subkey Binding Signatures (RFC 4880 5.2.1) were added to the identity
	// preceding the Subkey Packet if the Subkey Packet was followed by more than one signature.
	// For example, the current key has the following layout:
	//    PUBKEY UID SELFSIG SUBKEY REV SELFSIG
	// The last SELFSIG would be added to the UID's signatures. This is wrong.
	if numIdentitySigs, numExpected := len(identity.Signatures), 0; numIdentitySigs != numExpected {
		t.Fatalf("got %d identity signatures, expected %d", numIdentitySigs, numExpected)
	}

	if numSubKeys, numExpected := len(keys[0].Subkeys), 1; numSubKeys != numExpected {
		t.Fatalf("got %d subkeys, expected %d", numSubKeys, numExpected)
	}

	subKey := keys[0].Subkeys[0]
	if subKey.Sig == nil {
		t.Fatalf("subkey signature is nil")
	}

}

func TestSubkeyRevocation(t *testing.T) {
	kring, err := ReadKeyRing(readerFromHex(revokedSubkeyHex))
	if err != nil {
		t.Fatal(err)
	}

	// revokedSubkeyHex contains these keys:
	// pub   1024R/4EF7E4BECCDE97F0 2014-03-25
	// sub   1024R/D63636E2B96AE423 2014-03-25
	// sub   1024D/DBCE4EE19529437F 2014-03-25
	// sub   1024R/677815E371C2FD23 2014-03-25 [revoked: 2014-03-25]
	validKeys := []uint64{0x4EF7E4BECCDE97F0, 0xD63636E2B96AE423, 0xDBCE4EE19529437F}
	revokedKey := uint64(0x677815E371C2FD23)

	for _, id := range validKeys {
		keys := kring.KeysById(id)
		if len(keys) != 1 {
			t.Errorf("Expected KeysById to find key %X, but got %d matches", id, len(keys))
		}
		keys = kring.KeysByIdUsage(id, 0)
		if len(keys) != 1 {
			t.Errorf("Expected KeysByIdUsage to find key %X, but got %d matches", id, len(keys))
		}
	}

	keys := kring.KeysById(revokedKey)
	if len(keys) != 1 {
		t.Errorf("Expected KeysById to find key %X, but got %d matches", revokedKey, len(keys))
	}

	keys = kring.KeysByIdUsage(revokedKey, 0)
	if len(keys) != 0 {
		t.Errorf("Expected KeysByIdUsage to filter out revoked key %X, but got %d matches", revokedKey, len(keys))
	}
}

func TestKeyWithSubKeyAndBadSelfSigOrder(t *testing.T) {
	// This key was altered so that the self signatures following the
	// subkey are in a sub-optimal order.
	//
	// Note: Should someone have to create a similar key again, look into
	//       gpgsplit, gpg --dearmor, and gpg --enarmor.
	//
	// The packet ordering is the following:
	//    PUBKEY UID UIDSELFSIG SUBKEY SELFSIG1 SELFSIG2
	//
	// Where:
	//    SELFSIG1 expires on 2018-06-14 and was created first
	//    SELFSIG2 does not expire and was created after SELFSIG1
	//
	// Test for RFC 4880 5.2.3.3:
	// > An implementation that encounters multiple self-signatures on the
	// > same object may resolve the ambiguity in any way it sees fit, but it
	// > is RECOMMENDED that priority be given to the most recent self-
	// > signature.
	//
	// This means that we should keep SELFSIG2.

	keys, err := ReadArmoredKeyRing(bytes.NewBufferString(keyWithSubKeyAndBadSelfSigOrder))
	if err != nil {
		t.Fatal(err)
	}

	if len(keys) != 1 {
		t.Fatal("Failed to read key with a sub key and a bad selfsig packet order")
	}

	key := keys[0]

	if numKeys, expected := len(key.Subkeys), 1; numKeys != expected {
		t.Fatalf("Read %d subkeys, expected %d", numKeys, expected)
	}

	subKey := key.Subkeys[0]

	if lifetime := subKey.Sig.KeyLifetimeSecs; lifetime != nil {
		t.Errorf("The signature has a key lifetime (%d), but it should be nil", *lifetime)
	}

}

func TestKeyUsage(t *testing.T) {
	kring, err := ReadKeyRing(readerFromHex(subkeyUsageHex))
	if err != nil {
		t.Fatal(err)
	}

	// subkeyUsageHex contains these keys:
	// pub  1024R/2866382A  created: 2014-04-01  expires: never       usage: SC
	// sub  1024R/936C9153  created: 2014-04-01  expires: never       usage: E
	// sub  1024R/64D5F5BB  created: 2014-04-02  expires: never       usage: E
	// sub  1024D/BC0BA992  created: 2014-04-02  expires: never       usage: S
	certifiers := []uint64{0xA42704B92866382A}
	signers := []uint64{0xA42704B92866382A, 0x42CE2C64BC0BA992}
	encrypters := []uint64{0x09C0C7D9936C9153, 0xC104E98664D5F5BB}

	for _, id := range certifiers {
		keys := kring.KeysByIdUsage(id, packet.KeyFlagCertify)
		if len(keys) == 1 {
			if keys[0].PublicKey.KeyId != id {
				t.Errorf("Expected to find certifier key id %X, but got %X", id, keys[0].PublicKey.KeyId)
			}
		} else {
			t.Errorf("Expected one match for certifier key id %X, but got %d matches", id, len(keys))
		}
	}

	for _, id := range signers {
		keys := kring.KeysByIdUsage(id, packet.KeyFlagSign)
		if len(keys) == 1 {
			if keys[0].PublicKey.KeyId != id {
				t.Errorf("Expected to find signing key id %X, but got %X", id, keys[0].PublicKey.KeyId)
			}
		} else {
			t.Errorf("Expected one match for signing key id %X, but got %d matches", id, len(keys))
		}

		// This keyring contains no encryption keys that are also good for signing.
		keys = kring.KeysByIdUsage(id, packet.KeyFlagEncryptStorage|packet.KeyFlagEncryptCommunications)
		if len(keys) != 0 {
			t.Errorf("Unexpected match for encryption key id %X", id)
		}
	}

	for _, id := range encrypters {
		keys := kring.KeysByIdUsage(id, packet.KeyFlagEncryptStorage|packet.KeyFlagEncryptCommunications)
		if len(keys) == 1 {
			if keys[0].PublicKey.KeyId != id {
				t.Errorf("Expected to find encryption key id %X, but got %X", id, keys[0].PublicKey.KeyId)
			}
		} else {
			t.Errorf("Expected one match for encryption key id %X, but got %d matches", id, len(keys))
		}

		// This keyring contains no encryption keys that are also good for signing.
		keys = kring.KeysByIdUsage(id, packet.KeyFlagSign)
		if len(keys) != 0 {
			t.Errorf("Unexpected match for signing key id %X", id)
		}
	}
}

func TestIdVerification(t *testing.T) {
	kring, err := ReadKeyRing(readerFromHex(testKeys1And2PrivateHex))
	if err != nil {
		t.Fatal(err)
	}
	if err := kring[1].PrivateKey.Decrypt([]byte("passphrase")); err != nil {
		t.Fatal(err)
	}

	const identity = "Test Key 1 (RSA)"
	if err := kring[0].SignIdentity(identity, kring[1], nil); err != nil {
		t.Fatal(err)
	}

	ident, ok := kring[0].Identities[identity]
	if !ok {
		t.Fatal("identity missing from key after signing")
	}

	checked := false
	for _, sig := range ident.Signatures {
		if sig.IssuerKeyId == nil || *sig.IssuerKeyId != kring[1].PrimaryKey.KeyId {
			continue
		}

		if err := kring[1].PrimaryKey.VerifyUserIdSignature(identity, kring[0].PrimaryKey, sig); err != nil {
			t.Fatalf("error verifying new identity signature: %s", err)
		}
		checked = true
		break
	}

	if !checked {
		t.Fatal("didn't find identity signature in Entity")
	}
}

func TestNewEntityWithPreferredHash(t *testing.T) {
	c := &packet.Config{
		DefaultHash: crypto.SHA256,
	}
	entity, err := NewEntity("Golang Gopher", "Test Key", "no-reply@golang.com", c)
	if err != nil {
		t.Fatal(err)
	}

	for _, identity := range entity.Identities {
		if len(identity.SelfSignature.PreferredHash) == 0 {
			t.Fatal("didn't find a preferred hash in self signature")
		}
		ph := hashToHashId(c.DefaultHash)
		if identity.SelfSignature.PreferredHash[0] != ph {
			t.Fatalf("Expected preferred hash to be %d, got %d", ph, identity.SelfSignature.PreferredHash[0])
		}
	}
}

func TestNewEntityWithoutPreferredHash(t *testing.T) {
	entity, err := NewEntity("Golang Gopher", "Test Key", "no-reply@golang.com", nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, identity := range entity.Identities {
		if len(identity.SelfSignature.PreferredHash) != 0 {
			t.Fatalf("Expected preferred hash to be empty but got length %d", len(identity.SelfSignature.PreferredHash))
		}
	}
}

func TestNewEntityCorrectName(t *testing.T) {
	entity, err := NewEntity("Golang Gopher", "Test Key", "no-reply@golang.com", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(entity.Identities) != 1 {
		t.Fatalf("len(entity.Identities) = %d, want 1", len(entity.Identities))
	}
	var got string
	for _, i := range entity.Identities {
		got = i.Name
	}
	want := "Golang Gopher (Test Key) <no-reply@golang.com>"
	if got != want {
		t.Fatalf("Identity.Name = %q, want %q", got, want)
	}
}

func TestNewEntityWithPreferredSymmetric(t *testing.T) {
	c := &packet.Config{
		DefaultCipher: packet.CipherAES256,
	}
	entity, err := NewEntity("Golang Gopher", "Test Key", "no-reply@golang.com", c)
	if err != nil {
		t.Fatal(err)
	}

	for _, identity := range entity.Identities {
		if len(identity.SelfSignature.PreferredSymmetric) == 0 {
			t.Fatal("didn't find a preferred cipher in self signature")
		}
		if identity.SelfSignature.PreferredSymmetric[0] != uint8(c.DefaultCipher) {
			t.Fatalf("Expected preferred cipher to be %d, got %d", uint8(c.DefaultCipher), identity.SelfSignature.PreferredSymmetric[0])
		}
	}
}

func TestNewEntityWithoutPreferredSymmetric(t *testing.T) {
	entity, err := NewEntity("Golang Gopher", "Test Key", "no-reply@golang.com", nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, identity := range entity.Identities {
		if len(identity.SelfSignature.PreferredSymmetric) != 0 {
			t.Fatalf("Expected preferred cipher to be empty but got length %d", len(identity.SelfSignature.PreferredSymmetric))
		}
	}
}

func TestNewEntityPublicSerialization(t *testing.T) {
	entity, err := NewEntity("Golang Gopher", "Test Key", "no-reply@golang.com", nil)
	if err != nil {
		t.Fatal(err)
	}
	serializedEntity := bytes.NewBuffer(nil)
	entity.Serialize(serializedEntity)

	_, err = ReadEntity(packet.NewReader(bytes.NewBuffer(serializedEntity.Bytes())))
	if err != nil {
		t.Fatal(err)
	}
}
