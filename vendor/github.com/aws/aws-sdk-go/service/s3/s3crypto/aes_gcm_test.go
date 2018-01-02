package s3crypto

import (
	"bytes"
	"encoding/hex"
	"io/ioutil"
	"testing"
)

// AES GCM
func TestAES_GCM_NIST_gcmEncryptExtIV256_PTLen_128_Test_0(t *testing.T) {
	iv, _ := hex.DecodeString("0d18e06c7c725ac9e362e1ce")
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	plaintext, _ := hex.DecodeString("2db5168e932556f8089a0622981d017d")
	expected, _ := hex.DecodeString("fa4362189661d163fcd6a56d8bf0405ad636ac1bbedd5cc3ee727dc2ab4a9489")
	tag, _ := hex.DecodeString("d636ac1bbedd5cc3ee727dc2ab4a9489")
	aesgcmTest(t, iv, key, plaintext, expected, tag)
}

func TestAES_GCM_NIST_gcmEncryptExtIV256_PTLen_104_Test_3(t *testing.T) {
	iv, _ := hex.DecodeString("4742357c335913153ff0eb0f")
	key, _ := hex.DecodeString("e5a0eb92cc2b064e1bc80891faf1fab5e9a17a9c3a984e25416720e30e6c2b21")
	plaintext, _ := hex.DecodeString("8499893e16b0ba8b007d54665a")
	expected, _ := hex.DecodeString("eb8e6175f1fe38eb1acf95fd5188a8b74bb74fda553e91020a23deed45")
	tag, _ := hex.DecodeString("88a8b74bb74fda553e91020a23deed45")
	aesgcmTest(t, iv, key, plaintext, expected, tag)
}

func TestAES_GCM_NIST_gcmEncryptExtIV256_PTLen_256_Test_6(t *testing.T) {
	iv, _ := hex.DecodeString("a291484c3de8bec6b47f525f")
	key, _ := hex.DecodeString("37f39137416bafde6f75022a7a527cc593b6000a83ff51ec04871a0ff5360e4e")
	plaintext, _ := hex.DecodeString("fafd94cede8b5a0730394bec68a8e77dba288d6ccaa8e1563a81d6e7ccc7fc97")
	expected, _ := hex.DecodeString("44dc868006b21d49284016565ffb3979cc4271d967628bf7cdaf86db888e92e501a2b578aa2f41ec6379a44a31cc019c")
	tag, _ := hex.DecodeString("01a2b578aa2f41ec6379a44a31cc019c")
	aesgcmTest(t, iv, key, plaintext, expected, tag)
}

func TestAES_GCM_NIST_gcmEncryptExtIV256_PTLen_408_Test_8(t *testing.T) {
	iv, _ := hex.DecodeString("92f258071d79af3e63672285")
	key, _ := hex.DecodeString("595f259c55abe00ae07535ca5d9b09d6efb9f7e9abb64605c337acbd6b14fc7e")
	plaintext, _ := hex.DecodeString("a6fee33eb110a2d769bbc52b0f36969c287874f665681477a25fc4c48015c541fbe2394133ba490a34ee2dd67b898177849a91")
	expected, _ := hex.DecodeString("bbca4a9e09ae9690c0f6f8d405e53dccd666aa9c5fa13c8758bc30abe1ddd1bcce0d36a1eaaaaffef20cd3c5970b9673f8a65c26ccecb9976fd6ac9c2c0f372c52c821")
	tag, _ := hex.DecodeString("26ccecb9976fd6ac9c2c0f372c52c821")
	aesgcmTest(t, iv, key, plaintext, expected, tag)
}

func aesgcmTest(t *testing.T, iv, key, plaintext, expected, tag []byte) {
	cd := CipherData{
		Key: key,
		IV:  iv,
	}
	gcm, err := newAESGCM(cd)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	cipherdata := gcm.Encrypt(bytes.NewReader(plaintext))

	ciphertext, err := ioutil.ReadAll(cipherdata)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	// splitting tag and ciphertext
	etag := ciphertext[len(ciphertext)-16:]
	if !bytes.Equal(etag, tag) {
		t.Errorf("expected tags to be equivalent")
	}
	if !bytes.Equal(ciphertext, expected) {
		t.Errorf("expected ciphertext to be equivalent")
	}

	data := gcm.Decrypt(bytes.NewReader(ciphertext))
	text, err := ioutil.ReadAll(data)
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}
	if !bytes.Equal(plaintext, text) {
		t.Errorf("expected ciphertext to be equivalent")
	}
}
