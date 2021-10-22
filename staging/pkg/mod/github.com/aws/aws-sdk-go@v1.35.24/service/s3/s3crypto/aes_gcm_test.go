// +build go1.9

package s3crypto

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

// AES GCM
func TestAES_GCM_NIST_gcmEncryptExtIV256_PTLen_128_Test_0(t *testing.T) {
	iv, _ := hex.DecodeString("0d18e06c7c725ac9e362e1ce")
	key, _ := hex.DecodeString("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
	plaintext, _ := hex.DecodeString("2db5168e932556f8089a0622981d017d")
	expected, _ := hex.DecodeString("fa4362189661d163fcd6a56d8bf0405a")
	tag, _ := hex.DecodeString("d636ac1bbedd5cc3ee727dc2ab4a9489")
	aesgcmTest(t, iv, key, plaintext, expected, tag)
}

func TestAES_GCM_NIST_gcmEncryptExtIV256_PTLen_104_Test_3(t *testing.T) {
	iv, _ := hex.DecodeString("4742357c335913153ff0eb0f")
	key, _ := hex.DecodeString("e5a0eb92cc2b064e1bc80891faf1fab5e9a17a9c3a984e25416720e30e6c2b21")
	plaintext, _ := hex.DecodeString("8499893e16b0ba8b007d54665a")
	expected, _ := hex.DecodeString("eb8e6175f1fe38eb1acf95fd51")
	tag, _ := hex.DecodeString("88a8b74bb74fda553e91020a23deed45")
	aesgcmTest(t, iv, key, plaintext, expected, tag)
}

func TestAES_GCM_NIST_gcmEncryptExtIV256_PTLen_256_Test_6(t *testing.T) {
	iv, _ := hex.DecodeString("a291484c3de8bec6b47f525f")
	key, _ := hex.DecodeString("37f39137416bafde6f75022a7a527cc593b6000a83ff51ec04871a0ff5360e4e")
	plaintext, _ := hex.DecodeString("fafd94cede8b5a0730394bec68a8e77dba288d6ccaa8e1563a81d6e7ccc7fc97")
	expected, _ := hex.DecodeString("44dc868006b21d49284016565ffb3979cc4271d967628bf7cdaf86db888e92e5")
	tag, _ := hex.DecodeString("01a2b578aa2f41ec6379a44a31cc019c")
	aesgcmTest(t, iv, key, plaintext, expected, tag)
}

func TestAES_GCM_NIST_gcmEncryptExtIV256_PTLen_408_Test_8(t *testing.T) {
	iv, _ := hex.DecodeString("92f258071d79af3e63672285")
	key, _ := hex.DecodeString("595f259c55abe00ae07535ca5d9b09d6efb9f7e9abb64605c337acbd6b14fc7e")
	plaintext, _ := hex.DecodeString("a6fee33eb110a2d769bbc52b0f36969c287874f665681477a25fc4c48015c541fbe2394133ba490a34ee2dd67b898177849a91")
	expected, _ := hex.DecodeString("bbca4a9e09ae9690c0f6f8d405e53dccd666aa9c5fa13c8758bc30abe1ddd1bcce0d36a1eaaaaffef20cd3c5970b9673f8a65c")
	tag, _ := hex.DecodeString("26ccecb9976fd6ac9c2c0f372c52c821")
	aesgcmTest(t, iv, key, plaintext, expected, tag)
}

type KAT struct {
	IV         string `json:"iv"`
	Key        string `json:"key"`
	Plaintext  string `json:"pt"`
	AAD        string `json:"aad"`
	CipherText string `json:"ct"`
	Tag        string `json:"tag"`
}

func TestAES_GCM_KATS(t *testing.T) {
	fileContents, err := ioutil.ReadFile("testdata/aes_gcm.json")
	if err != nil {
		t.Fatalf("failed to read KAT file: %v", err)
	}

	var kats []KAT
	err = json.Unmarshal(fileContents, &kats)
	if err != nil {
		t.Fatalf("failed to unmarshal KAT json file: %v", err)
	}

	for i, kat := range kats {
		t.Run(fmt.Sprintf("Case%d", i), func(t *testing.T) {
			if len(kat.AAD) > 0 {
				t.Skip("Skipping... SDK implementation does not expose additional authenticated data")
			}
			iv, err := hex.DecodeString(kat.IV)
			if err != nil {
				t.Fatalf("failed to decode iv: %v", err)
			}
			key, err := hex.DecodeString(kat.Key)
			if err != nil {
				t.Fatalf("failed to decode key: %v", err)
			}
			plaintext, err := hex.DecodeString(kat.Plaintext)
			if err != nil {
				t.Fatalf("failed to decode plaintext: %v", err)
			}
			ciphertext, err := hex.DecodeString(kat.CipherText)
			if err != nil {
				t.Fatalf("failed to decode ciphertext: %v", err)
			}
			tag, err := hex.DecodeString(kat.Tag)
			if err != nil {
				t.Fatalf("failed to decode tag: %v", err)
			}
			aesgcmTest(t, iv, key, plaintext, ciphertext, tag)
		})
	}
}

func TestGCMEncryptReader_SourceError(t *testing.T) {
	gcm := &gcmEncryptReader{
		encrypter: &mockCipherAEAD{},
		src:       &mockSourceReader{err: fmt.Errorf("test read error")},
	}

	b := make([]byte, 10)
	n, err := gcm.Read(b)
	if err == nil {
		t.Fatalf("expected error, but got nil")
	} else if err != nil && !strings.Contains(err.Error(), "test read error") {
		t.Fatalf("expected source read error, but got %v", err)
	}

	if n != 0 {
		t.Errorf("expected number of read bytes to be zero, but got %v", n)
	}
}

func TestGCMDecryptReader_SourceError(t *testing.T) {
	gcm := &gcmDecryptReader{
		decrypter: &mockCipherAEAD{},
		src:       &mockSourceReader{err: fmt.Errorf("test read error")},
	}

	b := make([]byte, 10)
	n, err := gcm.Read(b)
	if err == nil {
		t.Fatalf("expected error, but got nil")
	} else if err != nil && !strings.Contains(err.Error(), "test read error") {
		t.Fatalf("expected source read error, but got %v", err)
	}

	if n != 0 {
		t.Errorf("expected number of read bytes to be zero, but got %v", n)
	}
}

func TestGCMDecryptReader_DecrypterOpenError(t *testing.T) {
	gcm := &gcmDecryptReader{
		decrypter: &mockCipherAEAD{openError: fmt.Errorf("test open error")},
		src:       &mockSourceReader{err: io.EOF},
	}

	b := make([]byte, 10)
	n, err := gcm.Read(b)
	if err == nil {
		t.Fatalf("expected error, but got nil")
	} else if err != nil && !strings.Contains(err.Error(), "test open error") {
		t.Fatalf("expected source read error, but got %v", err)
	}

	if n != 0 {
		t.Errorf("expected number of read bytes to be zero, but got %v", n)
	}
}

func aesgcmTest(t *testing.T, iv, key, plaintext, expected, tag []byte) {
	t.Helper()
	const gcmTagSize = 16
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
	etag := ciphertext[len(ciphertext)-gcmTagSize:]
	if !bytes.Equal(etag, tag) {
		t.Errorf("expected tags to be equivalent")
	}
	if !bytes.Equal(ciphertext[:len(ciphertext)-gcmTagSize], expected) {
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

type mockSourceReader struct {
	n   int
	err error
}

func (b mockSourceReader) Read(p []byte) (n int, err error) {
	return b.n, b.err
}

type mockCipherAEAD struct {
	seal      []byte
	openError error
}

func (m mockCipherAEAD) NonceSize() int {
	panic("implement me")
}

func (m mockCipherAEAD) Overhead() int {
	panic("implement me")
}

func (m mockCipherAEAD) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	return m.seal
}

func (m mockCipherAEAD) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	return []byte("mocked decrypt"), m.openError
}
