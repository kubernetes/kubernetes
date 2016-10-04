/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package aes

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"io"

	gcfg "gopkg.in/gcfg.v1"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/storage/encryptionprovider"
)

// ProviderName is the name of this encryption provider.
const ProviderName = "aes"

type EncryptionProvider struct {
	Config EncryptionConfig
	Block  cipher.Block
}

// EncryptionConfig stores the config file passed in
type EncryptionConfig struct {
	Global struct {
		KeyText string
	}
}

func newAESEncryption(config io.Reader) (*EncryptionProvider, error) {
	glog.Infof("Build AES Encryption provider")

	eConfig, _ := readAESEncryptionConfig(config)

	// Create the aes encryption algorithm
	block, err := aes.NewCipher([]byte(eConfig.Global.KeyText))
	if err != nil {
		fmt.Printf("Error: NewCipher(%d bytes) = %s", len(eConfig.Global.KeyText), err)
	}

	return &EncryptionProvider{Config: *eConfig, Block: block}, nil
}

func readAESEncryptionConfig(config io.Reader) (*EncryptionConfig, error) {
	var cfg EncryptionConfig
	var err error

	if config != nil {
		err = gcfg.ReadInto(&cfg, config)
		if err != nil {
			fmt.Errorf("readAESEncryptionConfig: ", err)
			return nil, err
		}
	}

	if cfg.Global.KeyText == "" {
		return nil, fmt.Errorf("no encryption key specified in configuration file")
	}

	return &cfg, nil
}

func (e *EncryptionProvider) Encrypt(obj string) (string, error) {

	glog.Error("STEVE: encrypt!")
	plaintext := []byte(obj)

	block, err := aes.NewCipher([]byte(e.Config.Global.KeyText))
	if err != nil {
		panic(err)
	}

	ciphertext := make([]byte, aes.BlockSize+len(plaintext))

	iv := ciphertext[:aes.BlockSize]
	if _, err := io.ReadFull(rand.Reader, iv); err != nil {
		return "", err
	}

	stream := cipher.NewCFBEncrypter(block, iv)
	stream.XORKeyStream(ciphertext[aes.BlockSize:], plaintext)

	// convert to base64
	encryptedText := base64.URLEncoding.EncodeToString(ciphertext)

	return encryptedText, nil
}

func (e *EncryptionProvider) Decrypt(obj string) (string, error) {

	glog.Error("STEVE: decrypt!")

	ciphertext, _ := base64.URLEncoding.DecodeString(obj)

	block, err := aes.NewCipher([]byte(e.Config.Global.KeyText))
	if err != nil {
		panic(err)
	}

	// The IV needs to be unique, but not secure. Therefore it's common to
	// include it at the beginning of the ciphertext.
	if len(ciphertext) < aes.BlockSize {
		return "", fmt.Errorf("ciphertext too short")
	}
	iv := ciphertext[:aes.BlockSize]
	ciphertext = ciphertext[aes.BlockSize:]

	stream := cipher.NewCFBDecrypter(block, iv)

	// XORKeyStream can work in-place if the two arguments are the same.
	stream.XORKeyStream(ciphertext, ciphertext)

	return fmt.Sprintf("%s", ciphertext), nil
}

func init() {
	encryptionprovider.RegisterEncryptionProvider(ProviderName, func(config io.Reader) (encryptionprovider.Interface, error) {
		return newAESEncryption(config)
	})
}
