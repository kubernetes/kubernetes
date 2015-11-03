/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package sshkey

import (
  "crypto"
  "crypto/ecdsa"
  "crypto/elliptic"
  "crypto/rand"
  "crypto/rsa"
  "crypto/x509"
  "encoding/pem"
  "fmt"
  "io"
  "strconv"

  "golang.org/x/crypto/ssh"

  "k8s.io/kubernetes/pkg/api"
  client "k8s.io/kubernetes/pkg/client/unversioned"
  "k8s.io/kubernetes/pkg/secret"
)

const (
  // GeneratorName is the name of the generator
  GeneratorName = "kubernetes.io/sshkey"

  // KeySizeAnnotation is the name of the annotation for key size
  KeySizeAnnotation = GeneratorName + "-size"

  // KeyTypeAnnotation is the name of the annotation for key type
  KeyTypeAnnotation = GeneratorName + "-type"

  DefaultKeyType = "rsa"
)

var (
  defaultKeySize = map[string]int{
    "rsa":   2048,
    "ecdsa": 384,
  }

  curves = map[int]elliptic.Curve{
    256: elliptic.P256(),
    384: elliptic.P384(),
    521: elliptic.P521(),
  }
)

func init() {
  secret.RegisterPlugin(GeneratorName, func(client client.Interface, config io.Reader) (secret.Interface, error) {
    generator := New(client)
    return generator, nil
  })
}

var _ = secret.Interface(&sshkey{})

// New returns an secret.Interface implementation which generates passwords.
func New(cl client.Interface) *sshkey {
  return &sshkey{}
}

type sshkey struct{}

type keyGenFunc func(io.Reader, int)

func (s *sshkey) GenerateValues(req *api.GenerateSecretRequest) (map[string][]byte, error) {
  keyType := DefaultKeyType
  keyTypeAnnotation := req.Annotations[KeyTypeAnnotation]
  if keyTypeAnnotation != "" {
    keyType = keyTypeAnnotation
  }

  keySize := defaultKeySize[keyType]
  keySizeAnnotation := req.Annotations[KeySizeAnnotation]
  if keySizeAnnotation != "" {
    ks, err := strconv.Atoi(keySizeAnnotation)
    if err != nil {
      return nil, fmt.Errorf("invalid key size: %s", keySizeAnnotation)
    }
    keySize = ks
  }

  var (
    privBytes  []byte
    publicKey  crypto.PublicKey
    typePrefix string
  )
  switch keyType {
  case "rsa":
    priv, err := rsa.GenerateKey(rand.Reader, keySize)
    if err != nil {
      return nil, err
    }
    privBytes = x509.MarshalPKCS1PrivateKey(priv)
    publicKey = &priv.PublicKey
    typePrefix = "RSA"
  case "ecdsa":
    c, found := curves[keySize]
    if !found {
      return nil, fmt.Errorf("unknown ecdsa key size %d", keySize)
    }
    priv, err := ecdsa.GenerateKey(c, rand.Reader)
    if err != nil {
      return nil, err
    }
    p, err := x509.MarshalECPrivateKey(priv)
    if err != nil {
      return nil, err
    }
    privBytes = p
    publicKey = &priv.PublicKey
    typePrefix = "EC"
  default:
    return nil, fmt.Errorf("unknown key type: %s", keyType)
  }

  privBlk := pem.Block{
    Type:    typePrefix + " PRIVATE KEY",
    Headers: nil,
    Bytes:   privBytes,
  }

  privPem := pem.EncodeToMemory(&privBlk)

  sshPublicKey, err := ssh.NewPublicKey(publicKey)
  if err != nil {
    return nil, err
  }
  pubBytes := ssh.MarshalAuthorizedKey(sshPublicKey)

  return map[string][]byte{
    "private": privPem,
    "public":  pubBytes,
  }, nil
}
