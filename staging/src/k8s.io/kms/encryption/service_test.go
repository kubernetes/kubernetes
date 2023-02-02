/*
Copyright 2023 The Kubernetes Authors.

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

package encryption

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"k8s.io/kms/service"
)

var (
	errCreateKey         = errors.New("can't create key")
	errCreateTransformer = errors.New("can't create transformer")
	errCreateUID         = errors.New("can't create uid")
	errRemoteEncrypt     = errors.New("can't encrypt with remote kms")
	errRemoteDecrypt     = errors.New("can't decrypt with remote kms")
	errLocallyEncrypt    = errors.New("can't encrypt with local kms")
	errLocallyDecrypt    = errors.New("can't decrypt with local kms")
	errStoreDecrypt      = errors.New("can't decrypt with local store")
	errWrongValue        = errors.New("wrong value")
	errNotBeCalled       = errors.New("should not be called")

	keyID          = "id:123"
	genKey         = []byte("genkey:123")
	storeKey       = []byte("storekey:123")
	anonKey        = []byte("anonkey:123")
	plainKey       = []byte("plainkey:123")
	encryptKey     = []byte("enckey:123")
	decryptKey     = []byte("deckey:123")
	encryptMessage = []byte("encrypt_to_storage")
	decryptMessage = []byte("decrypt_from_storage")
)

func TestService(t *testing.T) {
	t.Parallel()

	initTestCases := []struct {
		name            string
		localKEKService func() (*LocalKEKService, error)
		err             error
	}{
		{
			name: "should fail if no remote service is given",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					nil, nil, nil, nil,
				)
			},
			err: ErrNoCipher,
		},
		{
			name: "should fail if you can't create a key",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{},
					nil,
					&createTransformer{
						key: func() ([]byte, error) {
							return nil, errCreateKey
						},
					},
					nil,
				)
			},
			err: errCreateKey,
		},
		{
			name: "should fail if you can't create a transformer",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{},
					nil,
					&createTransformer{
						key: genLocalKey,
						transformer: func(ctx context.Context, key []byte) (Transformer, error) {
							if !bytes.Equal(key, genKey) {
								return nil, errWrongValue
							}

							return nil, errCreateTransformer
						},
					},
					nil,
				)
			},
			err: errCreateTransformer,
		},
		{
			name: "should fail if you can't create a uid",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{},
					nil,
					&createTransformer{key: genLocalKey, transformer: makeEmptyTransformer},
					func() (string, error) {
						return "", errCreateUID
					},
				)
			},
			err: errCreateUID,
		},
		{
			name: "should fail if you can't encrypt remotely",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: func(ctx context.Context, uid string, key []byte) (*service.EncryptResponse, error) {
							return nil, errRemoteEncrypt
						},
					},
					nil,
					&createTransformer{key: genLocalKey, transformer: makeEmptyTransformer},
					genUID,
				)
			},
			err: errRemoteEncrypt,
		},
		{
			name: "should succeed initializing",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{encrypt: remoteEncryptDefault},
					makeEmptyStore(),
					&createTransformer{key: genLocalKey, transformer: makeEmptyTransformer},
					genUID,
				)
			},
		},
	}

	for _, tc := range initTestCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			localKEKService, err := tc.localKEKService()
			if err == tc.err {
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if localKEKService == nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	encryptTestCases := []struct {
		name            string
		localKEKService func() (*LocalKEKService, error)
		err             error
	}{
		{
			name: "should fail if local kek is not in cache and remote decrypt fails",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: remoteEncryptDefault,
						decrypt: func(ctx context.Context, uid string, decReq *service.DecryptRequest) ([]byte, error) {
							return nil, errRemoteDecrypt
						},
					},
					makeEmptyStore(),
					&createTransformer{
						key: genLocalKey,
						transformer: func(context.Context, []byte) (Transformer, error) {
							return &testTransformer{
								transformToStorage: func(ctx context.Context, data []byte, dataCtx Context) ([]byte, error) {
									return encryptMessage, nil
								},
							}, nil
						},
					},
					genUID,
				)
			},
			err: errRemoteDecrypt,
		},
		{
			name: "should fail if you can't encrypt locally",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{encrypt: remoteEncryptDefault, decrypt: remoteDecryptShouldNotBeCalled},
					&testStore{
						add: func([]byte, Transformer) {},
						get: func(encKey []byte) (Transformer, bool) {
							return &testTransformer{
								transformToStorage: func(ctx context.Context, data []byte, dataCtx Context) ([]byte, error) {
									return nil, errLocallyEncrypt
								},
							}, true
						},
					},
					&createTransformer{
						key: genLocalKey,
						transformer: func(context.Context, []byte) (Transformer, error) {
							return &testTransformer{}, nil
						},
					},
					genUID,
				)
			},
			err: errLocallyEncrypt,
		},
		{
			name: "should succeed encrypting with stored transformer",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{encrypt: remoteEncryptDefault, decrypt: remoteDecryptShouldNotBeCalled},
					&testStore{
						add: func([]byte, Transformer) {},
						get: func(encKey []byte) (Transformer, bool) {
							return &testTransformer{
								transformToStorage: func(ctx context.Context, data []byte, dataCtx Context) ([]byte, error) {
									return encryptMessage, nil
								},
							}, true
						},
					},
					&createTransformer{
						key: genLocalKey,
						transformer: func(context.Context, []byte) (Transformer, error) {
							return &testTransformer{}, nil
						},
					},
					genUID,
				)
			},
		},
		{
			name: "should succeed encrypting with stored transformer",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{encrypt: remoteEncryptDefault, decrypt: remoteDecryptShouldNotBeCalled},
					&testStore{
						add: func([]byte, Transformer) {},
						get: func(encKey []byte) (Transformer, bool) {
							return &testTransformer{
								transformToStorage: func(ctx context.Context, data []byte, dataCtx Context) ([]byte, error) {
									return encryptMessage, nil
								},
							}, true
						},
					},
					&createTransformer{
						key: genLocalKey,
						transformer: func(context.Context, []byte) (Transformer, error) {
							return &testTransformer{}, nil
						},
					},
					genUID,
				)
			},
		},
		{
			name: "should succeed encrypting with created transformer",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: remoteEncryptDefault,
						decrypt: func(ctx context.Context, uid string, decReq *service.DecryptRequest) ([]byte, error) {
							return encryptKey, nil
						},
					},
					&testStore{
						add: func([]byte, Transformer) {},
						get: func(encKey []byte) (Transformer, bool) {
							return nil, false
						},
					},
					&createTransformer{
						key: genLocalKey,
						transformer: func(context.Context, []byte) (Transformer, error) {
							return &testTransformer{
								transformToStorage: func(ctx context.Context, data []byte, dataCtx Context) ([]byte, error) {
									return encryptMessage, nil
								},
							}, nil
						},
					},
					genUID,
				)
			},
		},
	}

	for _, tc := range encryptTestCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			localKEKService, err := tc.localKEKService()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			encRes, err := localKEKService.Encrypt(context.Background(), "id:999", []byte("message"))
			if err == tc.err {
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !bytes.Equal(encryptMessage, encRes.Ciphertext) {
				t.Fatalf("unexpected ciphertext - want: %s, have: %s", encryptMessage, encRes.Ciphertext)
			}
		})
	}

	noAnnotationsDecReq := &service.DecryptRequest{
		Ciphertext:  encryptMessage,
		KeyID:       keyID,
		Annotations: map[string][]byte{},
	}

	anonDecReq := &service.DecryptRequest{
		Ciphertext: encryptMessage,
		KeyID:      keyID,
		Annotations: map[string][]byte{
			LocalKEKID: anonKey,
		},
	}

	decryptTestCases := []struct {
		name            string
		localKEKService func() (*LocalKEKService, error)
		decReq          *service.DecryptRequest
		err             error
	}{
		{
			name: "should fail decrypting without annotations, if remote decryption doesn't work",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: remoteEncryptDefault,
						decrypt: func(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
							return nil, errRemoteDecrypt
						},
					},
					makeEmptyStore(),
					&createTransformer{key: genLocalKey, transformer: makeEmptyTransformer},
					genUID,
				)
			},
			decReq: noAnnotationsDecReq,
			err:    errRemoteDecrypt,
		},
		{
			name: "should succeed decrypting without annotations, if remote decryption works",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: remoteEncryptDefault,
						decrypt: func(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
							return decryptMessage, nil
						},
					},
					makeEmptyStore(),
					&createTransformer{key: genLocalKey, transformer: makeEmptyTransformer},
					genUID,
				)
			},
			decReq: noAnnotationsDecReq,
		},
		{
			name: "should fail decrypting, if decrypting with remote KMS doesn't work for unknown key",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: remoteEncryptDefault,
						decrypt: func(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
							return nil, errRemoteDecrypt
						},
					},
					&testStore{
						add: func([]byte, Transformer) {},
						get: func(encKey []byte) (Transformer, bool) { return nil, false },
					},
					&createTransformer{
						key: genLocalKey,
						transformer: func(context.Context, []byte) (Transformer, error) {
							return &testTransformer{
								transformFromStorage: func(ctx context.Context, ct []byte, defaultCtx Context) ([]byte, bool, error) {
									return nil, false, errNotBeCalled
								},
							}, nil
						},
					},
					genUID,
				)
			},
			decReq: anonDecReq,
			err:    errRemoteDecrypt,
		},
		{
			name: "should fail decrypting, if creating a transformer fails with unknown key",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: remoteEncryptDefault,
						decrypt: func(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
							return plainKey, nil
						},
					},
					&testStore{
						add: func(key []byte, transformer Transformer) {},
						get: func(key []byte) (Transformer, bool) {
							return nil, false
						},
					},
					&createTransformer{
						key: genLocalKey,
						transformer: func(ctx context.Context, key []byte) (Transformer, error) {
							if bytes.Equal(key, plainKey) {
								return nil, errCreateTransformer
							}

							return &testTransformer{}, nil
						},
					},
					genUID,
				)
			},
			decReq: &service.DecryptRequest{
				Ciphertext: encryptMessage,
				KeyID:      keyID,
				Annotations: map[string][]byte{
					LocalKEKID: anonKey,
				},
			},
			err: errCreateTransformer,
		},
		{
			name: "should fail decrypting, if decryption fails with stored transformer",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: remoteEncryptDefault,
						decrypt: remoteDecryptShouldNotBeCalled,
					},
					&testStore{
						add: func(key []byte, transformer Transformer) {},
						get: func(key []byte) (Transformer, bool) {
							if !bytes.Equal(key, storeKey) {
								return nil, false
							}

							return &testTransformer{
								transformFromStorage: func(ctx context.Context, ct []byte, defaultCtx Context) ([]byte, bool, error) {
									return nil, false, errLocallyDecrypt
								},
							}, true
						},
					},
					&createTransformer{key: genLocalKey, transformer: makeEmptyTransformer},
					genUID,
				)
			},
			decReq: &service.DecryptRequest{
				Ciphertext: encryptMessage,
				KeyID:      keyID,
				Annotations: map[string][]byte{
					LocalKEKID: storeKey,
				},
			},
			err: errLocallyDecrypt,
		},
		{
			name: "should succeed decrypting, if decryption works with stored transformer",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: remoteEncryptDefault,
						decrypt: remoteDecryptShouldNotBeCalled,
					},
					&testStore{
						add: func(key []byte, transformer Transformer) {},
						get: func(key []byte) (Transformer, bool) {
							if !bytes.Equal(key, storeKey) {
								return nil, false
							}

							return &testTransformer{
								transformFromStorage: func(ctx context.Context, ct []byte, defaultCtx Context) ([]byte, bool, error) {
									return decryptMessage, false, nil
								},
							}, true
						},
					},
					&createTransformer{key: genLocalKey, transformer: makeEmptyTransformer},
					genUID,
				)
			},
			decReq: &service.DecryptRequest{
				Ciphertext: encryptMessage,
				KeyID:      keyID,
				Annotations: map[string][]byte{
					LocalKEKID: storeKey,
				},
			},
		},
		{
			name: "should succeed decrypting, with newly created transformer",
			localKEKService: func() (*LocalKEKService, error) {
				return NewLocalKEKService(
					context.Background(),
					&testService{
						encrypt: remoteEncryptDefault,
						decrypt: func(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
							return plainKey, nil
						},
					},
					makeEmptyStore(),
					&createTransformer{
						key: genLocalKey,
						transformer: func(ctx context.Context, key []byte) (Transformer, error) {
							return &testTransformer{
								transformFromStorage: func(ctx context.Context, ct []byte, defaultCtx Context) ([]byte, bool, error) {
									return decryptMessage, false, nil
								},
							}, nil
						},
					},
					genUID,
				)
			},
			decReq: anonDecReq,
		},
	}

	for _, tc := range decryptTestCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			localKEKService, err := tc.localKEKService()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			pt, err := localKEKService.Decrypt(context.Background(), tc.name, tc.decReq)
			if err == tc.err {
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !bytes.Equal(pt, decryptMessage) {
				t.Fatalf("unexpected plaintext - want: %s, have: %s", decryptMessage, pt)
			}
		})
	}
}

type testStore struct {
	add func([]byte, Transformer)
	get func([]byte) (Transformer, bool)
}

var _ Store = (*testStore)(nil)

func (t *testStore) Add(key []byte, transformer Transformer) {
	t.add(key, transformer)
}

func (t *testStore) Get(key []byte) (Transformer, bool) {
	return t.get(key)
}

type testTransformer struct {
	transformFromStorage func(context.Context, []byte, Context) ([]byte, bool, error)
	transformToStorage   func(context.Context, []byte, Context) ([]byte, error)
}

var _ Transformer = (*testTransformer)(nil)

func (t *testTransformer) TransformFromStorage(ctx context.Context, data []byte, context Context) ([]byte, bool, error) {
	return t.transformFromStorage(ctx, data, context)
}

func (t *testTransformer) TransformToStorage(ctx context.Context, data []byte, context Context) ([]byte, error) {
	return t.transformToStorage(ctx, data, context)
}

type createTransformer struct {
	transformer func(context.Context, []byte) (Transformer, error)
	key         func() ([]byte, error)
}

var _ CreateTransformer = (*createTransformer)(nil)

func (c *createTransformer) Transformer(ctx context.Context, key []byte) (Transformer, error) {
	return c.transformer(ctx, key)
}

func (c *createTransformer) Key() ([]byte, error) {
	return c.key()
}

type testService struct {
	decrypt func(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error)
	encrypt func(ctx context.Context, uid string, data []byte) (*service.EncryptResponse, error)
	status  func(ctx context.Context) (*service.StatusResponse, error)
}

var _ service.Service = (*testService)(nil)

func (s *testService) Decrypt(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
	return s.decrypt(ctx, uid, req)
}

func (s *testService) Encrypt(ctx context.Context, uid string, data []byte) (*service.EncryptResponse, error) {
	return s.encrypt(ctx, uid, data)
}

func (s *testService) Status(ctx context.Context) (*service.StatusResponse, error) {
	return s.status(ctx)
}

func genUID() (string, error) {
	return "id:001", nil
}

func genLocalKey() ([]byte, error) {
	return genKey, nil
}

func makeEmptyTransformer(context.Context, []byte) (Transformer, error) {
	return &testTransformer{}, nil
}

func makeEmptyStore() *testStore {
	return &testStore{
		add: func([]byte, Transformer) {},
		get: func(key []byte) (Transformer, bool) {
			return nil, false
		},
	}
}

func remoteEncryptDefault(ctx context.Context, uid string, key []byte) (*service.EncryptResponse, error) {
	return &service.EncryptResponse{
		Ciphertext:  encryptKey,
		KeyID:       keyID,
		Annotations: map[string][]byte{},
	}, nil
}

func remoteDecryptShouldNotBeCalled(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
	return nil, errNotBeCalled

}
