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

package service

import "context"

// Service allows encrypting and decrypting data using an external Key Management Service.
type Service interface {
	// Decrypt a given bytearray to obtain the original data as bytes.
	Decrypt(ctx context.Context, uid string, req *DecryptRequest) ([]byte, error)
	// Encrypt bytes to a ciphertext.
	Encrypt(ctx context.Context, uid string, data []byte) (*EncryptResponse, error)
	// Status returns the status of the KMS.
	Status(ctx context.Context) (*StatusResponse, error)
}

// EncryptResponse is the response from the Envelope service when encrypting data.
type EncryptResponse struct {
	Ciphertext  []byte
	KeyID       string
	Annotations map[string][]byte
}

// DecryptRequest is the request to the Envelope service when decrypting data.
type DecryptRequest struct {
	Ciphertext  []byte
	KeyID       string
	Annotations map[string][]byte
}

// StatusResponse is the response from the Envelope service when getting the status of the service.
type StatusResponse struct {
	Version string
	Healthz string
	KeyID   string
}
