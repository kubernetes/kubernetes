/*
Copyright 2017 The Kubernetes Authors.

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

package value

type KmsStorage interface {
	Setup() error
	GetAllDEKs() (map[string]string, error)
	StoreNewDEKs(newDEKs map[string]string) error
}

type KmsService interface {
	Decrypt(data string) ([]byte, error)
	Encrypt(data []byte) (string, error)

	GetReadingTransformer(keyname string) (Transformer, error)
	GetWritingTransformer() (Transformer, error)

	Rotate(rotateIfNotEmpty bool) error
	Refresh() error
}
