/*
Copyright 2018 The Kubernetes Authors.

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

package etcd

// Obfuscator represents a symmetric encryption algorithm. he value of the output of this
// function should be pseudorandom, but the purpose isn't to securely protect data, only to make it
// very clear that the only valid operation for a client to perform on two resource versions is to
// check if they are equal.
type Obfuscator interface {
	// Encode takes a key and a resource version and performs a symmetric encryption
	// algorithm on it.
	Encode(key string, etcdResourceVersion uint64) uint64
	// Encode takes a key and an encoded resource version and reverses the encryption
	// algorithm. The value of v == Decode(key, Encode(key, v)) should always be true.
	Decode(key string, clientResourceVersion uint64) uint64
}
