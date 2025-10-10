//go:build !go1.24

/*-
 * Copyright 2014 Square Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package jose

import (
	"hash"

	"golang.org/x/crypto/pbkdf2"
)

func pbkdf2Key(h func() hash.Hash, password string, salt []byte, iter, keyLen int) ([]byte, error) {
	return pbkdf2.Key([]byte(password), salt, iter, keyLen, h), nil
}
