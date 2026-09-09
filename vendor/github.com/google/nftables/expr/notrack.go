// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package expr

import (
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

type Notrack struct{}

func (e *Notrack) marshal(fam byte) ([]byte, error) {
	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("notrack\x00")},
	})
}

func (e *Notrack) marshalData(fam byte) ([]byte, error) {
	return []byte("notrack\x00"), nil
}

func (e *Notrack) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)

	if err != nil {
		return err
	}

	return ad.Err()
}
