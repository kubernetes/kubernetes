/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package toolbox

import (
	"errors"

	"github.com/vmware/vmw-guestinfo/message"
	"github.com/vmware/vmw-guestinfo/vmcheck"
)

const (
	rpciProtocol uint32 = 0x49435052
	tcloProtocol uint32 = 0x4f4c4354
)

var (
	ErrNotVirtualWorld = errors.New("not in a virtual world")
)

type backdoorChannel struct {
	protocol uint32

	*message.Channel
}

func (b *backdoorChannel) Start() error {
	if !vmcheck.IsVirtualCPU() {
		return ErrNotVirtualWorld
	}

	channel, err := message.NewChannel(b.protocol)
	if err != nil {
		return err
	}

	b.Channel = channel

	return nil
}

func (b *backdoorChannel) Stop() error {
	if b.Channel == nil {
		return nil
	}

	err := b.Channel.Close()

	b.Channel = nil

	return err
}

// NewBackdoorChannelOut creates a Channel for use with the RPCI protocol
func NewBackdoorChannelOut() Channel {
	return &backdoorChannel{
		protocol: rpciProtocol,
	}
}

// NewBackdoorChannelIn creates a Channel for use with the TCLO protocol
func NewBackdoorChannelIn() Channel {
	return &backdoorChannel{
		protocol: tcloProtocol,
	}
}
