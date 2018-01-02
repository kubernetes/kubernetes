/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package units

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
)

type ByteSize int64

const (
	_  = iota
	KB = 1 << (10 * iota)
	MB
	GB
	TB
	PB
	EB
)

func (b ByteSize) String() string {
	switch {
	case b >= EB:
		return fmt.Sprintf("%.1fEB", float32(b)/EB)
	case b >= PB:
		return fmt.Sprintf("%.1fPB", float32(b)/PB)
	case b >= TB:
		return fmt.Sprintf("%.1fTB", float32(b)/TB)
	case b >= GB:
		return fmt.Sprintf("%.1fGB", float32(b)/GB)
	case b >= MB:
		return fmt.Sprintf("%.1fMB", float32(b)/MB)
	case b >= KB:
		return fmt.Sprintf("%.1fKB", float32(b)/KB)
	}
	return fmt.Sprintf("%dB", b)
}

var bytesRegexp = regexp.MustCompile(`^(?i)(\d+)([BKMGTPE]?)(ib|b)?$`)

func (b *ByteSize) Set(s string) error {
	m := bytesRegexp.FindStringSubmatch(s)
	if len(m) == 0 {
		return errors.New("invalid byte value")
	}

	i, err := strconv.ParseInt(m[1], 10, 64)
	if err != nil {
		return err
	}
	*b = ByteSize(i)

	switch m[2] {
	case "B", "b", "":
	case "K", "k":
		*b *= ByteSize(KB)
	case "M", "m":
		*b *= ByteSize(MB)
	case "G", "g":
		*b *= ByteSize(GB)
	case "T", "t":
		*b *= ByteSize(TB)
	case "P", "p":
		*b *= ByteSize(PB)
	case "E", "e":
		*b *= ByteSize(EB)
	default:
		return errors.New("invalid byte suffix")
	}

	return nil
}
