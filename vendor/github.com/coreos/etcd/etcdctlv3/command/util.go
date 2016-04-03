// Copyright 2015 CoreOS, Inc.
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

package command

import (
	"encoding/hex"
	"fmt"
	"regexp"

	pb "github.com/coreos/etcd/storage/storagepb"
)

func printKV(isHex bool, kv *pb.KeyValue) {
	k, v := string(kv.Key), string(kv.Value)
	if isHex {
		k = addHexPrefix(hex.EncodeToString(kv.Key))
		v = addHexPrefix(hex.EncodeToString(kv.Value))
	}
	fmt.Println(k)
	fmt.Println(v)
}

func addHexPrefix(s string) string {
	ns := make([]byte, len(s)*2)
	for i := 0; i < len(s); i += 2 {
		ns[i*2] = '\\'
		ns[i*2+1] = 'x'
		ns[i*2+2] = s[i]
		ns[i*2+3] = s[i+1]
	}
	return string(ns)
}

func argify(s string) []string {
	r := regexp.MustCompile("'.+'|\".+\"|\\S+")
	return r.FindAllString(s, -1)
}
