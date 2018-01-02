// Copyright 2015 The etcd Authors
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

	pb "github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

func printKV(isHex bool, valueOnly bool, kv *pb.KeyValue) {
	k, v := string(kv.Key), string(kv.Value)
	if isHex {
		k = addHexPrefix(hex.EncodeToString(kv.Key))
		v = addHexPrefix(hex.EncodeToString(kv.Value))
	}
	if !valueOnly {
		fmt.Println(k)
	}
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
	r := regexp.MustCompile(`"(?:[^"\\]|\\.)*"|'[^']*'|[^'"\s]\S*[^'"\s]?`)
	args := r.FindAllString(s, -1)
	for i := range args {
		if len(args[i]) == 0 {
			continue
		}
		if args[i][0] == '\'' {
			// 'single-quoted string'
			args[i] = args[i][1 : len(args)-1]
		} else if args[i][0] == '"' {
			// "double quoted string"
			if _, err := fmt.Sscanf(args[i], "%q", &args[i]); err != nil {
				ExitWithError(ExitInvalidInput, err)
			}
		}
	}
	return args
}

func commandCtx(cmd *cobra.Command) (context.Context, context.CancelFunc) {
	timeOut, err := cmd.Flags().GetDuration("command-timeout")
	if err != nil {
		ExitWithError(ExitError, err)
	}
	return context.WithTimeout(context.Background(), timeOut)
}
