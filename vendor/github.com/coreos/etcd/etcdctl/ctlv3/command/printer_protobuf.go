// Copyright 2016 The etcd Authors
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
	"fmt"
	"os"

	v3 "github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	mvccpb "github.com/coreos/etcd/mvcc/mvccpb"
)

type pbPrinter struct{ printer }

type pbMarshal interface {
	Marshal() ([]byte, error)
}

func newPBPrinter() printer {
	return &pbPrinter{
		&printerRPC{newPrinterUnsupported("protobuf"), printPB},
	}
}

func (p *pbPrinter) Watch(r v3.WatchResponse) {
	evs := make([]*mvccpb.Event, len(r.Events))
	for i, ev := range r.Events {
		evs[i] = (*mvccpb.Event)(ev)
	}
	wr := pb.WatchResponse{
		Header:          &r.Header,
		Events:          evs,
		CompactRevision: r.CompactRevision,
		Canceled:        r.Canceled,
		Created:         r.Created,
	}
	printPB(&wr)
}

func printPB(v interface{}) {
	m, ok := v.(pbMarshal)
	if !ok {
		ExitWithError(ExitBadFeature, fmt.Errorf("marshal unsupported for type %T (%v)", v, v))
	}
	b, err := m.Marshal()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		return
	}
	fmt.Printf(string(b))
}
