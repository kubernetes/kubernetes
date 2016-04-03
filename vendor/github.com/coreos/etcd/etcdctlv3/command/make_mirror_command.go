// Copyright 2016 CoreOS, Inc.
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
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/mirror"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/coreos/etcd/storage/storagepb"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
)

var (
	mmcert   string
	mmkey    string
	mmcacert string
	mmprefix string
)

// NewMakeMirrorCommand returns the cobra command for "makeMirror".
func NewMakeMirrorCommand() *cobra.Command {
	c := &cobra.Command{
		Use:   "make-mirror [options] <destination>",
		Short: "make-mirror makes a mirror at the destination etcd cluster",
		Run:   makeMirrorCommandFunc,
	}

	c.Flags().StringVar(&mmprefix, "prefix", "", "the key-value prefix to mirror")
	// TODO: add dest-prefix to mirror a prefix to a different prefix in the destionation cluster?
	c.Flags().StringVar(&mmcert, "dest-cert", "", "identify secure client using this TLS certificate file for the destination cluster")
	c.Flags().StringVar(&mmkey, "dest-key", "", "identify secure client using this TLS key file")
	c.Flags().StringVar(&mmcacert, "dest-cacert", "", "verify certificates of TLS enabled secure servers using this CA bundle")

	return c
}

func makeMirrorCommandFunc(cmd *cobra.Command, args []string) {
	if len(args) != 1 {
		ExitWithError(ExitBadArgs, errors.New("make-mirror takes one destination arguement."))
	}

	dc := mustClient([]string{args[0]}, mmcert, mmkey, mmcacert)
	c := mustClientFromCmd(cmd)

	err := makeMirror(context.TODO(), c, dc)
	ExitWithError(ExitError, err)
}

func makeMirror(ctx context.Context, c *clientv3.Client, dc *clientv3.Client) error {
	total := int64(0)

	go func() {
		for {
			time.Sleep(30 * time.Second)
			fmt.Println(atomic.LoadInt64(&total))
		}
	}()

	// TODO: remove the prefix of the destination cluster?
	s := mirror.NewSyncer(c, mmprefix, 0)

	rc, errc := s.SyncBase(ctx)

	for r := range rc {
		for _, kv := range r.Kvs {
			_, err := dc.Put(ctx, string(kv.Key), string(kv.Value))
			if err != nil {
				return err
			}
			atomic.AddInt64(&total, 1)
		}
	}

	err := <-errc
	if err != nil {
		return err
	}

	wc := s.SyncUpdates(ctx)

	for wr := range wc {
		if wr.CompactRevision != 0 {
			return rpctypes.ErrCompacted
		}

		var rev int64
		ops := []clientv3.Op{}

		for _, ev := range wr.Events {
			nrev := ev.Kv.ModRevision
			if rev != 0 && nrev > rev {
				_, err := dc.Txn(ctx).Then(ops...).Commit()
				if err != nil {
					return err
				}
				ops = []clientv3.Op{}
			}
			switch ev.Type {
			case storagepb.PUT:
				ops = append(ops, clientv3.OpPut(string(ev.Kv.Key), string(ev.Kv.Value)))
				atomic.AddInt64(&total, 1)
			case storagepb.DELETE, storagepb.EXPIRE:
				ops = append(ops, clientv3.OpDelete(string(ev.Kv.Key)))
				atomic.AddInt64(&total, 1)
			default:
				panic("unexpected event type")
			}
		}

		if len(ops) != 0 {
			_, err := dc.Txn(ctx).Then(ops...).Commit()
			if err != nil {
				return err
			}
		}
	}

	return nil
}
