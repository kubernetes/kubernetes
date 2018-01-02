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

package e2e

import (
	"strings"
	"testing"
)

func TestCtlV3Watch(t *testing.T)          { testCtl(t, watchTest) }
func TestCtlV3WatchNoTLS(t *testing.T)     { testCtl(t, watchTest, withCfg(configNoTLS)) }
func TestCtlV3WatchClientTLS(t *testing.T) { testCtl(t, watchTest, withCfg(configClientTLS)) }
func TestCtlV3WatchPeerTLS(t *testing.T)   { testCtl(t, watchTest, withCfg(configPeerTLS)) }
func TestCtlV3WatchTimeout(t *testing.T)   { testCtl(t, watchTest, withDialTimeout(0)) }

func TestCtlV3WatchInteractive(t *testing.T) {
	testCtl(t, watchTest, withInteractive())
}
func TestCtlV3WatchInteractiveNoTLS(t *testing.T) {
	testCtl(t, watchTest, withInteractive(), withCfg(configNoTLS))
}
func TestCtlV3WatchInteractiveClientTLS(t *testing.T) {
	testCtl(t, watchTest, withInteractive(), withCfg(configClientTLS))
}
func TestCtlV3WatchInteractivePeerTLS(t *testing.T) {
	testCtl(t, watchTest, withInteractive(), withCfg(configPeerTLS))
}

func watchTest(cx ctlCtx) {
	tests := []struct {
		puts []kv
		args []string

		wkv []kv
	}{
		{ // watch 1 key
			[]kv{{"sample", "value"}},
			[]string{"sample", "--rev", "1"},
			[]kv{{"sample", "value"}},
		},
		{ // watch 3 keys by prefix
			[]kv{{"key1", "val1"}, {"key2", "val2"}, {"key3", "val3"}},
			[]string{"key", "--rev", "1", "--prefix"},
			[]kv{{"key1", "val1"}, {"key2", "val2"}, {"key3", "val3"}},
		},
		{ // watch by revision
			[]kv{{"etcd", "revision_1"}, {"etcd", "revision_2"}, {"etcd", "revision_3"}},
			[]string{"etcd", "--rev", "2"},
			[]kv{{"etcd", "revision_2"}, {"etcd", "revision_3"}},
		},
		{ // watch 3 keys by range
			[]kv{{"key1", "val1"}, {"key3", "val3"}, {"key2", "val2"}},
			[]string{"key", "key3", "--rev", "1"},
			[]kv{{"key1", "val1"}, {"key2", "val2"}},
		},
	}

	for i, tt := range tests {
		donec := make(chan struct{})
		go func(i int, puts []kv) {
			for j := range puts {
				if err := ctlV3Put(cx, puts[j].key, puts[j].val, ""); err != nil {
					cx.t.Fatalf("watchTest #%d-%d: ctlV3Put error (%v)", i, j, err)
				}
			}
			close(donec)
		}(i, tt.puts)
		if err := ctlV3Watch(cx, tt.args, tt.wkv...); err != nil {
			if cx.dialTimeout > 0 && !isGRPCTimedout(err) {
				cx.t.Errorf("watchTest #%d: ctlV3Watch error (%v)", i, err)
			}
		}
		<-donec
	}
}

func ctlV3Watch(cx ctlCtx, args []string, kvs ...kv) error {
	cmdArgs := append(cx.PrefixArgs(), "watch")
	if cx.interactive {
		cmdArgs = append(cmdArgs, "--interactive")
	} else {
		cmdArgs = append(cmdArgs, args...)
	}

	proc, err := spawnCmd(cmdArgs)
	if err != nil {
		return err
	}

	if cx.interactive {
		wl := strings.Join(append([]string{"watch"}, args...), " ") + "\r"
		if err = proc.Send(wl); err != nil {
			return err
		}
	}

	for _, elem := range kvs {
		if _, err = proc.Expect(elem.key); err != nil {
			return err
		}
		if _, err = proc.Expect(elem.val); err != nil {
			return err
		}
	}
	return proc.Stop()
}
