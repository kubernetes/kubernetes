/*
Copyright 2015 The Kubernetes Authors.

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

package zk

import (
	"fmt"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/samuel/go-zookeeper/zk"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/components/framework/frameworkid"
)

const RPC_TIMEOUT = time.Second * 5

type storage struct {
	frameworkid.LookupFunc
	frameworkid.StoreFunc
	frameworkid.RemoveFunc
}

func Store(zkurl, frameworkName string) frameworkid.Storage {
	// TODO(jdef) validate Config
	zkServers, zkChroot, parseErr := parseZk(zkurl)
	withConnection := func(ctx context.Context, f func(c *zk.Conn) error) error {
		if parseErr != nil {
			return parseErr
		}
		timeout, err := timeout(ctx)
		if err != nil {
			return err
		}
		c, _, err := zk.Connect(zkServers, timeout)
		if err != nil {
			return err
		}
		defer c.Close()
		return f(c)
	}
	return &storage{
		LookupFunc: func(ctx context.Context) (rawData string, lookupErr error) {
			lookupErr = withConnection(ctx, func(c *zk.Conn) error {
				data, _, err := c.Get(path.Join(zkChroot, frameworkName))
				if err == nil {
					rawData = string(data)
				} else if err != zk.ErrNoNode {
					return err
				}
				return nil
			})
			return
		},
		RemoveFunc: func(ctx context.Context) error {
			return withConnection(ctx, func(c *zk.Conn) error {
				err := c.Delete(path.Join(zkChroot, frameworkName), -1)
				if err != zk.ErrNoNode {
					return err
				}
				return nil
			})
		},
		StoreFunc: func(ctx context.Context, id string) error {
			return withConnection(ctx, func(c *zk.Conn) error {
				// attempt to create the path
				_, err := c.Create(
					zkChroot,
					[]byte(""),
					0,
					zk.WorldACL(zk.PermAll),
				)
				if err != nil && err != zk.ErrNodeExists {
					return err
				}
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}
				// attempt to write framework ID to <path> / <frameworkName>
				fpath := path.Join(zkChroot, frameworkName)
				_, err = c.Create(fpath, []byte(id), 0, zk.WorldACL(zk.PermAll))
				if err != nil && err == zk.ErrNodeExists {
					select {
					case <-ctx.Done():
						return ctx.Err()
					default:
					}
					// cross-check value
					data, _, err := c.Get(fpath)
					if err != nil {
						return err
					}
					if string(data) != id {
						return frameworkid.ErrMismatch
					}
					return nil
				}
				return err
			})
		},
	}
}

func parseZk(zkurl string) ([]string, string, error) {
	u, err := url.Parse(zkurl)
	if err != nil {
		return nil, "", fmt.Errorf("bad zk url: %v", err)
	}
	if u.Scheme != "zk" {
		return nil, "", fmt.Errorf("invalid url scheme for zk url: '%v'", u.Scheme)
	}
	return strings.Split(u.Host, ","), u.Path, nil
}

func timeout(ctx context.Context) (time.Duration, error) {
	deadline, ok := ctx.Deadline()
	if !ok {
		// no deadline set
		return RPC_TIMEOUT, nil
	}
	if now := time.Now(); now.Before(deadline) {
		d := deadline.Sub(now)
		if d > RPC_TIMEOUT {
			// deadline is too far out, use our built-in
			return RPC_TIMEOUT, nil
		}
		return d, nil
	}

	// deadline has expired..
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
		// this should never happen because Done() should be closed
		// according to the contract of context. but we have this here
		// just in case.
		return 0, context.DeadlineExceeded
	}
}
