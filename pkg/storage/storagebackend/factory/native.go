/*
Copyright 2016 The Kubernetes Authors.

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

package factory

import (
	"fmt"
	"github.com/golang/glog"
	"google.golang.org/grpc"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/native"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
	"net/url"
	"sync"
	"time"
)

var initMutex sync.Mutex
var client native.StorageServiceClient
var embeddedServer *native.Server

func newNativeStorage(c storagebackend.Config, embedded bool) (storage.Interface, DestroyFunc, error) {
	destroyFunc := func() {
		// TODO: what is the behaviour of this currently?
		glog.Infof("native destroy function called")
	}

	// TODO: yuk
	initMutex.Lock()
	defer initMutex.Unlock()
	if client == nil {
		var grpcServerUrlString string
		if embedded {
			options := &native.ServerOptions{}
			options.InitDefaults()

			if embeddedServer == nil {
				embeddedServer = native.NewServer(options)
				go func() {
					err := embeddedServer.Run()
					if err != nil {
						glog.Fatalf("embedded state server exited unexpectedly: %v", err)
					}
				}()

				for {
					if embeddedServer.IsStarted() && embeddedServer.IsLeader() {
						break
					}
					time.Sleep(100 * time.Millisecond)
				}
			}

			grpcServerUrlString = "http://" + options.ClientBind
		} else {
			if len(c.ServerList) == 0 {
				return nil, nil, fmt.Errorf("no servers provided")
			}

			if len(c.ServerList) > 1 {
				glog.Warningf("ignoring additional state servers: %s", c.ServerList)
			}
			grpcServerUrlString = c.ServerList[0]
		}

		grpcServerUrl, err := url.Parse(grpcServerUrlString)
		if err != nil {
			return nil, nil, fmt.Errorf("cannot parse server url: %q", grpcServerUrlString)
		}
		var opts []grpc.DialOption
		if grpcServerUrl.Scheme == "http" {
			opts = append(opts, grpc.WithInsecure())
		} else {
			return nil, nil, fmt.Errorf("unhandled scheme: %q", grpcServerUrlString)
		}

		host := grpcServerUrl.Host

		conn, err := grpc.Dial(host, opts...)
		if err != nil {
			return nil, nil, fmt.Errorf("unable to connect to grpc server %q: %v", host, err)
		}

		// TODO: Close conn

		client = native.NewStorageServiceClient(conn)
	}
	nativeStore := native.NewStore(c.Prefix, c.Codec, client)
	return nativeStore, destroyFunc, nil
}
