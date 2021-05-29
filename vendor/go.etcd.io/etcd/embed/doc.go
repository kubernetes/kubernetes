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

/*
Package embed provides bindings for embedding an etcd server in a program.

Launch an embedded etcd server using the configuration defaults:

	import (
		"log"
		"time"

		"go.etcd.io/etcd/embed"
	)

	func main() {
		cfg := embed.NewConfig()
		cfg.Dir = "default.etcd"
		e, err := embed.StartEtcd(cfg)
		if err != nil {
			log.Fatal(err)
		}
		defer e.Close()
		select {
		case <-e.Server.ReadyNotify():
			log.Printf("Server is ready!")
		case <-time.After(60 * time.Second):
			e.Server.Stop() // trigger a shutdown
			log.Printf("Server took too long to start!")
		}
		log.Fatal(<-e.Err())
	}
*/
package embed
