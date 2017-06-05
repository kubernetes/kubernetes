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
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"sync"
	"time"

	"github.com/coreos/etcd/client"
	"github.com/coreos/etcd/store"
	"github.com/urfave/cli"
	"golang.org/x/net/context"
)

type set struct {
	key   string
	value string
	ttl   int64
}

func NewImportSnapCommand() cli.Command {
	return cli.Command{
		Name:      "import",
		Usage:     "import a snapshot to a cluster",
		ArgsUsage: " ",
		Flags: []cli.Flag{
			cli.StringFlag{Name: "snap", Value: "", Usage: "Path to the valid etcd 0.4.x snapshot."},
			cli.StringSliceFlag{Name: "hidden", Value: new(cli.StringSlice), Usage: "Hidden key spaces to import from snapshot"},
			cli.IntFlag{Name: "c", Value: 10, Usage: "Number of concurrent clients to import the data"},
		},
		Action: handleImportSnap,
	}
}

func handleImportSnap(c *cli.Context) error {
	d, err := ioutil.ReadFile(c.String("snap"))
	if err != nil {
		if c.String("snap") == "" {
			fmt.Printf("no snapshot file provided (use --snap)\n")
		} else {
			fmt.Printf("cannot read snapshot file %s\n", c.String("snap"))
		}
		os.Exit(1)
	}

	st := store.New()
	err = st.Recovery(d)

	wg := &sync.WaitGroup{}
	setc := make(chan set)
	concurrent := c.Int("c")
	fmt.Printf("starting to import snapshot %s with %d clients\n", c.String("snap"), concurrent)
	for i := 0; i < concurrent; i++ {
		go runSet(mustNewKeyAPI(c), setc, wg)
	}

	all, err := st.Get("/", true, true)
	if err != nil {
		handleError(ExitServerError, err)
	}
	n := copyKeys(all.Node, setc)

	hiddens := c.StringSlice("hidden")
	for _, h := range hiddens {
		allh, err := st.Get(h, true, true)
		if err != nil {
			handleError(ExitServerError, err)
		}
		n += copyKeys(allh.Node, setc)
	}
	close(setc)
	wg.Wait()
	fmt.Printf("finished importing %d keys\n", n)
	return nil
}

func copyKeys(n *store.NodeExtern, setc chan set) int {
	num := 0
	if !n.Dir {
		setc <- set{n.Key, *n.Value, n.TTL}
		return 1
	}
	log.Println("entering dir:", n.Key)
	for _, nn := range n.Nodes {
		sub := copyKeys(nn, setc)
		num += sub
	}
	return num
}

func runSet(ki client.KeysAPI, setc chan set, wg *sync.WaitGroup) {
	for s := range setc {
		log.Println("copying key:", s.key)
		if s.ttl != 0 && s.ttl < 300 {
			log.Printf("extending key %s's ttl to 300 seconds", s.key)
			s.ttl = 5 * 60
		}
		_, err := ki.Set(context.TODO(), s.key, s.value, &client.SetOptions{TTL: time.Duration(s.ttl) * time.Second})
		if err != nil {
			log.Fatalf("failed to copy key: %v\n", err)
		}
	}
	wg.Done()
}
