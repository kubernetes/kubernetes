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

package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
)

func main() {
	log.SetFlags(log.Lmicroseconds)

	endpointStr := flag.String("endpoints", "localhost:2379", "endpoints of etcd cluster")
	mode := flag.String("mode", "lock-racer", "test mode (election, lock-racer, lease-renewer)")
	round := flag.Int("rounds", 100, "number of rounds to run")
	flag.Parse()
	eps := strings.Split(*endpointStr, ",")

	switch *mode {
	case "election":
		runElection(eps, *round)
	case "lock-racer":
		runRacer(eps, *round)
	case "lease-renewer":
		runLeaseRenewer(eps)
	default:
		fmt.Fprintf(os.Stderr, "unsupported mode %v\n", *mode)
	}
}

func runElection(eps []string, rounds int) {
	rcs := make([]roundClient, 15)
	validatec, releasec := make(chan struct{}, len(rcs)), make(chan struct{}, len(rcs))
	for range rcs {
		releasec <- struct{}{}
	}

	for i := range rcs {
		v := fmt.Sprintf("%d", i)
		observedLeader := ""
		validateWaiters := 0

		rcs[i].c = randClient(eps)
		e := concurrency.NewElection(rcs[i].c, "electors")

		rcs[i].acquire = func() error {
			<-releasec
			ctx, cancel := context.WithCancel(context.Background())
			go func() {
				if ol, ok := <-e.Observe(ctx); ok {
					observedLeader = string(ol.Kvs[0].Value)
					if observedLeader != v {
						cancel()
					}
				}
			}()
			err := e.Campaign(ctx, v)
			if err == nil {
				observedLeader = v
			}
			if observedLeader == v {
				validateWaiters = len(rcs)
			}
			select {
			case <-ctx.Done():
				return nil
			default:
				cancel()
				return err
			}
		}
		rcs[i].validate = func() error {
			if l, err := e.Leader(context.TODO()); err == nil && l != observedLeader {
				return fmt.Errorf("expected leader %q, got %q", observedLeader, l)
			}
			validatec <- struct{}{}
			return nil
		}
		rcs[i].release = func() error {
			for validateWaiters > 0 {
				select {
				case <-validatec:
					validateWaiters--
				default:
					return fmt.Errorf("waiting on followers")
				}
			}
			if err := e.Resign(context.TODO()); err != nil {
				return err
			}
			if observedLeader == v {
				for range rcs {
					releasec <- struct{}{}
				}
			}
			observedLeader = ""
			return nil
		}
	}
	doRounds(rcs, rounds)
}

func runLeaseRenewer(eps []string) {
	c := randClient(eps)
	ctx := context.Background()

	for {
		var (
			l   *clientv3.LeaseGrantResponse
			lk  *clientv3.LeaseKeepAliveResponse
			err error
		)
		for {
			l, err = c.Lease.Grant(ctx, 5)
			if err == nil {
				break
			}
		}
		expire := time.Now().Add(time.Duration(l.TTL-1) * time.Second)

		for {
			lk, err = c.Lease.KeepAliveOnce(ctx, l.ID)
			if grpc.Code(err) == codes.NotFound {
				if time.Since(expire) < 0 {
					log.Printf("bad renew! exceeded: %v", time.Since(expire))
					for {
						lk, err = c.Lease.KeepAliveOnce(ctx, l.ID)
						fmt.Println(lk, err)
						time.Sleep(time.Second)
					}
				}
				log.Printf("lost lease %d, expire: %v\n", l.ID, expire)
				break
			}
			if err != nil {
				continue
			}
			expire = time.Now().Add(time.Duration(lk.TTL-1) * time.Second)
			log.Printf("renewed lease %d, expire: %v\n", lk.ID, expire)
			time.Sleep(time.Duration(lk.TTL-2) * time.Second)
		}
	}
}

func runRacer(eps []string, round int) {
	rcs := make([]roundClient, 15)
	ctx := context.Background()
	cnt := 0
	for i := range rcs {
		rcs[i].c = randClient(eps)
		m := concurrency.NewMutex(rcs[i].c, "racers")
		rcs[i].acquire = func() error { return m.Lock(ctx) }
		rcs[i].validate = func() error {
			if cnt++; cnt != 1 {
				return fmt.Errorf("bad lock; count: %d", cnt)
			}
			return nil
		}
		rcs[i].release = func() error {
			if err := m.Unlock(ctx); err != nil {
				return err
			}
			cnt = 0
			return nil
		}
	}
	doRounds(rcs, round)
}

func randClient(eps []string) *clientv3.Client {
	neps := make([]string, len(eps))
	copy(neps, eps)

	for i := range neps {
		j := rand.Intn(i + 1)
		neps[i], neps[j] = neps[j], neps[i]
	}

	c, err := clientv3.New(clientv3.Config{
		Endpoints:   eps,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		log.Fatal(err)
	}
	return c
}

type roundClient struct {
	c        *clientv3.Client
	progress int
	acquire  func() error
	validate func() error
	release  func() error
}

func doRounds(rcs []roundClient, rounds int) {
	var mu sync.Mutex
	var wg sync.WaitGroup

	wg.Add(len(rcs))
	finished := make(chan struct{}, 0)
	for i := range rcs {
		go func(rc *roundClient) {
			defer wg.Done()
			for rc.progress < rounds {
				for rc.acquire() != nil { /* spin */
				}

				mu.Lock()
				if err := rc.validate(); err != nil {
					log.Fatal(err)
				}
				mu.Unlock()

				time.Sleep(10 * time.Millisecond)
				rc.progress++
				finished <- struct{}{}

				mu.Lock()
				for rc.release() != nil {
					mu.Unlock()
					mu.Lock()
				}
				mu.Unlock()
			}
		}(&rcs[i])
	}

	start := time.Now()
	for i := 1; i < len(rcs)*rounds+1; i++ {
		select {
		case <-finished:
			if i%100 == 0 {
				fmt.Printf("finished %d, took %v\n", i, time.Since(start))
				start = time.Now()
			}
		case <-time.After(time.Minute):
			log.Panic("no progress after 1 minute!")
		}
	}

	wg.Wait()
	for _, rc := range rcs {
		rc.c.Close()
	}
}
