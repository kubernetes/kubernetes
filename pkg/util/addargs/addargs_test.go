/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package addargs

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"testing"
)

func TestStackLevels(t *testing.T) {
	showStack := func() {
		stack := stackPool.Get().([]byte)
		defer stackPool.Put(stack)
		runtime.Stack(stack, false)
		t.Logf("stack:\n%s\n\n", stack)
	}
	f := func(i int) string { return "" }
	f = func(i int) string {
		mine, ok := computeKey(false)
		if !ok {
			showStack()
			t.Fatalf("%v: couldn't get key at all", i)
		}
		if i == 0 {
			v, ok := Get()
			if !ok || v != "foo" {
				showStack()
				t.Errorf("didn't get expected value, got %v", v)
			}
			return mine
		}
		theirs := f(i - 1)
		if mine != theirs {
			showStack()
			t.Fatalf("At level %v, expected %v but got %v", i, mine, theirs)
		}
		return mine
	}
	Invoke("foo", func() error {
		// TODO: 97 seems to be as high as this can go; making the
		// buffer bigger doesn't help. Go just stops printing stuff.
		f(80)
		Invoke("bar", func() error {
			val, _ := Get()
			if fmt.Sprintf("%v", val) != "bar" {
				showStack()
				t.Errorf("Wanted bar, got %v", val)
			}
			return nil
		})
		// Make sure that nested Invoke didn't clobber "foo"
		val, _ := Get()
		if fmt.Sprintf("%v", val) != "foo" {
			t.Errorf("Wanted foo, got %v", val)
		}
		return nil
	})
}

func TestRE(t *testing.T) {
	example := []byte("k8s.io/kubernetes/pkg/util/addargs.addAndCall(0xc8200165d0, 0xc8200b1f58, 0x0, 0x0)")
	out := keyRE.FindAllSubmatch(example, 1)
	if len(out) < 1 {
		t.Fatalf("got: %#v", out)
	}
	out2 := out[0]
	if len(out2) < 2 {
		t.Fatalf("got: %#v", out2)
	}
	got := string(out2[1])
	expect := "0xc8200165d0"
	if got != expect {
		t.Errorf("Got %v wanted %v", got, expect)
	}

	got, ok := tlsKey(example)
	if !ok {
		t.Errorf("tlsKey failed the example")
	}
	if got != expect {
		t.Errorf("Got %v wanted %v", got, expect)
	}
}

func TestArgs(t *testing.T) {
	runtime.GOMAXPROCS(12)
	wg := sync.WaitGroup{}
	const (
		threads = 50
		trials  = 500
	)

	wg.Add(threads)
	for i := 0; i < threads; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < trials; j++ {
				value := rand.Int()
				Invoke(value, func() error {
					//runtime.Gosched()
					got, ok := Get()
					if !ok {
						t.Errorf("unexpected failure to get value")
						return nil
					}
					if e, a := value, got; e != a {
						t.Errorf("wanted %v, got %v", e, a)
					}
					return nil
				})
			}
		}()
	}
	wg.Wait()
}

// This is on the order of 45us on lavalamp's desktop.
func BenchmarkInvoke(b *testing.B) {
	wg := sync.WaitGroup{}
	const (
		threads = 1
	)

	value := rand.Int()

	wg.Add(threads)
	for i := 0; i < threads; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < b.N; j++ {
				Invoke(value, func() error {
					got, ok := Get()
					if !ok {
						panic(fmt.Errorf("unexpected failure to get value"))
					}
					if e, a := value, got; e != a {
						panic(fmt.Errorf("wanted %v, got %v", e, a))
					}
					return nil
				})
			}
		}()
	}
	wg.Wait()
}

// About 24us on lavalamp's desktop.
func BenchmarkGet(b *testing.B) {
	value := rand.Int()
	Invoke(value, func() error {
		for j := 0; j < b.N; j++ {
			got, ok := Get()
			if !ok {
				panic(fmt.Errorf("unexpected failure to get value"))
			}
			if e, a := value, got; e != a {
				panic(fmt.Errorf("wanted %v, got %v", e, a))
			}
		}
		return nil
	})

}
