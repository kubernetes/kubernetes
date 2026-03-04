/*
Copyright 2024 The Kubernetes Authors.

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

package modes

import (
	"testing"
)

type mockPool struct {
	v interface{}
}

func (*mockPool) Get() interface{} {
	return nil
}

func (p *mockPool) Put(v interface{}) {
	p.v = v
}

func TestBufferProviderPut(t *testing.T) {
	{
		p := new(mockPool)
		bp := &BufferProvider{p: p}
		small := new(buffer)
		small.Grow(3 * 1024 * 1024)
		small.WriteString("hello world")
		bp.Put(small)
		if p.v != small {
			t.Errorf("expected buf with capacity %d to be returned to pool", small.Cap())
		}
		if small.Len() != 0 {
			t.Errorf("expected buf to be reset before returning to pool")
		}
	}

	{
		p := new(mockPool)
		bp := &BufferProvider{p: p}
		big := new(buffer)
		big.Grow(3*1024*1024 + 1)
		bp.Put(big)
		if p.v != nil {
			t.Errorf("expected buf with capacity %d not to be returned to pool", big.Cap())
		}
	}
}
