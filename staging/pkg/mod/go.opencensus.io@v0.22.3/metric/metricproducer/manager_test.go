// Copyright 2019, OpenCensus Authors
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

package metricproducer

import (
	"testing"

	"go.opencensus.io/metric/metricdata"
)

type testProducer struct {
	name string
}

var (
	myProd1 = newTestProducer("foo")
	myProd2 = newTestProducer("bar")
	myProd3 = newTestProducer("foobar")
	pm      = GlobalManager()
)

func newTestProducer(name string) *testProducer {
	return &testProducer{name}
}

func (mp *testProducer) Read() []*metricdata.Metric {
	return nil
}

func TestAdd(t *testing.T) {
	pm.AddProducer(myProd1)
	pm.AddProducer(myProd2)

	got := pm.GetAll()
	want := []*testProducer{myProd1, myProd2}
	checkSlice(got, want, t)
	deleteAll()
}

func TestAddExisting(t *testing.T) {
	pm.AddProducer(myProd1)
	pm.AddProducer(myProd2)
	pm.AddProducer(myProd1)

	got := pm.GetAll()
	want := []*testProducer{myProd2, myProd1}
	checkSlice(got, want, t)
	deleteAll()
}

func TestAddNil(t *testing.T) {
	pm.AddProducer(nil)

	got := pm.GetAll()
	want := []*testProducer{}
	checkSlice(got, want, t)
	deleteAll()
}

func TestDelete(t *testing.T) {
	pm.AddProducer(myProd1)
	pm.AddProducer(myProd2)
	pm.AddProducer(myProd3)
	pm.DeleteProducer(myProd2)

	got := pm.GetAll()
	want := []*testProducer{myProd1, myProd3}
	checkSlice(got, want, t)
	deleteAll()
}

func TestDeleteNonExisting(t *testing.T) {
	pm.AddProducer(myProd1)
	pm.AddProducer(myProd3)
	pm.DeleteProducer(myProd2)

	got := pm.GetAll()
	want := []*testProducer{myProd1, myProd3}
	checkSlice(got, want, t)
	deleteAll()
}

func TestDeleteNil(t *testing.T) {
	pm.AddProducer(myProd1)
	pm.AddProducer(myProd3)
	pm.DeleteProducer(nil)

	got := pm.GetAll()
	want := []*testProducer{myProd1, myProd3}
	checkSlice(got, want, t)
	deleteAll()
}

func TestGetAllNil(t *testing.T) {
	got := pm.GetAll()
	want := []*testProducer{}
	checkSlice(got, want, t)
	deleteAll()
}

func TestImmutableProducerList(t *testing.T) {
	pm.AddProducer(myProd1)
	pm.AddProducer(myProd2)

	producersToMutate := pm.GetAll()
	producersToMutate[0] = myProd3
	got := pm.GetAll()
	want := []*testProducer{myProd1, myProd2}
	checkSlice(got, want, t)
	deleteAll()
}

func checkSlice(got []Producer, want []*testProducer, t *testing.T) {
	gotLen := len(got)
	wantLen := len(want)
	if gotLen != wantLen {
		t.Errorf("got len: %d want: %d\n", gotLen, wantLen)
	} else {
		gotMap := map[Producer]struct{}{}
		for i := 0; i < gotLen; i++ {
			gotMap[got[i]] = struct{}{}
		}
		for i := 0; i < wantLen; i++ {
			delete(gotMap, want[i])
		}
		if len(gotMap) > 0 {
			t.Errorf("got %v, want %v\n", got, want)
		}
	}
}

func deleteAll() {
	pm.DeleteProducer(myProd1)
	pm.DeleteProducer(myProd2)
	pm.DeleteProducer(myProd3)
}
