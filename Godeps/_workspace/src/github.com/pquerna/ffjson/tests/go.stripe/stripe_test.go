/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package goser

import (
	"encoding/json"
	base "github.com/pquerna/ffjson/tests/go.stripe/base"
	ff "github.com/pquerna/ffjson/tests/go.stripe/ff"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	var customerTripped ff.Customer
	customer := ff.NewCustomer()

	buf1, err := json.Marshal(&customer)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}

	err = json.Unmarshal(buf1, &customerTripped)
	if err != nil {
		print(string(buf1))
		t.Fatalf("Unmarshal: %v", err)
	}
}

func BenchmarkMarshalJSON(b *testing.B) {
	cust := base.NewCustomer()

	buf, err := json.Marshal(&cust)
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := json.Marshal(&cust)
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
	}
}

func BenchmarkFFMarshalJSON(b *testing.B) {
	cust := ff.NewCustomer()

	buf, err := cust.MarshalJSON()
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := cust.MarshalJSON()
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
	}
}

type fatalF interface {
	Fatalf(format string, args ...interface{})
}

func getBaseData(b fatalF) []byte {
	cust := base.NewCustomer()
	buf, err := json.MarshalIndent(&cust, "", "    ")
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	return buf
}

func BenchmarkUnmarshalJSON(b *testing.B) {
	rec := base.Customer{}
	buf := getBaseData(b)
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := json.Unmarshal(buf, &rec)
		if err != nil {
			b.Fatalf("Marshal: %v", err)
		}
	}
}

func BenchmarkFFUnmarshalJSON(b *testing.B) {
	rec := ff.Customer{}
	buf := getBaseData(b)
	b.SetBytes(int64(len(buf)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := rec.UnmarshalJSON(buf)
		if err != nil {
			b.Fatalf("UnmarshalJSON: %v", err)
		}
	}
}
