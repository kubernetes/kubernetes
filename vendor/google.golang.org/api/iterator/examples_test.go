// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package iterator_test

import (
	"bytes"
	"fmt"
	"html/template"
	"log"
	"math"
	"net/http"
	"sort"
	"strconv"

	"golang.org/x/net/context"
	"google.golang.org/api/iterator"
)

var (
	client *Client
	ctx    = context.Background()
)

var pageTemplate = template.Must(template.New("").Parse(`
<table>
  {{range .Entries}}
    <tr><td>{{.}}</td></tr>
  {{end}}
</table>
{{with .Next}}
  <a href="/entries?pageToken={{.}}">Next Page</a>
{{end}}
`))

// This example demonstrates how to use Pager to support
// pagination on a web site.
func Example_webHandler(w http.ResponseWriter, r *http.Request) {
	const pageSize = 25
	it := client.Items(ctx)
	var items []int
	pageToken, err := iterator.NewPager(it, pageSize, r.URL.Query().Get("pageToken")).NextPage(&items)
	if err != nil {
		http.Error(w, fmt.Sprintf("getting next page: %v", err), http.StatusInternalServerError)
	}
	data := struct {
		Items []int
		Next  string
	}{
		items,
		pageToken,
	}
	var buf bytes.Buffer
	if err := pageTemplate.Execute(&buf, data); err != nil {
		http.Error(w, fmt.Sprintf("executing page template: %v", err), http.StatusInternalServerError)
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if _, err := buf.WriteTo(w); err != nil {
		log.Printf("writing response: %v", err)
	}
}

// This example demonstrates how to use a Pager to page through an iterator in a loop.
func Example_pageLoop() {
	// Find all primes up to 42, in pages of size 5.
	const max = 42
	const pageSize = 5
	p := iterator.NewPager(Primes(max), pageSize, "" /* start from the beginning */)
	for page := 0; ; page++ {
		var items []int
		pageToken, err := p.NextPage(&items)
		if err != nil {
			log.Fatalf("Iterator paging failed: %v", err)
		}
		fmt.Printf("Page %d: %v\n", page, items)
		if pageToken == "" {
			break
		}
	}
	// Output:
	// Page 0: [2 3 5 7 11]
	// Page 1: [13 17 19 23 29]
	// Page 2: [31 37 41]
}

// The example demonstrates how to use a Pager to request a page from a given token.
func Example_pageToken() {
	const pageSize = 5
	const pageToken = "1337"
	p := iterator.NewPager(Primes(0), pageSize, pageToken)

	var items []int
	nextPage, err := p.NextPage(&items)
	if err != nil {
		log.Fatalf("Iterator paging failed: %v", err)
	}
	fmt.Printf("Primes: %v\nToken:  %q\n", items, nextPage)
	// Output:
	// Primes: [1361 1367 1373 1381 1399]
	// Token:  "1400"
}

// This example demonstrates how to get exactly the items in the buffer, without
// triggering an extra RPC.
func Example_serverPages() {
	// The iterator returned by Primes has a default page size of 20, which means
	// it will return all the primes in the range [2, 21).
	it := Primes(0)
	var items []int
	for {
		item, err := it.Next()
		if err != nil && err != iterator.Done {
			log.Fatal(err)
		}
		if err == iterator.Done {
			break
		}
		items = append(items, item)
		if it.PageInfo().Remaining() == 0 {
			break
		}
	}
	fmt.Println(items)
	// Output:
	// [2 3 5 7 11 13 17 19]
}

// Primes returns a iterator which returns a sequence of prime numbers.
// If non-zero, max specifies the maximum number which could possibly be
// returned.
func Primes(max int) *SieveIterator {
	it := &SieveIterator{pos: 2, max: max}
	it.pageInfo, it.nextFunc = iterator.NewPageInfo(
		it.fetch,
		func() int { return len(it.items) },
		func() interface{} { b := it.items; it.items = nil; return b })
	return it
}

// SieveIterator is an iterator that returns primes using the sieve of
// Eratosthenes. It is a demonstration of how an iterator might work.
// Internally, it uses "page size" as the number of ints to consider,
// and "page token" as the first number to consider (defaults to 2).
type SieveIterator struct {
	pageInfo *iterator.PageInfo
	nextFunc func() error
	max      int   // The largest number to consider.
	p        []int // Primes in the range [2, pos).
	pos      int   // Next number to consider when generating p.
	items    []int
}

// PageInfo returns a PageInfo, which supports pagination.
func (it *SieveIterator) PageInfo() *iterator.PageInfo { return it.pageInfo }

func (it *SieveIterator) fetch(pageSize int, pageToken string) (string, error) {
	start := 2
	if pageToken != "" {
		s, err := strconv.Atoi(pageToken)
		if err != nil || s < 2 {
			return "", fmt.Errorf("invalid token %q", pageToken)
		}
		start = s
	}
	if pageSize == 0 {
		pageSize = 20 // Default page size.
	}

	// Make sure sufficient primes have been calculated.
	it.calc(start + pageSize)

	// Find the subslice of primes which match this page.
	// Note that PageInfo requires that fetch does not remove any existing items,
	// so we cannot assume that items is empty at this call.
	items := it.p[sort.SearchInts(it.p, start):]
	items = items[:sort.SearchInts(items, start+pageSize)]
	it.items = append(it.items, items...)

	if it.max > 0 && start+pageSize > it.max {
		return "", nil // No more possible numbers to return.
	}

	return strconv.Itoa(start + pageSize), nil
}

// calc populates p with all primes up to, but not including, max.
func (it *SieveIterator) calc(max int) {
	if it.max > 0 && max > it.max+1 { // it.max is an inclusive bounds, max is exclusive.
		max = it.max + 1
	}
outer:
	for x := it.pos; x < max; x++ {
		sqrt := int(math.Sqrt(float64(x)))
		for _, p := range it.p {
			switch {
			case x%p == 0:
				// Not a prime.
				continue outer
			case p > sqrt:
				// Only need to check up to sqrt.
				break
			}
		}
		it.p = append(it.p, x)
	}
	it.pos = max
}

func (it *SieveIterator) Next() (int, error) {
	if err := it.nextFunc(); err != nil {
		return 0, err
	}
	item := it.items[0]
	it.items = it.items[1:]
	return item, nil
}
