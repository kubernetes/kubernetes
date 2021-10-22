// +build go1.9

package credentials

import (
	"fmt"
	"strconv"
	"sync"
	"testing"
	"time"
)

func BenchmarkCredentials_Get(b *testing.B) {
	stub := &stubProvider{}

	cases := []int{1, 10, 100, 500, 1000, 10000}

	for _, c := range cases {
		b.Run(strconv.Itoa(c), func(b *testing.B) {
			creds := NewCredentials(stub)
			var wg sync.WaitGroup
			wg.Add(c)
			for i := 0; i < c; i++ {
				go func() {
					for j := 0; j < b.N; j++ {
						v, err := creds.Get()
						if err != nil {
							b.Fatalf("expect no error %v, %v", v, err)
						}
					}
					wg.Done()
				}()
			}
			b.ResetTimer()

			wg.Wait()
		})
	}
}

func BenchmarkCredentials_Get_Expire(b *testing.B) {
	p := &blockProvider{}

	expRates := []int{10000, 1000, 100}
	cases := []int{1, 10, 100, 500, 1000, 10000}

	for _, expRate := range expRates {
		for _, c := range cases {
			b.Run(fmt.Sprintf("%d-%d", expRate, c), func(b *testing.B) {
				creds := NewCredentials(p)
				var wg sync.WaitGroup
				wg.Add(c)
				for i := 0; i < c; i++ {
					go func(id int) {
						for j := 0; j < b.N; j++ {
							v, err := creds.Get()
							if err != nil {
								b.Fatalf("expect no error %v, %v", v, err)
							}
							// periodically expire creds to cause rwlock
							if id == 0 && j%expRate == 0 {
								creds.Expire()
							}
						}
						wg.Done()
					}(i)
				}
				b.ResetTimer()

				wg.Wait()
			})
		}
	}
}

type blockProvider struct {
	creds   Value
	expired bool
	err     error
}

func (s *blockProvider) Retrieve() (Value, error) {
	s.expired = false
	s.creds.ProviderName = "blockProvider"
	time.Sleep(time.Millisecond)
	return s.creds, s.err
}
func (s *blockProvider) IsExpired() bool {
	return s.expired
}
