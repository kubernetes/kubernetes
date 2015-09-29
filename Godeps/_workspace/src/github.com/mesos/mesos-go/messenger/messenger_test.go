package messenger

import (
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/mesos/mesos-go/messenger/testmessage"
	"github.com/mesos/mesos-go/upid"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
)

var (
	globalWG = new(sync.WaitGroup)
)

func noopHandler(*upid.UPID, proto.Message) {
	globalWG.Done()
}

func shuffleMessages(queue *[]proto.Message) {
	for i := range *queue {
		index := rand.Intn(i + 1)
		(*queue)[i], (*queue)[index] = (*queue)[index], (*queue)[i]
	}
}

func generateSmallMessages(n int) []proto.Message {
	queue := make([]proto.Message, n)
	for i := range queue {
		queue[i] = testmessage.GenerateSmallMessage()
	}
	return queue
}

func generateMediumMessages(n int) []proto.Message {
	queue := make([]proto.Message, n)
	for i := range queue {
		queue[i] = testmessage.GenerateMediumMessage()
	}
	return queue
}

func generateBigMessages(n int) []proto.Message {
	queue := make([]proto.Message, n)
	for i := range queue {
		queue[i] = testmessage.GenerateBigMessage()
	}
	return queue
}

func generateLargeMessages(n int) []proto.Message {
	queue := make([]proto.Message, n)
	for i := range queue {
		queue[i] = testmessage.GenerateLargeMessage()
	}
	return queue
}

func generateMixedMessages(n int) []proto.Message {
	queue := make([]proto.Message, n*4)
	for i := 0; i < n*4; i = i + 4 {
		queue[i] = testmessage.GenerateSmallMessage()
		queue[i+1] = testmessage.GenerateMediumMessage()
		queue[i+2] = testmessage.GenerateBigMessage()
		queue[i+3] = testmessage.GenerateLargeMessage()
	}
	shuffleMessages(&queue)
	return queue
}

func installMessages(t *testing.T, m Messenger, queue *[]proto.Message, counts *[]int, done chan struct{}) {
	testCounts := func(counts []int, done chan struct{}) {
		for i := range counts {
			if counts[i] != cap(*queue)/4 {
				return
			}
		}
		close(done)
	}
	hander1 := func(from *upid.UPID, pbMsg proto.Message) {
		(*queue) = append(*queue, pbMsg)
		(*counts)[0]++
		testCounts(*counts, done)
	}
	hander2 := func(from *upid.UPID, pbMsg proto.Message) {
		(*queue) = append(*queue, pbMsg)
		(*counts)[1]++
		testCounts(*counts, done)
	}
	hander3 := func(from *upid.UPID, pbMsg proto.Message) {
		(*queue) = append(*queue, pbMsg)
		(*counts)[2]++
		testCounts(*counts, done)
	}
	hander4 := func(from *upid.UPID, pbMsg proto.Message) {
		(*queue) = append(*queue, pbMsg)
		(*counts)[3]++
		testCounts(*counts, done)
	}
	assert.NoError(t, m.Install(hander1, &testmessage.SmallMessage{}))
	assert.NoError(t, m.Install(hander2, &testmessage.MediumMessage{}))
	assert.NoError(t, m.Install(hander3, &testmessage.BigMessage{}))
	assert.NoError(t, m.Install(hander4, &testmessage.LargeMessage{}))
}

func runTestServer(b *testing.B, wg *sync.WaitGroup) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/testserver/mesos.internal.SmallMessage", func(http.ResponseWriter, *http.Request) {
		wg.Done()
	})
	mux.HandleFunc("/testserver/mesos.internal.MediumMessage", func(http.ResponseWriter, *http.Request) {
		wg.Done()
	})
	mux.HandleFunc("/testserver/mesos.internal.BigMessage", func(http.ResponseWriter, *http.Request) {
		wg.Done()
	})
	mux.HandleFunc("/testserver/mesos.internal.LargeMessage", func(http.ResponseWriter, *http.Request) {
		wg.Done()
	})
	return httptest.NewServer(mux)
}

func TestMessengerFailToInstall(t *testing.T) {
	m := NewHttp(upid.UPID{ID: "mesos"})
	handler := func(from *upid.UPID, pbMsg proto.Message) {}
	assert.NotNil(t, m)
	assert.NoError(t, m.Install(handler, &testmessage.SmallMessage{}))
	assert.Error(t, m.Install(handler, &testmessage.SmallMessage{}))
}

func TestMessengerFailToSend(t *testing.T) {
	m := NewHttp(upid.UPID{ID: "foo", Host: "localhost"})
	assert.NoError(t, m.Start())
	self := m.UPID()
	assert.Error(t, m.Send(context.TODO(), &self, &testmessage.SmallMessage{}))
}

func TestMessenger(t *testing.T) {
	messages := generateMixedMessages(1000)

	m1 := NewHttp(upid.UPID{ID: "mesos1", Host: "localhost"})
	m2 := NewHttp(upid.UPID{ID: "mesos2", Host: "localhost"})

	done := make(chan struct{})
	counts := make([]int, 4)
	msgQueue := make([]proto.Message, 0, len(messages))
	installMessages(t, m2, &msgQueue, &counts, done)

	assert.NoError(t, m1.Start())
	assert.NoError(t, m2.Start())
	upid2 := m2.UPID()

	go func() {
		for _, msg := range messages {
			assert.NoError(t, m1.Send(context.TODO(), &upid2, msg))
		}
	}()

	select {
	case <-time.After(time.Second * 10):
		t.Fatalf("Timeout")
	case <-done:
	}

	for i := range counts {
		assert.Equal(t, 1000, counts[i])
	}
	assert.Equal(t, messages, msgQueue)
}

func BenchmarkMessengerSendSmallMessage(b *testing.B) {
	messages := generateSmallMessages(1000)

	wg := new(sync.WaitGroup)
	wg.Add(b.N)
	srv := runTestServer(b, wg)
	defer srv.Close()

	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))
	assert.NoError(b, err)

	m1 := NewHttp(upid.UPID{ID: "mesos1", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func BenchmarkMessengerSendMediumMessage(b *testing.B) {
	messages := generateMediumMessages(1000)

	wg := new(sync.WaitGroup)
	wg.Add(b.N)
	srv := runTestServer(b, wg)
	defer srv.Close()

	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))
	assert.NoError(b, err)

	m1 := NewHttp(upid.UPID{ID: "mesos1", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func BenchmarkMessengerSendBigMessage(b *testing.B) {
	messages := generateBigMessages(1000)

	wg := new(sync.WaitGroup)
	wg.Add(b.N)
	srv := runTestServer(b, wg)
	defer srv.Close()

	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))
	assert.NoError(b, err)

	m1 := NewHttp(upid.UPID{ID: "mesos1", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func BenchmarkMessengerSendLargeMessage(b *testing.B) {
	messages := generateLargeMessages(1000)

	wg := new(sync.WaitGroup)
	wg.Add(b.N)
	srv := runTestServer(b, wg)
	defer srv.Close()

	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))
	assert.NoError(b, err)

	m1 := NewHttp(upid.UPID{ID: "mesos1", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func BenchmarkMessengerSendMixedMessage(b *testing.B) {
	messages := generateMixedMessages(1000)

	wg := new(sync.WaitGroup)
	wg.Add(b.N)
	srv := runTestServer(b, wg)
	defer srv.Close()

	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))
	assert.NoError(b, err)

	m1 := NewHttp(upid.UPID{ID: "mesos1", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func BenchmarkMessengerSendRecvSmallMessage(b *testing.B) {
	globalWG.Add(b.N)

	messages := generateSmallMessages(1000)

	m1 := NewHttp(upid.UPID{ID: "foo1", Host: "localhost"})
	m2 := NewHttp(upid.UPID{ID: "foo2", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	assert.NoError(b, m2.Start())
	defer m2.Stop()

	assert.NoError(b, m2.Install(noopHandler, &testmessage.SmallMessage{}))

	upid2 := m2.UPID()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), &upid2, messages[i%1000])
	}
	globalWG.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func BenchmarkMessengerSendRecvMediumMessage(b *testing.B) {
	globalWG.Add(b.N)

	messages := generateMediumMessages(1000)

	m1 := NewHttp(upid.UPID{ID: "foo1", Host: "localhost"})
	m2 := NewHttp(upid.UPID{ID: "foo2", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	assert.NoError(b, m2.Start())
	defer m2.Stop()

	assert.NoError(b, m2.Install(noopHandler, &testmessage.MediumMessage{}))

	upid2 := m2.UPID()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), &upid2, messages[i%1000])
	}
	globalWG.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func BenchmarkMessengerSendRecvBigMessage(b *testing.B) {
	globalWG.Add(b.N)

	messages := generateBigMessages(1000)

	m1 := NewHttp(upid.UPID{ID: "foo1", Host: "localhost"})
	m2 := NewHttp(upid.UPID{ID: "foo2", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	assert.NoError(b, m2.Start())
	defer m2.Stop()

	assert.NoError(b, m2.Install(noopHandler, &testmessage.BigMessage{}))

	upid2 := m2.UPID()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), &upid2, messages[i%1000])
	}
	globalWG.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func BenchmarkMessengerSendRecvLargeMessage(b *testing.B) {
	globalWG.Add(b.N)
	messages := generateLargeMessages(1000)

	m1 := NewHttp(upid.UPID{ID: "foo1", Host: "localhost"})
	m2 := NewHttp(upid.UPID{ID: "foo2", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	assert.NoError(b, m2.Start())
	defer m2.Stop()

	assert.NoError(b, m2.Install(noopHandler, &testmessage.LargeMessage{}))

	upid2 := m2.UPID()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), &upid2, messages[i%1000])
	}
	globalWG.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func BenchmarkMessengerSendRecvMixedMessage(b *testing.B) {
	globalWG.Add(b.N)
	messages := generateMixedMessages(1000)

	m1 := NewHttp(upid.UPID{ID: "foo1", Host: "localhost"})
	m2 := NewHttp(upid.UPID{ID: "foo2", Host: "localhost"})
	assert.NoError(b, m1.Start())
	defer m1.Stop()

	assert.NoError(b, m2.Start())
	defer m2.Stop()

	assert.NoError(b, m2.Install(noopHandler, &testmessage.SmallMessage{}))
	assert.NoError(b, m2.Install(noopHandler, &testmessage.MediumMessage{}))
	assert.NoError(b, m2.Install(noopHandler, &testmessage.BigMessage{}))
	assert.NoError(b, m2.Install(noopHandler, &testmessage.LargeMessage{}))

	upid2 := m2.UPID()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), &upid2, messages[i%1000])
	}
	globalWG.Wait()
	b.StopTimer()
	time.Sleep(2 * time.Second) // allow time for connection cleanup
}

func TestUPIDBindingAddress(t *testing.T) {
	tt := []struct {
		hostname string
		binding  net.IP
		expected string
	}{
		{"", nil, ""},
		{"", net.IPv4(1, 2, 3, 4), "1.2.3.4"},
		{"", net.IPv4(0, 0, 0, 0), ""},
		{"localhost", nil, "127.0.0.1"},
		{"localhost", net.IPv4(5, 6, 7, 8), "5.6.7.8"},
		{"localhost", net.IPv4(0, 0, 0, 0), "127.0.0.1"},
		{"0.0.0.0", nil, ""},
		{"7.8.9.1", nil, "7.8.9.1"},
		{"7.8.9.1", net.IPv4(0, 0, 0, 0), "7.8.9.1"},
		{"7.8.9.1", net.IPv4(8, 9, 1, 2), "8.9.1.2"},
	}

	for i, tc := range tt {
		actual, err := UPIDBindingAddress(tc.hostname, tc.binding)
		if err != nil && tc.expected != "" {
			t.Fatalf("test case %d failed; expected %q instead of error %v", i+1, tc.expected, err)
		}
		if err == nil && actual != tc.expected {
			t.Fatalf("test case %d failed; expected %q instead of %q", i+1, tc.expected, actual)
		}
		if err != nil {
			t.Logf("test case %d; received expected error %v", i+1, err)
		}
	}
}
