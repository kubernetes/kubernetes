package messenger

import (
	"fmt"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strconv"
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
	startPort = 10000 + rand.Intn(30000)
	globalWG  = new(sync.WaitGroup)
)

func noopHandler(*upid.UPID, proto.Message) {
	globalWG.Done()
}

func getNewPort() int {
	startPort++
	return startPort
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
	m := NewHttp(&upid.UPID{ID: "mesos"})
	handler := func(from *upid.UPID, pbMsg proto.Message) {}
	assert.NotNil(t, m)
	assert.NoError(t, m.Install(handler, &testmessage.SmallMessage{}))
	assert.Error(t, m.Install(handler, &testmessage.SmallMessage{}))
}

func TestMessengerFailToStart(t *testing.T) {
	port := strconv.Itoa(getNewPort())
	m1 := NewHttp(&upid.UPID{ID: "mesos", Host: "localhost", Port: port})
	m2 := NewHttp(&upid.UPID{ID: "mesos", Host: "localhost", Port: port})
	assert.NoError(t, m1.Start())
	assert.Error(t, m2.Start())
}

func TestMessengerFailToSend(t *testing.T) {
	upid, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(t, err)
	m := NewHttp(upid)
	assert.NoError(t, m.Start())
	assert.Error(t, m.Send(context.TODO(), upid, &testmessage.SmallMessage{}))
}

func TestMessenger(t *testing.T) {
	messages := generateMixedMessages(1000)

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(t, err)
	upid2, err := upid.Parse(fmt.Sprintf("mesos2@localhost:%d", getNewPort()))
	assert.NoError(t, err)

	m1 := NewHttp(upid1)
	m2 := NewHttp(upid2)

	done := make(chan struct{})
	counts := make([]int, 4)
	msgQueue := make([]proto.Message, 0, len(messages))
	installMessages(t, m2, &msgQueue, &counts, done)

	assert.NoError(t, m1.Start())
	assert.NoError(t, m2.Start())

	go func() {
		for _, msg := range messages {
			assert.NoError(t, m1.Send(context.TODO(), upid2, msg))
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

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))

	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	assert.NoError(b, m1.Start())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
}

func BenchmarkMessengerSendMediumMessage(b *testing.B) {
	messages := generateMediumMessages(1000)

	wg := new(sync.WaitGroup)
	wg.Add(b.N)
	srv := runTestServer(b, wg)
	defer srv.Close()

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))
	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	assert.NoError(b, m1.Start())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
}

func BenchmarkMessengerSendBigMessage(b *testing.B) {
	messages := generateBigMessages(1000)

	wg := new(sync.WaitGroup)
	wg.Add(b.N)
	srv := runTestServer(b, wg)
	defer srv.Close()

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))
	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	assert.NoError(b, m1.Start())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
}

func BenchmarkMessengerSendLargeMessage(b *testing.B) {
	messages := generateLargeMessages(1000)

	wg := new(sync.WaitGroup)
	wg.Add(b.N)
	srv := runTestServer(b, wg)
	defer srv.Close()

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))
	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	assert.NoError(b, m1.Start())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
}

func BenchmarkMessengerSendMixedMessage(b *testing.B) {
	messages := generateMixedMessages(1000)

	wg := new(sync.WaitGroup)
	wg.Add(b.N)
	srv := runTestServer(b, wg)
	defer srv.Close()

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("testserver@%s", srv.Listener.Addr().String()))
	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	assert.NoError(b, m1.Start())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	wg.Wait()
}

func BenchmarkMessengerSendRecvSmallMessage(b *testing.B) {
	globalWG.Add(b.N)

	messages := generateSmallMessages(1000)

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("mesos2@localhost:%d", getNewPort()))
	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	m2 := NewHttp(upid2)
	assert.NoError(b, m1.Start())
	assert.NoError(b, m2.Start())
	assert.NoError(b, m2.Install(noopHandler, &testmessage.SmallMessage{}))

	time.Sleep(time.Second) // Avoid race on upid.
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	globalWG.Wait()
}

func BenchmarkMessengerSendRecvMediumMessage(b *testing.B) {
	globalWG.Add(b.N)

	messages := generateMediumMessages(1000)

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("mesos2@localhost:%d", getNewPort()))
	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	m2 := NewHttp(upid2)
	assert.NoError(b, m1.Start())
	assert.NoError(b, m2.Start())
	assert.NoError(b, m2.Install(noopHandler, &testmessage.MediumMessage{}))

	time.Sleep(time.Second) // Avoid race on upid.
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	globalWG.Wait()
}

func BenchmarkMessengerSendRecvBigMessage(b *testing.B) {
	globalWG.Add(b.N)

	messages := generateBigMessages(1000)

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("mesos2@localhost:%d", getNewPort()))
	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	m2 := NewHttp(upid2)
	assert.NoError(b, m1.Start())
	assert.NoError(b, m2.Start())
	assert.NoError(b, m2.Install(noopHandler, &testmessage.BigMessage{}))

	time.Sleep(time.Second) // Avoid race on upid.
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	globalWG.Wait()
}

func BenchmarkMessengerSendRecvLargeMessage(b *testing.B) {
	globalWG.Add(b.N)
	messages := generateLargeMessages(1000)

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("mesos2@localhost:%d", getNewPort()))
	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	m2 := NewHttp(upid2)
	assert.NoError(b, m1.Start())
	assert.NoError(b, m2.Start())
	assert.NoError(b, m2.Install(noopHandler, &testmessage.LargeMessage{}))

	time.Sleep(time.Second) // Avoid race on upid.
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	globalWG.Wait()
}

func BenchmarkMessengerSendRecvMixedMessage(b *testing.B) {
	globalWG.Add(b.N)
	messages := generateMixedMessages(1000)

	upid1, err := upid.Parse(fmt.Sprintf("mesos1@localhost:%d", getNewPort()))
	assert.NoError(b, err)
	upid2, err := upid.Parse(fmt.Sprintf("mesos2@localhost:%d", getNewPort()))
	assert.NoError(b, err)

	m1 := NewHttp(upid1)
	m2 := NewHttp(upid2)
	assert.NoError(b, m1.Start())
	assert.NoError(b, m2.Start())
	assert.NoError(b, m2.Install(noopHandler, &testmessage.SmallMessage{}))
	assert.NoError(b, m2.Install(noopHandler, &testmessage.MediumMessage{}))
	assert.NoError(b, m2.Install(noopHandler, &testmessage.BigMessage{}))
	assert.NoError(b, m2.Install(noopHandler, &testmessage.LargeMessage{}))

	time.Sleep(time.Second) // Avoid race on upid.
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m1.Send(context.TODO(), upid2, messages[i%1000])
	}
	globalWG.Wait()
}
