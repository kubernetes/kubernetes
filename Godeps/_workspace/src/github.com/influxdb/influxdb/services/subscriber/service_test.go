package subscriber_test

import (
	"net/url"
	"testing"
	"time"

	"github.com/influxdb/influxdb/cluster"
	"github.com/influxdb/influxdb/meta"
	"github.com/influxdb/influxdb/services/subscriber"
)

type MetaStore struct {
	DatabasesFn          func() ([]meta.DatabaseInfo, error)
	WaitForDataChangedFn func() error
}

func (m MetaStore) Databases() ([]meta.DatabaseInfo, error) {
	return m.DatabasesFn()
}

func (m MetaStore) WaitForDataChanged() error {
	return m.WaitForDataChangedFn()
}

type Subscription struct {
	WritePointsFn func(*cluster.WritePointsRequest) error
}

func (s Subscription) WritePoints(p *cluster.WritePointsRequest) error {
	return s.WritePointsFn(p)
}

func TestService_IgnoreNonMatch(t *testing.T) {
	dataChanged := make(chan bool)
	ms := MetaStore{}
	ms.WaitForDataChangedFn = func() error {
		<-dataChanged
		return nil
	}
	ms.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return []meta.DatabaseInfo{
			{
				Name: "db0",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name: "rp0",
						Subscriptions: []meta.SubscriptionInfo{
							{Name: "s0", Mode: "ANY", Destinations: []string{"udp://h0:9093", "udp://h1:9093"}},
						},
					},
				},
			},
		}, nil
	}

	prs := make(chan *cluster.WritePointsRequest, 2)
	urls := make(chan url.URL, 2)
	newPointsWriter := func(u url.URL) (subscriber.PointsWriter, error) {
		sub := Subscription{}
		sub.WritePointsFn = func(p *cluster.WritePointsRequest) error {
			prs <- p
			return nil
		}
		urls <- u
		return sub, nil
	}

	s := subscriber.NewService(subscriber.NewConfig())
	s.MetaStore = ms
	s.NewPointsWriter = newPointsWriter
	s.Open()
	defer s.Close()

	// Signal that data has changed
	dataChanged <- true

	for _, expURLStr := range []string{"udp://h0:9093", "udp://h1:9093"} {
		var u url.URL
		expURL, _ := url.Parse(expURLStr)
		select {
		case u = <-urls:
		case <-time.After(10 * time.Millisecond):
			t.Fatal("expected urls")
		}
		if expURL.String() != u.String() {
			t.Fatalf("unexpected url: got %s exp %s", u.String(), expURL.String())
		}
	}

	// Write points that don't match any subscription.
	s.Points() <- &cluster.WritePointsRequest{
		Database:        "db1",
		RetentionPolicy: "rp0",
	}
	s.Points() <- &cluster.WritePointsRequest{
		Database:        "db0",
		RetentionPolicy: "rp2",
	}

	// Shouldn't get any prs back
	select {
	case pr := <-prs:
		t.Fatalf("unexpected points request %v", pr)
	default:
	}
	close(dataChanged)
}

func TestService_ModeALL(t *testing.T) {
	dataChanged := make(chan bool)
	ms := MetaStore{}
	ms.WaitForDataChangedFn = func() error {
		<-dataChanged
		return nil
	}
	ms.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return []meta.DatabaseInfo{
			{
				Name: "db0",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name: "rp0",
						Subscriptions: []meta.SubscriptionInfo{
							{Name: "s0", Mode: "ALL", Destinations: []string{"udp://h0:9093", "udp://h1:9093"}},
						},
					},
				},
			},
		}, nil
	}

	prs := make(chan *cluster.WritePointsRequest, 2)
	urls := make(chan url.URL, 2)
	newPointsWriter := func(u url.URL) (subscriber.PointsWriter, error) {
		sub := Subscription{}
		sub.WritePointsFn = func(p *cluster.WritePointsRequest) error {
			prs <- p
			return nil
		}
		urls <- u
		return sub, nil
	}

	s := subscriber.NewService(subscriber.NewConfig())
	s.MetaStore = ms
	s.NewPointsWriter = newPointsWriter
	s.Open()
	defer s.Close()

	// Signal that data has changed
	dataChanged <- true

	for _, expURLStr := range []string{"udp://h0:9093", "udp://h1:9093"} {
		var u url.URL
		expURL, _ := url.Parse(expURLStr)
		select {
		case u = <-urls:
		case <-time.After(10 * time.Millisecond):
			t.Fatal("expected urls")
		}
		if expURL.String() != u.String() {
			t.Fatalf("unexpected url: got %s exp %s", u.String(), expURL.String())
		}
	}

	// Write points that match subscription with mode ALL
	expPR := &cluster.WritePointsRequest{
		Database:        "db0",
		RetentionPolicy: "rp0",
	}
	s.Points() <- expPR

	// Should get pr back twice
	for i := 0; i < 2; i++ {
		var pr *cluster.WritePointsRequest
		select {
		case pr = <-prs:
		case <-time.After(10 * time.Millisecond):
			t.Fatalf("expected points request: got %d exp 2", i)
		}
		if pr != expPR {
			t.Errorf("unexpected points request: got %v, exp %v", pr, expPR)
		}
	}
	close(dataChanged)
}

func TestService_ModeANY(t *testing.T) {
	dataChanged := make(chan bool)
	ms := MetaStore{}
	ms.WaitForDataChangedFn = func() error {
		<-dataChanged
		return nil
	}
	ms.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return []meta.DatabaseInfo{
			{
				Name: "db0",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name: "rp0",
						Subscriptions: []meta.SubscriptionInfo{
							{Name: "s0", Mode: "ANY", Destinations: []string{"udp://h0:9093", "udp://h1:9093"}},
						},
					},
				},
			},
		}, nil
	}

	prs := make(chan *cluster.WritePointsRequest, 2)
	urls := make(chan url.URL, 2)
	newPointsWriter := func(u url.URL) (subscriber.PointsWriter, error) {
		sub := Subscription{}
		sub.WritePointsFn = func(p *cluster.WritePointsRequest) error {
			prs <- p
			return nil
		}
		urls <- u
		return sub, nil
	}

	s := subscriber.NewService(subscriber.NewConfig())
	s.MetaStore = ms
	s.NewPointsWriter = newPointsWriter
	s.Open()
	defer s.Close()

	// Signal that data has changed
	dataChanged <- true

	for _, expURLStr := range []string{"udp://h0:9093", "udp://h1:9093"} {
		var u url.URL
		expURL, _ := url.Parse(expURLStr)
		select {
		case u = <-urls:
		case <-time.After(10 * time.Millisecond):
			t.Fatal("expected urls")
		}
		if expURL.String() != u.String() {
			t.Fatalf("unexpected url: got %s exp %s", u.String(), expURL.String())
		}
	}
	// Write points that match subscription with mode ANY
	expPR := &cluster.WritePointsRequest{
		Database:        "db0",
		RetentionPolicy: "rp0",
	}
	s.Points() <- expPR

	// Validate we get the pr back just once
	var pr *cluster.WritePointsRequest
	select {
	case pr = <-prs:
	case <-time.After(10 * time.Millisecond):
		t.Fatal("expected points request")
	}
	if pr != expPR {
		t.Errorf("unexpected points request: got %v, exp %v", pr, expPR)
	}

	// shouldn't get it a second time
	select {
	case pr = <-prs:
		t.Fatalf("unexpected points request %v", pr)
	default:
	}
	close(dataChanged)
}

func TestService_Multiple(t *testing.T) {
	dataChanged := make(chan bool)
	ms := MetaStore{}
	ms.WaitForDataChangedFn = func() error {
		<-dataChanged
		return nil
	}
	ms.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		return []meta.DatabaseInfo{
			{
				Name: "db0",
				RetentionPolicies: []meta.RetentionPolicyInfo{
					{
						Name: "rp0",
						Subscriptions: []meta.SubscriptionInfo{
							{Name: "s0", Mode: "ANY", Destinations: []string{"udp://h0:9093", "udp://h1:9093"}},
						},
					},
					{
						Name: "rp1",
						Subscriptions: []meta.SubscriptionInfo{
							{Name: "s1", Mode: "ALL", Destinations: []string{"udp://h2:9093", "udp://h3:9093"}},
						},
					},
				},
			},
		}, nil
	}

	prs := make(chan *cluster.WritePointsRequest, 4)
	urls := make(chan url.URL, 4)
	newPointsWriter := func(u url.URL) (subscriber.PointsWriter, error) {
		sub := Subscription{}
		sub.WritePointsFn = func(p *cluster.WritePointsRequest) error {
			prs <- p
			return nil
		}
		urls <- u
		return sub, nil
	}

	s := subscriber.NewService(subscriber.NewConfig())
	s.MetaStore = ms
	s.NewPointsWriter = newPointsWriter
	s.Open()
	defer s.Close()

	// Signal that data has changed
	dataChanged <- true

	for _, expURLStr := range []string{"udp://h0:9093", "udp://h1:9093", "udp://h2:9093", "udp://h3:9093"} {
		var u url.URL
		expURL, _ := url.Parse(expURLStr)
		select {
		case u = <-urls:
		case <-time.After(10 * time.Millisecond):
			t.Fatal("expected urls")
		}
		if expURL.String() != u.String() {
			t.Fatalf("unexpected url: got %s exp %s", u.String(), expURL.String())
		}
	}

	// Write points that don't match any subscription.
	s.Points() <- &cluster.WritePointsRequest{
		Database:        "db1",
		RetentionPolicy: "rp0",
	}
	s.Points() <- &cluster.WritePointsRequest{
		Database:        "db0",
		RetentionPolicy: "rp2",
	}

	// Write points that match subscription with mode ANY
	expPR := &cluster.WritePointsRequest{
		Database:        "db0",
		RetentionPolicy: "rp0",
	}
	s.Points() <- expPR

	// Validate we get the pr back just once
	var pr *cluster.WritePointsRequest
	select {
	case pr = <-prs:
	case <-time.After(10 * time.Millisecond):
		t.Fatal("expected points request")
	}
	if pr != expPR {
		t.Errorf("unexpected points request: got %v, exp %v", pr, expPR)
	}

	// shouldn't get it a second time
	select {
	case pr = <-prs:
		t.Fatalf("unexpected points request %v", pr)
	default:
	}

	// Write points that match subscription with mode ALL
	expPR = &cluster.WritePointsRequest{
		Database:        "db0",
		RetentionPolicy: "rp1",
	}
	s.Points() <- expPR

	// Should get pr back twice
	for i := 0; i < 2; i++ {
		select {
		case pr = <-prs:
		case <-time.After(10 * time.Millisecond):
			t.Fatalf("expected points request: got %d exp 2", i)
		}
		if pr != expPR {
			t.Errorf("unexpected points request: got %v, exp %v", pr, expPR)
		}
	}
	close(dataChanged)
}

func TestService_WaitForDataChanged(t *testing.T) {
	dataChanged := make(chan bool)
	ms := MetaStore{}
	ms.WaitForDataChangedFn = func() error {
		<-dataChanged
		return nil
	}
	calls := make(chan bool, 2)
	ms.DatabasesFn = func() ([]meta.DatabaseInfo, error) {
		calls <- true
		return nil, nil
	}

	s := subscriber.NewService(subscriber.NewConfig())
	s.MetaStore = ms
	// Explicitly closed below for testing
	s.Open()

	// Should be called once during open
	select {
	case <-calls:
	case <-time.After(10 * time.Millisecond):
		t.Fatal("expected call")
	}

	select {
	case <-calls:
		t.Fatal("unexpected call")
	case <-time.After(time.Millisecond):
	}

	// Signal that data has changed
	dataChanged <- true

	// Should be called once more after data changed
	select {
	case <-calls:
	case <-time.After(10 * time.Millisecond):
		t.Fatal("expected call")
	}

	select {
	case <-calls:
		t.Fatal("unexpected call")
	case <-time.After(time.Millisecond):
	}

	//Close service ensure not called
	s.Close()
	dataChanged <- true
	select {
	case <-calls:
		t.Fatal("unexpected call")
	case <-time.After(time.Millisecond):
	}

	close(dataChanged)
}
