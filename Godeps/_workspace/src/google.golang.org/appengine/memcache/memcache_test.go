package memcache

import (
	"fmt"
	"testing"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal/aetesting"
	pb "google.golang.org/appengine/internal/memcache"
)

var errRPC = fmt.Errorf("RPC error")

func TestGetRequest(t *testing.T) {
	serviceCalled := false
	apiKey := "lyric"

	c := aetesting.FakeSingleContext(t, "memcache", "Get", func(req *pb.MemcacheGetRequest, _ *pb.MemcacheGetResponse) error {
		// Test request.
		if n := len(req.Key); n != 1 {
			t.Errorf("got %d want 1", n)
			return nil
		}
		if k := string(req.Key[0]); k != apiKey {
			t.Errorf("got %q want %q", k, apiKey)
		}

		serviceCalled = true
		return nil
	})

	// Test the "forward" path from the API call parameters to the
	// protobuf request object. (The "backward" path from the
	// protobuf response object to the API call response,
	// including the error response, are handled in the next few
	// tests).
	Get(c, apiKey)
	if !serviceCalled {
		t.Error("Service was not called as expected")
	}
}

func TestGetResponseHit(t *testing.T) {
	key := "lyric"
	value := "Where the buffalo roam"

	c := aetesting.FakeSingleContext(t, "memcache", "Get", func(_ *pb.MemcacheGetRequest, res *pb.MemcacheGetResponse) error {
		res.Item = []*pb.MemcacheGetResponse_Item{
			{Key: []byte(key), Value: []byte(value)},
		}
		return nil
	})
	apiItem, err := Get(c, key)
	if apiItem == nil || apiItem.Key != key || string(apiItem.Value) != value {
		t.Errorf("got %q, %q want {%q,%q}, nil", apiItem, err, key, value)
	}
}

func TestGetResponseMiss(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "memcache", "Get", func(_ *pb.MemcacheGetRequest, res *pb.MemcacheGetResponse) error {
		// don't fill in any of the response
		return nil
	})
	_, err := Get(c, "something")
	if err != ErrCacheMiss {
		t.Errorf("got %v want ErrCacheMiss", err)
	}
}

func TestGetResponseRPCError(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "memcache", "Get", func(_ *pb.MemcacheGetRequest, res *pb.MemcacheGetResponse) error {
		return errRPC
	})

	if _, err := Get(c, "something"); err != errRPC {
		t.Errorf("got %v want errRPC", err)
	}
}

func TestAddRequest(t *testing.T) {
	var apiItem = &Item{
		Key:   "lyric",
		Value: []byte("Oh, give me a home"),
	}

	serviceCalled := false

	c := aetesting.FakeSingleContext(t, "memcache", "Set", func(req *pb.MemcacheSetRequest, _ *pb.MemcacheSetResponse) error {
		// Test request.
		pbItem := req.Item[0]
		if k := string(pbItem.Key); k != apiItem.Key {
			t.Errorf("got %q want %q", k, apiItem.Key)
		}
		if v := string(apiItem.Value); v != string(pbItem.Value) {
			t.Errorf("got %q want %q", v, string(pbItem.Value))
		}
		if p := *pbItem.SetPolicy; p != pb.MemcacheSetRequest_ADD {
			t.Errorf("got %v want %v", p, pb.MemcacheSetRequest_ADD)
		}

		serviceCalled = true
		return nil
	})

	Add(c, apiItem)
	if !serviceCalled {
		t.Error("Service was not called as expected")
	}
}

func TestAddResponseStored(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "memcache", "Set", func(_ *pb.MemcacheSetRequest, res *pb.MemcacheSetResponse) error {
		res.SetStatus = []pb.MemcacheSetResponse_SetStatusCode{pb.MemcacheSetResponse_STORED}
		return nil
	})

	if err := Add(c, &Item{}); err != nil {
		t.Errorf("got %v want nil", err)
	}
}

func TestAddResponseNotStored(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "memcache", "Set", func(_ *pb.MemcacheSetRequest, res *pb.MemcacheSetResponse) error {
		res.SetStatus = []pb.MemcacheSetResponse_SetStatusCode{pb.MemcacheSetResponse_NOT_STORED}
		return nil
	})

	if err := Add(c, &Item{}); err != ErrNotStored {
		t.Errorf("got %v want ErrNotStored", err)
	}
}

func TestAddResponseError(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "memcache", "Set", func(_ *pb.MemcacheSetRequest, res *pb.MemcacheSetResponse) error {
		res.SetStatus = []pb.MemcacheSetResponse_SetStatusCode{pb.MemcacheSetResponse_ERROR}
		return nil
	})

	if err := Add(c, &Item{}); err != ErrServerError {
		t.Errorf("got %v want ErrServerError", err)
	}
}

func TestAddResponseRPCError(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "memcache", "Set", func(_ *pb.MemcacheSetRequest, res *pb.MemcacheSetResponse) error {
		return errRPC
	})

	if err := Add(c, &Item{}); err != errRPC {
		t.Errorf("got %v want errRPC", err)
	}
}

func TestSetRequest(t *testing.T) {
	var apiItem = &Item{
		Key:   "lyric",
		Value: []byte("Where the buffalo roam"),
	}

	serviceCalled := false

	c := aetesting.FakeSingleContext(t, "memcache", "Set", func(req *pb.MemcacheSetRequest, _ *pb.MemcacheSetResponse) error {
		// Test request.
		if n := len(req.Item); n != 1 {
			t.Errorf("got %d want 1", n)
			return nil
		}
		pbItem := req.Item[0]
		if k := string(pbItem.Key); k != apiItem.Key {
			t.Errorf("got %q want %q", k, apiItem.Key)
		}
		if v := string(pbItem.Value); v != string(apiItem.Value) {
			t.Errorf("got %q want %q", v, string(apiItem.Value))
		}
		if p := *pbItem.SetPolicy; p != pb.MemcacheSetRequest_SET {
			t.Errorf("got %v want %v", p, pb.MemcacheSetRequest_SET)
		}

		serviceCalled = true
		return nil
	})

	Set(c, apiItem)
	if !serviceCalled {
		t.Error("Service was not called as expected")
	}
}

func TestSetResponse(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "memcache", "Set", func(_ *pb.MemcacheSetRequest, res *pb.MemcacheSetResponse) error {
		res.SetStatus = []pb.MemcacheSetResponse_SetStatusCode{pb.MemcacheSetResponse_STORED}
		return nil
	})

	if err := Set(c, &Item{}); err != nil {
		t.Errorf("got %v want nil", err)
	}
}

func TestSetResponseError(t *testing.T) {
	c := aetesting.FakeSingleContext(t, "memcache", "Set", func(_ *pb.MemcacheSetRequest, res *pb.MemcacheSetResponse) error {
		res.SetStatus = []pb.MemcacheSetResponse_SetStatusCode{pb.MemcacheSetResponse_ERROR}
		return nil
	})

	if err := Set(c, &Item{}); err != ErrServerError {
		t.Errorf("got %v want ErrServerError", err)
	}
}

func TestNamespaceResetting(t *testing.T) {
	var nsField *string
	c := aetesting.FakeSingleContext(t, "memcache", "Get", func(req *pb.MemcacheGetRequest, res *pb.MemcacheGetResponse) error {
		nsField = req.NameSpace
		return errRPC
	})

	// Check that wrapping c in a namespace twice works correctly.
	nc, err := appengine.Namespace(c, "A")
	if err != nil {
		t.Fatalf("appengine.Namespace: %v", err)
	}
	c0, err := appengine.Namespace(nc, "") // should act as the original context
	if err != nil {
		t.Fatalf("appengine.Namespace: %v", err)
	}

	Get(c, "key")
	if nsField != nil {
		t.Fatalf("Get with c yielded %q", *nsField)
	}
	Get(nc, "key")
	if nsField == nil || *nsField != "A" {
		t.Fatalf("Get with nc yielded %v", nsField)
	}
	Get(c0, "key")
	if nsField != nil && *nsField != "" {
		t.Fatalf("Get with c0 yielded %q", *nsField)
	}
}

func TestGetMultiEmpty(t *testing.T) {
	serviceCalled := false
	c := aetesting.FakeSingleContext(t, "memcache", "Get", func(req *pb.MemcacheGetRequest, _ *pb.MemcacheGetResponse) error {
		serviceCalled = true
		return nil
	})

	// Test that the Memcache service is not called when
	// GetMulti is passed an empty slice of keys.
	GetMulti(c, []string{})
	if serviceCalled {
		t.Error("Service was called but should not have been")
	}
}
