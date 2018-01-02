package plugins

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/docker/docker/pkg/plugins/transport"
	"github.com/docker/go-connections/tlsconfig"
	"github.com/stretchr/testify/assert"
)

const (
	fruitPlugin     = "fruit"
	fruitImplements = "apple"
)

// regression test for deadlock in handlers
func TestPluginAddHandler(t *testing.T) {
	// make a plugin which is pre-activated
	p := &Plugin{activateWait: sync.NewCond(&sync.Mutex{})}
	p.Manifest = &Manifest{Implements: []string{"bananas"}}
	storage.plugins["qwerty"] = p

	testActive(t, p)
	Handle("bananas", func(_ string, _ *Client) {})
	testActive(t, p)
}

func TestPluginWaitBadPlugin(t *testing.T) {
	p := &Plugin{activateWait: sync.NewCond(&sync.Mutex{})}
	p.activateErr = errors.New("some junk happened")
	testActive(t, p)
}

func testActive(t *testing.T, p *Plugin) {
	done := make(chan struct{})
	go func() {
		p.waitActive()
		close(done)
	}()

	select {
	case <-time.After(100 * time.Millisecond):
		_, f, l, _ := runtime.Caller(1)
		t.Fatalf("%s:%d: deadlock in waitActive", filepath.Base(f), l)
	case <-done:
	}

}

func TestGet(t *testing.T) {
	p := &Plugin{name: fruitPlugin, activateWait: sync.NewCond(&sync.Mutex{})}
	p.Manifest = &Manifest{Implements: []string{fruitImplements}}
	storage.plugins[fruitPlugin] = p

	plugin, err := Get(fruitPlugin, fruitImplements)
	if err != nil {
		t.Fatal(err)
	}
	if p.Name() != plugin.Name() {
		t.Fatalf("No matching plugin with name %s found", plugin.Name())
	}
	if plugin.Client() != nil {
		t.Fatal("expected nil Client but found one")
	}
	if !plugin.IsV1() {
		t.Fatal("Expected true for V1 plugin")
	}

	// check negative case where plugin fruit doesn't implement banana
	_, err = Get("fruit", "banana")
	assert.Equal(t, err, ErrNotImplements)

	// check negative case where plugin vegetable doesn't exist
	_, err = Get("vegetable", "potato")
	assert.Equal(t, err, ErrNotFound)

}

func TestPluginWithNoManifest(t *testing.T) {
	addr := setupRemotePluginServer()
	defer teardownRemotePluginServer()

	m := Manifest{[]string{fruitImplements}}
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(m); err != nil {
		t.Fatal(err)
	}

	mux.HandleFunc("/Plugin.Activate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Fatalf("Expected POST, got %s\n", r.Method)
		}

		header := w.Header()
		header.Set("Content-Type", transport.VersionMimetype)

		io.Copy(w, &buf)
	})

	p := &Plugin{
		name:         fruitPlugin,
		activateWait: sync.NewCond(&sync.Mutex{}),
		Addr:         addr,
		TLSConfig:    &tlsconfig.Options{InsecureSkipVerify: true},
	}
	storage.plugins[fruitPlugin] = p

	plugin, err := Get(fruitPlugin, fruitImplements)
	if err != nil {
		t.Fatal(err)
	}
	if p.Name() != plugin.Name() {
		t.Fatalf("No matching plugin with name %s found", plugin.Name())
	}
}

func TestGetAll(t *testing.T) {
	tmpdir, unregister := Setup(t)
	defer unregister()

	p := filepath.Join(tmpdir, "example.json")
	spec := `{
	"Name": "example",
	"Addr": "https://example.com/docker/plugin"
}`

	if err := ioutil.WriteFile(p, []byte(spec), 0644); err != nil {
		t.Fatal(err)
	}

	r := newLocalRegistry()
	plugin, err := r.Plugin("example")
	if err != nil {
		t.Fatal(err)
	}
	plugin.Manifest = &Manifest{Implements: []string{"apple"}}
	storage.plugins["example"] = plugin

	fetchedPlugins, err := GetAll("apple")
	if err != nil {
		t.Fatal(err)
	}
	if fetchedPlugins[0].Name() != plugin.Name() {
		t.Fatalf("Expected to get plugin with name %s", plugin.Name())
	}
}
