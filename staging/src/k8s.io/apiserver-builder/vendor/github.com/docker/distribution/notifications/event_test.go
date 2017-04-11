package notifications

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/docker/distribution/manifest/schema1"
)

// TestEventJSONFormat provides silly test to detect if the event format or
// envelope has changed. If this code fails, the revision of the protocol may
// need to be incremented.
func TestEventEnvelopeJSONFormat(t *testing.T) {
	var expected = strings.TrimSpace(`
{
   "events": [
      {
         "id": "asdf-asdf-asdf-asdf-0",
         "timestamp": "2006-01-02T15:04:05Z",
         "action": "push",
         "target": {
            "mediaType": "application/vnd.docker.distribution.manifest.v1+prettyjws",
            "size": 1,
            "digest": "sha256:0123456789abcdef0",
            "length": 1,
            "repository": "library/test",
            "url": "http://example.com/v2/library/test/manifests/latest"
         },
         "request": {
            "id": "asdfasdf",
            "addr": "client.local",
            "host": "registrycluster.local",
            "method": "PUT",
            "useragent": "test/0.1"
         },
         "actor": {
            "name": "test-actor"
         },
         "source": {
            "addr": "hostname.local:port"
         }
      },
      {
         "id": "asdf-asdf-asdf-asdf-1",
         "timestamp": "2006-01-02T15:04:05Z",
         "action": "push",
         "target": {
            "mediaType": "application/vnd.docker.container.image.rootfs.diff+x-gtar",
            "size": 2,
            "digest": "sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5",
            "length": 2,
            "repository": "library/test",
            "url": "http://example.com/v2/library/test/manifests/latest"
         },
         "request": {
            "id": "asdfasdf",
            "addr": "client.local",
            "host": "registrycluster.local",
            "method": "PUT",
            "useragent": "test/0.1"
         },
         "actor": {
            "name": "test-actor"
         },
         "source": {
            "addr": "hostname.local:port"
         }
      },
      {
         "id": "asdf-asdf-asdf-asdf-2",
         "timestamp": "2006-01-02T15:04:05Z",
         "action": "push",
         "target": {
            "mediaType": "application/vnd.docker.container.image.rootfs.diff+x-gtar",
            "size": 3,
            "digest": "sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d6",
            "length": 3,
            "repository": "library/test",
            "url": "http://example.com/v2/library/test/manifests/latest"
         },
         "request": {
            "id": "asdfasdf",
            "addr": "client.local",
            "host": "registrycluster.local",
            "method": "PUT",
            "useragent": "test/0.1"
         },
         "actor": {
            "name": "test-actor"
         },
         "source": {
            "addr": "hostname.local:port"
         }
      }
   ]
}
	`)

	tm, err := time.Parse(time.RFC3339, time.RFC3339[:len(time.RFC3339)-5])
	if err != nil {
		t.Fatalf("error creating time: %v", err)
	}

	var prototype Event
	prototype.Action = EventActionPush
	prototype.Timestamp = tm
	prototype.Actor.Name = "test-actor"
	prototype.Request.ID = "asdfasdf"
	prototype.Request.Addr = "client.local"
	prototype.Request.Host = "registrycluster.local"
	prototype.Request.Method = "PUT"
	prototype.Request.UserAgent = "test/0.1"
	prototype.Source.Addr = "hostname.local:port"

	var manifestPush Event
	manifestPush = prototype
	manifestPush.ID = "asdf-asdf-asdf-asdf-0"
	manifestPush.Target.Digest = "sha256:0123456789abcdef0"
	manifestPush.Target.Length = 1
	manifestPush.Target.Size = 1
	manifestPush.Target.MediaType = schema1.MediaTypeSignedManifest
	manifestPush.Target.Repository = "library/test"
	manifestPush.Target.URL = "http://example.com/v2/library/test/manifests/latest"

	var layerPush0 Event
	layerPush0 = prototype
	layerPush0.ID = "asdf-asdf-asdf-asdf-1"
	layerPush0.Target.Digest = "sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5"
	layerPush0.Target.Length = 2
	layerPush0.Target.Size = 2
	layerPush0.Target.MediaType = layerMediaType
	layerPush0.Target.Repository = "library/test"
	layerPush0.Target.URL = "http://example.com/v2/library/test/manifests/latest"

	var layerPush1 Event
	layerPush1 = prototype
	layerPush1.ID = "asdf-asdf-asdf-asdf-2"
	layerPush1.Target.Digest = "sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d6"
	layerPush1.Target.Length = 3
	layerPush1.Target.Size = 3
	layerPush1.Target.MediaType = layerMediaType
	layerPush1.Target.Repository = "library/test"
	layerPush1.Target.URL = "http://example.com/v2/library/test/manifests/latest"

	var envelope Envelope
	envelope.Events = append(envelope.Events, manifestPush, layerPush0, layerPush1)

	p, err := json.MarshalIndent(envelope, "", "   ")
	if err != nil {
		t.Fatalf("unexpected error marshaling envelope: %v", err)
	}
	if string(p) != expected {
		t.Fatalf("format has changed\n%s\n != \n%s", string(p), expected)
	}
}
