package notifications

import (
	"testing"

	"github.com/docker/distribution"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/manifest/schema1"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/api/v2"
	"github.com/docker/distribution/uuid"
	"github.com/docker/libtrust"
)

var (
	// common environment for expected manifest events.

	repo   = "test/repo"
	source = SourceRecord{
		Addr:       "remote.test",
		InstanceID: uuid.Generate().String(),
	}
	ub = mustUB(v2.NewURLBuilderFromString("http://test.example.com/", false))

	actor = ActorRecord{
		Name: "test",
	}
	request = RequestRecord{}
	m       = schema1.Manifest{
		Name: repo,
		Tag:  "latest",
	}

	sm      *schema1.SignedManifest
	payload []byte
	dgst    digest.Digest
)

func TestEventBridgeManifestPulled(t *testing.T) {
	l := createTestEnv(t, testSinkFn(func(events ...Event) error {
		checkCommonManifest(t, EventActionPull, events...)

		return nil
	}))

	repoRef, _ := reference.ParseNamed(repo)
	if err := l.ManifestPulled(repoRef, sm); err != nil {
		t.Fatalf("unexpected error notifying manifest pull: %v", err)
	}
}

func TestEventBridgeManifestPushed(t *testing.T) {
	l := createTestEnv(t, testSinkFn(func(events ...Event) error {
		checkCommonManifest(t, EventActionPush, events...)

		return nil
	}))

	repoRef, _ := reference.ParseNamed(repo)
	if err := l.ManifestPushed(repoRef, sm); err != nil {
		t.Fatalf("unexpected error notifying manifest pull: %v", err)
	}
}

func TestEventBridgeManifestPushedWithTag(t *testing.T) {
	l := createTestEnv(t, testSinkFn(func(events ...Event) error {
		checkCommonManifest(t, EventActionPush, events...)
		if events[0].Target.Tag != "latest" {
			t.Fatalf("missing or unexpected tag: %#v", events[0].Target)
		}

		return nil
	}))

	repoRef, _ := reference.ParseNamed(repo)
	if err := l.ManifestPushed(repoRef, sm, distribution.WithTag(m.Tag)); err != nil {
		t.Fatalf("unexpected error notifying manifest pull: %v", err)
	}
}

func TestEventBridgeManifestPulledWithTag(t *testing.T) {
	l := createTestEnv(t, testSinkFn(func(events ...Event) error {
		checkCommonManifest(t, EventActionPull, events...)
		if events[0].Target.Tag != "latest" {
			t.Fatalf("missing or unexpected tag: %#v", events[0].Target)
		}

		return nil
	}))

	repoRef, _ := reference.ParseNamed(repo)
	if err := l.ManifestPulled(repoRef, sm, distribution.WithTag(m.Tag)); err != nil {
		t.Fatalf("unexpected error notifying manifest pull: %v", err)
	}
}

func TestEventBridgeManifestDeleted(t *testing.T) {
	l := createTestEnv(t, testSinkFn(func(events ...Event) error {
		checkDeleted(t, EventActionDelete, events...)
		return nil
	}))

	repoRef, _ := reference.ParseNamed(repo)
	if err := l.ManifestDeleted(repoRef, dgst); err != nil {
		t.Fatalf("unexpected error notifying manifest pull: %v", err)
	}
}

func createTestEnv(t *testing.T, fn testSinkFn) Listener {
	pk, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("error generating private key: %v", err)
	}

	sm, err = schema1.Sign(&m, pk)
	if err != nil {
		t.Fatalf("error signing manifest: %v", err)
	}

	payload = sm.Canonical
	dgst = digest.FromBytes(payload)

	return NewBridge(ub, source, actor, request, fn)
}

func checkDeleted(t *testing.T, action string, events ...Event) {
	if len(events) != 1 {
		t.Fatalf("unexpected number of events: %v != 1", len(events))
	}

	event := events[0]

	if event.Source != source {
		t.Fatalf("source not equal: %#v != %#v", event.Source, source)
	}

	if event.Request != request {
		t.Fatalf("request not equal: %#v != %#v", event.Request, request)
	}

	if event.Actor != actor {
		t.Fatalf("request not equal: %#v != %#v", event.Actor, actor)
	}

	if event.Target.Digest != dgst {
		t.Fatalf("unexpected digest on event target: %q != %q", event.Target.Digest, dgst)
	}

	if event.Target.Repository != repo {
		t.Fatalf("unexpected repository: %q != %q", event.Target.Repository, repo)
	}

}

func checkCommonManifest(t *testing.T, action string, events ...Event) {
	checkCommon(t, events...)

	event := events[0]
	if event.Action != action {
		t.Fatalf("unexpected event action: %q != %q", event.Action, action)
	}

	repoRef, _ := reference.ParseNamed(repo)
	ref, _ := reference.WithDigest(repoRef, dgst)
	u, err := ub.BuildManifestURL(ref)
	if err != nil {
		t.Fatalf("error building expected url: %v", err)
	}

	if event.Target.URL != u {
		t.Fatalf("incorrect url passed: \n%q != \n%q", event.Target.URL, u)
	}
}

func checkCommon(t *testing.T, events ...Event) {
	if len(events) != 1 {
		t.Fatalf("unexpected number of events: %v != 1", len(events))
	}

	event := events[0]

	if event.Source != source {
		t.Fatalf("source not equal: %#v != %#v", event.Source, source)
	}

	if event.Request != request {
		t.Fatalf("request not equal: %#v != %#v", event.Request, request)
	}

	if event.Actor != actor {
		t.Fatalf("request not equal: %#v != %#v", event.Actor, actor)
	}

	if event.Target.Digest != dgst {
		t.Fatalf("unexpected digest on event target: %q != %q", event.Target.Digest, dgst)
	}

	if event.Target.Length != int64(len(payload)) {
		t.Fatalf("unexpected target length: %v != %v", event.Target.Length, len(payload))
	}

	if event.Target.Repository != repo {
		t.Fatalf("unexpected repository: %q != %q", event.Target.Repository, repo)
	}

}

type testSinkFn func(events ...Event) error

func (tsf testSinkFn) Write(events ...Event) error {
	return tsf(events...)
}

func (tsf testSinkFn) Close() error { return nil }

func mustUB(ub *v2.URLBuilder, err error) *v2.URLBuilder {
	if err != nil {
		panic(err)
	}

	return ub
}
