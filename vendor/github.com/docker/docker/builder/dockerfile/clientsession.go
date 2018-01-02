package dockerfile

import (
	"time"

	"github.com/docker/docker/builder/fscache"
	"github.com/docker/docker/builder/remotecontext"
	"github.com/moby/buildkit/session"
	"github.com/moby/buildkit/session/filesync"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
)

const sessionConnectTimeout = 5 * time.Second

// ClientSessionTransport is a transport for copying files from docker client
// to the daemon.
type ClientSessionTransport struct{}

// NewClientSessionTransport returns new ClientSessionTransport instance
func NewClientSessionTransport() *ClientSessionTransport {
	return &ClientSessionTransport{}
}

// Copy data from a remote to a destination directory.
func (cst *ClientSessionTransport) Copy(ctx context.Context, id fscache.RemoteIdentifier, dest string, cu filesync.CacheUpdater) error {
	csi, ok := id.(*ClientSessionSourceIdentifier)
	if !ok {
		return errors.New("invalid identifier for client session")
	}

	return filesync.FSSync(ctx, csi.caller, filesync.FSSendRequestOpt{
		IncludePatterns: csi.includePatterns,
		DestDir:         dest,
		CacheUpdater:    cu,
	})
}

// ClientSessionSourceIdentifier is an identifier that can be used for requesting
// files from remote client
type ClientSessionSourceIdentifier struct {
	includePatterns []string
	caller          session.Caller
	sharedKey       string
	uuid            string
}

// NewClientSessionSourceIdentifier returns new ClientSessionSourceIdentifier instance
func NewClientSessionSourceIdentifier(ctx context.Context, sg SessionGetter, uuid string) (*ClientSessionSourceIdentifier, error) {
	csi := &ClientSessionSourceIdentifier{
		uuid: uuid,
	}
	caller, err := sg.Get(ctx, uuid)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get session for %s", uuid)
	}

	csi.caller = caller
	return csi, nil
}

// Transport returns transport identifier for remote identifier
func (csi *ClientSessionSourceIdentifier) Transport() string {
	return remotecontext.ClientSessionRemote
}

// SharedKey returns shared key for remote identifier. Shared key is used
// for finding the base for a repeated transfer.
func (csi *ClientSessionSourceIdentifier) SharedKey() string {
	return csi.caller.SharedKey()
}

// Key returns unique key for remote identifier. Requests with same key return
// same data.
func (csi *ClientSessionSourceIdentifier) Key() string {
	return csi.uuid
}
