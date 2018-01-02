package libcontainerd

import "golang.org/x/net/context"

type client struct {
	clientCommon

	// Platform specific properties below here.
	remote        *remote
	q             queue
	exitNotifiers map[string]*exitNotifier
	liveRestore   bool
}

// GetServerVersion returns the connected server version information
func (clnt *client) GetServerVersion(ctx context.Context) (*ServerVersion, error) {
	resp, err := clnt.remote.apiClient.GetServerVersion(ctx, &containerd.GetServerVersionRequest{})
	if err != nil {
		return nil, err
	}

	sv := &ServerVersion{
		GetServerVersionResponse: *resp,
	}

	return sv, nil
}

func (clnt *client) AddProcess(ctx context.Context, containerID, processFriendlyName string, specp Process, attachStdio StdioCallback) (int, error) {
	return -1, nil
}

func (clnt *client) SignalProcess(containerID string, pid string, sig int) error {
	return nil
}

func (clnt *client) Resize(containerID, processFriendlyName string, width, height int) error {
	return nil
}

func (clnt *client) Pause(containerID string) error {
	return nil
}

func (clnt *client) Resume(containerID string) error {
	return nil
}

func (clnt *client) Stats(containerID string) (*Stats, error) {
	return nil, nil
}

func (clnt *client) getExitNotifier(containerID string) *exitNotifier {
	clnt.mapMutex.RLock()
	defer clnt.mapMutex.RUnlock()
	return clnt.exitNotifiers[containerID]
}

func (clnt *client) getOrCreateExitNotifier(containerID string) *exitNotifier {
	clnt.mapMutex.Lock()
	defer clnt.mapMutex.Unlock()
	w, ok := clnt.exitNotifiers[containerID]
	if !ok {
		w = &exitNotifier{c: make(chan struct{}), client: clnt}
		clnt.exitNotifiers[containerID] = w
	}
	return w
}

// Restore is the handler for restoring a container
func (clnt *client) Restore(containerID string, attachStdio StdioCallback, options ...CreateOption) error {
	return nil
}

func (clnt *client) GetPidsForContainer(containerID string) ([]int, error) {
	return nil, nil
}

// Summary returns a summary of the processes running in a container.
func (clnt *client) Summary(containerID string) ([]Summary, error) {
	return nil, nil
}

// UpdateResources updates resources for a running container.
func (clnt *client) UpdateResources(containerID string, resources Resources) error {
	// Updating resource isn't supported on Solaris
	// but we should return nil for enabling updating container
	return nil
}

func (clnt *client) CreateCheckpoint(containerID string, checkpointID string, checkpointDir string, exit bool) error {
	return nil
}

func (clnt *client) DeleteCheckpoint(containerID string, checkpointID string, checkpointDir string) error {
	return nil
}

func (clnt *client) ListCheckpoints(containerID string, checkpointDir string) (*Checkpoints, error) {
	return nil, nil
}
