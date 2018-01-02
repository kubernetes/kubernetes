package exec

import (
	"runtime"
	"sync"

	"github.com/docker/docker/container/stream"
	"github.com/docker/docker/libcontainerd"
	"github.com/docker/docker/pkg/stringid"
	"github.com/sirupsen/logrus"
)

// Config holds the configurations for execs. The Daemon keeps
// track of both running and finished execs so that they can be
// examined both during and after completion.
type Config struct {
	sync.Mutex
	StreamConfig *stream.Config
	ID           string
	Running      bool
	ExitCode     *int
	OpenStdin    bool
	OpenStderr   bool
	OpenStdout   bool
	CanRemove    bool
	ContainerID  string
	DetachKeys   []byte
	Entrypoint   string
	Args         []string
	Tty          bool
	Privileged   bool
	User         string
	Env          []string
	Pid          int
}

// NewConfig initializes the a new exec configuration
func NewConfig() *Config {
	return &Config{
		ID:           stringid.GenerateNonCryptoID(),
		StreamConfig: stream.NewConfig(),
	}
}

// InitializeStdio is called by libcontainerd to connect the stdio.
func (c *Config) InitializeStdio(iop libcontainerd.IOPipe) error {
	c.StreamConfig.CopyToPipe(iop)

	if c.StreamConfig.Stdin() == nil && !c.Tty && runtime.GOOS == "windows" {
		if iop.Stdin != nil {
			if err := iop.Stdin.Close(); err != nil {
				logrus.Errorf("error closing exec stdin: %+v", err)
			}
		}
	}

	return nil
}

// CloseStreams closes the stdio streams for the exec
func (c *Config) CloseStreams() error {
	return c.StreamConfig.CloseStreams()
}

// Store keeps track of the exec configurations.
type Store struct {
	commands map[string]*Config
	sync.RWMutex
}

// NewStore initializes a new exec store.
func NewStore() *Store {
	return &Store{commands: make(map[string]*Config, 0)}
}

// Commands returns the exec configurations in the store.
func (e *Store) Commands() map[string]*Config {
	e.RLock()
	commands := make(map[string]*Config, len(e.commands))
	for id, config := range e.commands {
		commands[id] = config
	}
	e.RUnlock()
	return commands
}

// Add adds a new exec configuration to the store.
func (e *Store) Add(id string, Config *Config) {
	e.Lock()
	e.commands[id] = Config
	e.Unlock()
}

// Get returns an exec configuration by its id.
func (e *Store) Get(id string) *Config {
	e.RLock()
	res := e.commands[id]
	e.RUnlock()
	return res
}

// Delete removes an exec configuration from the store.
func (e *Store) Delete(id string) {
	e.Lock()
	delete(e.commands, id)
	e.Unlock()
}

// List returns the list of exec ids in the store.
func (e *Store) List() []string {
	var IDs []string
	e.RLock()
	for id := range e.commands {
		IDs = append(IDs, id)
	}
	e.RUnlock()
	return IDs
}
