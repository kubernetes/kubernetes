package file

import (
	"fmt"
	"io/ioutil"
	"strings"
	"time"

	"github.com/docker/docker/pkg/discovery"
)

// Discovery is exported
type Discovery struct {
	heartbeat time.Duration
	path      string
}

func init() {
	Init()
}

// Init is exported
func Init() {
	discovery.Register("file", &Discovery{})
}

// Initialize is exported
func (s *Discovery) Initialize(path string, heartbeat time.Duration, ttl time.Duration, _ map[string]string) error {
	s.path = path
	s.heartbeat = heartbeat
	return nil
}

func parseFileContent(content []byte) []string {
	var result []string
	for _, line := range strings.Split(strings.TrimSpace(string(content)), "\n") {
		line = strings.TrimSpace(line)
		// Ignoring line starts with #
		if strings.HasPrefix(line, "#") {
			continue
		}
		// Inlined # comment also ignored.
		if strings.Contains(line, "#") {
			line = line[0:strings.Index(line, "#")]
			// Trim additional spaces caused by above stripping.
			line = strings.TrimSpace(line)
		}
		result = append(result, discovery.Generate(line)...)
	}
	return result
}

func (s *Discovery) fetch() (discovery.Entries, error) {
	fileContent, err := ioutil.ReadFile(s.path)
	if err != nil {
		return nil, fmt.Errorf("failed to read '%s': %v", s.path, err)
	}
	return discovery.CreateEntries(parseFileContent(fileContent))
}

// Watch is exported
func (s *Discovery) Watch(stopCh <-chan struct{}) (<-chan discovery.Entries, <-chan error) {
	ch := make(chan discovery.Entries)
	errCh := make(chan error)
	ticker := time.NewTicker(s.heartbeat)

	go func() {
		defer close(errCh)
		defer close(ch)

		// Send the initial entries if available.
		currentEntries, err := s.fetch()
		if err != nil {
			errCh <- err
		} else {
			ch <- currentEntries
		}

		// Periodically send updates.
		for {
			select {
			case <-ticker.C:
				newEntries, err := s.fetch()
				if err != nil {
					errCh <- err
					continue
				}

				// Check if the file has really changed.
				if !newEntries.Equals(currentEntries) {
					ch <- newEntries
				}
				currentEntries = newEntries
			case <-stopCh:
				ticker.Stop()
				return
			}
		}
	}()

	return ch, errCh
}

// Register is exported
func (s *Discovery) Register(addr string) error {
	return discovery.ErrNotImplemented
}
