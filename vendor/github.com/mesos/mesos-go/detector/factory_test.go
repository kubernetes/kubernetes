package detector

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type testDetector string

func (d testDetector) Start() error                 { return nil }
func (d testDetector) Detect(f MasterChanged) error { return nil }

func (d testDetector) Done() <-chan struct{} { return make(<-chan struct{}) }
func (d testDetector) Cancel()               {}

// unregister a factory plugin according to its prefix.
// this is part of the testing module on purpose: during normal execution there
// should be no need to dynamically unregister plugins.
func Unregister(prefix string) {
	pluginLock.Lock()
	defer pluginLock.Unlock()
	delete(plugins, prefix)
}

func TestDetectorFactoryRegister(t *testing.T) {
	prefix := "bbm:"
	Register(prefix, func(spec string, _ ...Option) (Master, error) {
		return testDetector("Hello!"), nil
	})
	defer Unregister(prefix)

	f, ok := MatchingPlugin(prefix)

	assert.True(t, ok)
	assert.NotNil(t, f)
}

func TestDectorFactoryNew_EmptySpec(t *testing.T) {
	assert := assert.New(t)
	m, err := New("")
	assert.NotNil(err)
	assert.Nil(m)
}

func TestDectorFactoryNew_InvalidSpec(t *testing.T) {
	assert := assert.New(t)
	m, err := New("localhost")
	assert.NotNil(err)
	assert.Nil(m)
}

func TestDectorFactoryNew_TrivialSpec(t *testing.T) {
	assert := assert.New(t)
	m, err := New("localhost:1")
	assert.NoError(err)
	assert.NotNil(m)
	assert.IsType(&Standalone{}, m)
}
