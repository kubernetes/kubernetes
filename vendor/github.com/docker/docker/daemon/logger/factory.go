package logger

import (
	"fmt"
	"sort"
	"sync"

	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/pkg/plugingetter"
	units "github.com/docker/go-units"
	"github.com/pkg/errors"
)

// Creator builds a logging driver instance with given context.
type Creator func(Info) (Logger, error)

// LogOptValidator checks the options specific to the underlying
// logging implementation.
type LogOptValidator func(cfg map[string]string) error

type logdriverFactory struct {
	registry     map[string]Creator
	optValidator map[string]LogOptValidator
	m            sync.Mutex
}

func (lf *logdriverFactory) list() []string {
	ls := make([]string, 0, len(lf.registry))
	lf.m.Lock()
	for name := range lf.registry {
		ls = append(ls, name)
	}
	lf.m.Unlock()
	sort.Strings(ls)
	return ls
}

// ListDrivers gets the list of registered log driver names
func ListDrivers() []string {
	return factory.list()
}

func (lf *logdriverFactory) register(name string, c Creator) error {
	if lf.driverRegistered(name) {
		return fmt.Errorf("logger: log driver named '%s' is already registered", name)
	}

	lf.m.Lock()
	lf.registry[name] = c
	lf.m.Unlock()
	return nil
}

func (lf *logdriverFactory) driverRegistered(name string) bool {
	lf.m.Lock()
	_, ok := lf.registry[name]
	lf.m.Unlock()
	if !ok {
		if pluginGetter != nil { // this can be nil when the init functions are running
			if l, _ := getPlugin(name, plugingetter.Lookup); l != nil {
				return true
			}
		}
	}
	return ok
}

func (lf *logdriverFactory) registerLogOptValidator(name string, l LogOptValidator) error {
	lf.m.Lock()
	defer lf.m.Unlock()

	if _, ok := lf.optValidator[name]; ok {
		return fmt.Errorf("logger: log validator named '%s' is already registered", name)
	}
	lf.optValidator[name] = l
	return nil
}

func (lf *logdriverFactory) get(name string) (Creator, error) {
	lf.m.Lock()
	defer lf.m.Unlock()

	c, ok := lf.registry[name]
	if ok {
		return c, nil
	}

	c, err := getPlugin(name, plugingetter.Acquire)
	return c, errors.Wrapf(err, "logger: no log driver named '%s' is registered", name)
}

func (lf *logdriverFactory) getLogOptValidator(name string) LogOptValidator {
	lf.m.Lock()
	defer lf.m.Unlock()

	c := lf.optValidator[name]
	return c
}

var factory = &logdriverFactory{registry: make(map[string]Creator), optValidator: make(map[string]LogOptValidator)} // global factory instance

// RegisterLogDriver registers the given logging driver builder with given logging
// driver name.
func RegisterLogDriver(name string, c Creator) error {
	return factory.register(name, c)
}

// RegisterLogOptValidator registers the logging option validator with
// the given logging driver name.
func RegisterLogOptValidator(name string, l LogOptValidator) error {
	return factory.registerLogOptValidator(name, l)
}

// GetLogDriver provides the logging driver builder for a logging driver name.
func GetLogDriver(name string) (Creator, error) {
	return factory.get(name)
}

var builtInLogOpts = map[string]bool{
	"mode":            true,
	"max-buffer-size": true,
}

// ValidateLogOpts checks the options for the given log driver. The
// options supported are specific to the LogDriver implementation.
func ValidateLogOpts(name string, cfg map[string]string) error {
	if name == "none" {
		return nil
	}

	switch containertypes.LogMode(cfg["mode"]) {
	case containertypes.LogModeBlocking, containertypes.LogModeNonBlock, containertypes.LogModeUnset:
	default:
		return fmt.Errorf("logger: logging mode not supported: %s", cfg["mode"])
	}

	if s, ok := cfg["max-buffer-size"]; ok {
		if containertypes.LogMode(cfg["mode"]) != containertypes.LogModeNonBlock {
			return fmt.Errorf("logger: max-buffer-size option is only supported with 'mode=%s'", containertypes.LogModeNonBlock)
		}
		if _, err := units.RAMInBytes(s); err != nil {
			return errors.Wrap(err, "error parsing option max-buffer-size")
		}
	}

	if !factory.driverRegistered(name) {
		return fmt.Errorf("logger: no log driver named '%s' is registered", name)
	}

	filteredOpts := make(map[string]string, len(builtInLogOpts))
	for k, v := range cfg {
		if !builtInLogOpts[k] {
			filteredOpts[k] = v
		}
	}

	validator := factory.getLogOptValidator(name)
	if validator != nil {
		return validator(filteredOpts)
	}
	return nil
}
