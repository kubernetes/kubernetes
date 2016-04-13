package logger

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

// Creator is a method that builds a logging driver instance with given context
type Creator func(Context) (Logger, error)

//LogOptValidator is a method that validates the log opts provided
type LogOptValidator func(cfg map[string]string) error

// Context provides enough information for a logging driver to do its function
type Context struct {
	Config              map[string]string
	ContainerID         string
	ContainerName       string
	ContainerEntrypoint string
	ContainerArgs       []string
	ContainerImageID    string
	ContainerImageName  string
	ContainerCreated    time.Time
	LogPath             string
}

func (ctx *Context) Hostname() (string, error) {
	hostname, err := os.Hostname()
	if err != nil {
		return "", fmt.Errorf("logger: can not resolve hostname: %v", err)
	}
	return hostname, nil
}

func (ctx *Context) Command() string {
	terms := []string{ctx.ContainerEntrypoint}
	for _, arg := range ctx.ContainerArgs {
		terms = append(terms, arg)
	}
	command := strings.Join(terms, " ")
	return command
}

type logdriverFactory struct {
	registry     map[string]Creator
	optValidator map[string]LogOptValidator
	m            sync.Mutex
}

func (lf *logdriverFactory) register(name string, c Creator) error {
	lf.m.Lock()
	defer lf.m.Unlock()

	if _, ok := lf.registry[name]; ok {
		return fmt.Errorf("logger: log driver named '%s' is already registered", name)
	}
	lf.registry[name] = c
	return nil
}

func (lf *logdriverFactory) registerLogOptValidator(name string, l LogOptValidator) error {
	lf.m.Lock()
	defer lf.m.Unlock()

	if _, ok := lf.optValidator[name]; ok {
		return fmt.Errorf("logger: log driver named '%s' is already registered", name)
	}
	lf.optValidator[name] = l
	return nil
}

func (lf *logdriverFactory) get(name string) (Creator, error) {
	lf.m.Lock()
	defer lf.m.Unlock()

	c, ok := lf.registry[name]
	if !ok {
		return c, fmt.Errorf("logger: no log driver named '%s' is registered", name)
	}
	return c, nil
}

func (lf *logdriverFactory) getLogOptValidator(name string) LogOptValidator {
	lf.m.Lock()
	defer lf.m.Unlock()

	c, _ := lf.optValidator[name]
	return c
}

var factory = &logdriverFactory{registry: make(map[string]Creator), optValidator: make(map[string]LogOptValidator)} // global factory instance

// RegisterLogDriver registers the given logging driver builder with given logging
// driver name.
func RegisterLogDriver(name string, c Creator) error {
	return factory.register(name, c)
}

func RegisterLogOptValidator(name string, l LogOptValidator) error {
	return factory.registerLogOptValidator(name, l)
}

// GetLogDriver provides the logging driver builder for a logging driver name.
func GetLogDriver(name string) (Creator, error) {
	return factory.get(name)
}

func ValidateLogOpts(name string, cfg map[string]string) error {
	l := factory.getLogOptValidator(name)
	if l != nil {
		return l(cfg)
	}
	return fmt.Errorf("Log Opts are not valid for [%s] driver", name)
}
