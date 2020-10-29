package enablement

import (
	"fmt"
	"runtime/debug"

	kubecontrolplanev1 "github.com/openshift/api/kubecontrolplane/v1"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/rest"
)

func ForceOpenShift(newOpenshiftConfig *kubecontrolplanev1.KubeAPIServerConfig) {
	isOpenShift = true
	openshiftConfig = newOpenshiftConfig
}

func SetLoopbackClientConfig(kubeClientConfig *rest.Config) {
	loopbackClientConfig = rest.CopyConfig(kubeClientConfig)
}

var (
	isOpenShift                = false
	openshiftConfig            *kubecontrolplanev1.KubeAPIServerConfig
	postStartHooks             = map[string]PostStartHookConfigEntry{}
	appendPostStartHooksCalled = false
	loopbackClientConfig       *rest.Config
)

type PostStartHookConfigEntry struct {
	Hook genericapiserver.PostStartHookFunc
	// originatingStack holds the stack that registered postStartHooks. This allows us to show a more helpful message
	// for duplicate registration.
	OriginatingStack string
}

func IsOpenShift() bool {
	return isOpenShift
}

func OpenshiftConfig() *kubecontrolplanev1.KubeAPIServerConfig {
	return openshiftConfig
}

func LoopbackClientConfig() *rest.Config {
	return loopbackClientConfig
}

func AddPostStartHookOrDie(name string, hook genericapiserver.PostStartHookFunc) {
	if appendPostStartHooksCalled {
		panic(fmt.Errorf("already appended post start hooks"))
	}
	if len(name) == 0 {
		panic(fmt.Errorf("missing name"))
	}
	if hook == nil {
		panic(fmt.Errorf("hook func may not be nil: %q", name))
	}

	if postStartHook, exists := postStartHooks[name]; exists {
		// this is programmer error, but it can be hard to debug
		panic(fmt.Errorf("unable to add %q because it was already registered by: %s", name, postStartHook.OriginatingStack))
	}
	postStartHooks[name] = PostStartHookConfigEntry{Hook: hook, OriginatingStack: string(debug.Stack())}
}

func AppendPostStartHooksOrDie(config *genericapiserver.Config) {
	appendPostStartHooksCalled = true
	for name, curr := range postStartHooks {
		config.AddPostStartHookOrDie(name, curr.Hook)
	}
}
