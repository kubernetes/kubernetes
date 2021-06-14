package kubernetesservice

var KubeAPIServerEmitEventFn EventSinkFunc = nil

type EventSinkFunc func(eventType, reason, messageFmt string, args ...interface{})
