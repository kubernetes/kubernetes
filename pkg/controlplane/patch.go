package controlplane

var kubeAPIServerEmitEventFn EventSinkFunc = nil

type EventSinkFunc func(eventType, reason, messageFmt string, args ...interface{})
