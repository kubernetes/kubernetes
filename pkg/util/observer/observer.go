/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package observer

type SubjectEventType int
type Notifier func(se SubjectEvent)
type Handler func(enent interface{}) error

type forceFixer interface {
	ForceFix() bool
}

type CallBack struct {
	Handler Handler
}

type SubjectEvent struct {
	Cb    *CallBack //SubjectEvent需要可hash,func()不可hash
	Event interface{}
}

func (s SubjectEvent) Handle() error {
	return s.Cb.Handler(s.Event)
}

func (s SubjectEvent) ForceFix() bool {
	if force, ok := s.Event.(forceFixer); ok {
		return force.ForceFix()
	}
	return false
}

func NewSubjectEvent(h Handler, e interface{}) SubjectEvent {
	return SubjectEvent{
		Cb:    &CallBack{h},
		Event: e,
	}
}

type Observer interface {
	Attach(SubjectEventType, Notifier, Handler)
	AttachSubjectEvent(SubjectEventType, Notifier, SubjectEvent)
	Notify(se SubjectEventType, enent interface{})
}

func NewObserver() Observer {
	return &Registry{
		observers: make(map[SubjectEventType][]RegistInfo),
	}
}

type Registry struct {
	observers map[SubjectEventType][]RegistInfo
}

func (r *Registry) Attach(et SubjectEventType, n Notifier, h Handler) {
	r.observers[et] = append(r.observers[et],
		RegistInfo{
			se:       SubjectEvent{Cb: &CallBack{h}, Event: nil},
			notifier: n,
		})
}

func (r *Registry) AttachSubjectEvent(et SubjectEventType, n Notifier, se SubjectEvent) {
	r.observers[et] = append(r.observers[et],
		RegistInfo{
			se:       se,
			notifier: n,
		})
}

func (r *Registry) Notify(et SubjectEventType, enent interface{}) {
	if infos, ok := r.observers[et]; ok {
		for _, o := range infos {
			o.se.Event = enent
			o.notifier(o.se)
		}
	}
}

type RegistInfo struct {
	se       SubjectEvent
	notifier Notifier
}
