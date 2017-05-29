/*
Copyright 2017 The Kubernetes Authors.

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

package log

import (
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

type backend struct {
	out  io.Writer
	sink chan *auditinternal.Event
}

var _ audit.Backend = &backend{}

func NewBackend(out io.Writer) *backend {
	return &backend{
		out:  out,
		sink: make(chan *auditinternal.Event, 100),
	}
}

func (b *backend) ProcessEvents(events ...*auditinternal.Event) {
	for _, ev := range events {
		b.logEvent(ev)
	}
}

func (b *backend) logEvent(ev *auditinternal.Event) {
	username := "<none>"
	groups := "<none>"
	if len(ev.User.Username) > 0 {
		username = ev.User.Username
		if len(ev.User.Groups) > 0 {
			groups = auditStringSlice(ev.User.Groups)
		}
	}
	asuser := "<self>"
	asgroups := "<lookup>"
	if ev.ImpersonatedUser != nil {
		asuser = ev.ImpersonatedUser.Username
		if ev.ImpersonatedUser.Groups != nil {
			asgroups = auditStringSlice(ev.ImpersonatedUser.Groups)
		}
	}

	namespace := "<none>"
	if ev.ObjectRef != nil && len(ev.ObjectRef.Namespace) != 0 {
		namespace = ev.ObjectRef.Namespace
	}

	response := "<deferred>"
	if ev.ResponseStatus != nil {
		response = strconv.Itoa(int(ev.ResponseStatus.Code))
	}

	ip := "<unknown>"
	if len(ev.SourceIPs) > 0 {
		ip = ev.SourceIPs[0]
	}

	line := fmt.Sprintf("%s AUDIT: id=%q ip=%q method=%q user=%q groups=%q as=%q asgroups=%q namespace=%q uri=%q response=\"%s\"\n",
		ev.Timestamp.Format(time.RFC3339Nano), ev.AuditID, ip, ev.Verb, username, groups, asuser, asgroups, namespace, ev.RequestURI, response)
	if _, err := fmt.Fprint(b.out, line); err != nil {
		glog.Errorf("Unable to write audit log: %s, the error is: %v", line, err)
	}
}

func (b *backend) Run(stopCh <-chan struct{}) error {
	return nil
}

func auditStringSlice(inList []string) string {
	quotedElements := make([]string, len(inList))
	for i, in := range inList {
		quotedElements[i] = fmt.Sprintf("%q", in)
	}
	return strings.Join(quotedElements, ",")
}
