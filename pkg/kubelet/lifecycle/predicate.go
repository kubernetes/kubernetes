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

package lifecycle

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type getNodeAnyWayFuncType func() (*api.Node, error)
type predicateAdmitHandler struct {
	getNodeAnyWayFunc getNodeAnyWayFuncType
}

var _ PodAdmitHandler = &predicateAdmitHandler{}

func NewPredicateAdmitHandler(getNodeAnyWayFunc getNodeAnyWayFuncType) *predicateAdmitHandler {
	return &predicateAdmitHandler{
		getNodeAnyWayFunc,
	}
}

func (w *predicateAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	node, err := w.getNodeAnyWayFunc()
	if err != nil {
		glog.Errorf("Cannot get Node info: %v", err)
		return PodAdmitResult{
			Admit:   false,
			Reason:  "InvalidNodeInfo",
			Message: "Kubelet cannot get node info.",
		}
	}
	pod := attrs.Pod
	pods := attrs.OtherPods
	nodeInfo := schedulercache.NewNodeInfo(pods...)
	nodeInfo.SetNode(node)
	fit, reasons, err := predicates.GeneralPredicates(pod, nil, nodeInfo)
	if err != nil {
		message := fmt.Sprintf("GeneralPredicates failed due to %v, which is unexpected.", err)
		glog.Warningf("Failed to admit pod %v - %s", format.Pod(pod), message)
		return PodAdmitResult{
			Admit:   fit,
			Reason:  "UnexpectedError",
			Message: message,
		}
	}
	if !fit {
		var reason string
		var message string
		if len(reasons) == 0 {
			message = fmt.Sprint("GeneralPredicates failed due to unknown reason, which is unexpected.")
			glog.Warningf("Failed to admit pod %v - %s", format.Pod(pod), message)
			return PodAdmitResult{
				Admit:   fit,
				Reason:  "UnknownReason",
				Message: message,
			}
		}
		// If there are failed predicates, we only return the first one as a reason.
		r := reasons[0]
		switch re := r.(type) {
		case *predicates.PredicateFailureError:
			reason = re.PredicateName
			message = re.Error()
			glog.V(2).Infof("Predicate failed on Pod: %v, for reason: %v", format.Pod(pod), message)
		case *predicates.InsufficientResourceError:
			reason = fmt.Sprintf("OutOf%s", re.ResourceName)
			message = re.Error()
			glog.V(2).Infof("Predicate failed on Pod: %v, for reason: %v", format.Pod(pod), message)
		case *predicates.FailureReason:
			reason = re.GetReason()
			message = fmt.Sprintf("Failure: %s", re.GetReason())
			glog.V(2).Infof("Predicate failed on Pod: %v, for reason: %v", format.Pod(pod), message)
		default:
			reason = "UnexpectedPredicateFailureType"
			message = fmt.Sprintf("GeneralPredicates failed due to %v, which is unexpected.", r)
			glog.Warningf("Failed to admit pod %v - %s", format.Pod(pod), message)
		}
		return PodAdmitResult{
			Admit:   fit,
			Reason:  reason,
			Message: message,
		}
	}
	return PodAdmitResult{
		Admit: true,
	}
}
