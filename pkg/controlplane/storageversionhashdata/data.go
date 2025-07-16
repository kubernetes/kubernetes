/*
Copyright 2019 The Kubernetes Authors.

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

// Package storageversionhashdata is for test only.
package storageversionhashdata

import (
	"k8s.io/apimachinery/pkg/util/sets"
)

// NoStorageVersionHash lists resources that legitimately with empty storage
// version hash.
var NoStorageVersionHash = sets.NewString(
	"v1/bindings",
	"v1/componentstatuses",
	"authentication.k8s.io/v1/selfsubjectreviews",
	"authentication.k8s.io/v1/tokenreviews",
	"authorization.k8s.io/v1/localsubjectaccessreviews",
	"authorization.k8s.io/v1/selfsubjectaccessreviews",
	"authorization.k8s.io/v1/selfsubjectrulesreviews",
	"authorization.k8s.io/v1/subjectaccessreviews",
)

// GVRToStorageVersionHash shouldn't change unless we intentionally change the
// storage version of a resource.
var GVRToStorageVersionHash = map[string]string{
	"v1/configmaps":             "qFsyl6wFWjQ=",
	"v1/endpoints":              "fWeeMqaN/OA=",
	"v1/events":                 "r2yiGXH7wu8=",
	"v1/limitranges":            "EBKMFVe6cwo=",
	"v1/namespaces":             "Q3oi5N2YM8M=",
	"v1/nodes":                  "XwShjMxG9Fs=",
	"v1/persistentvolumeclaims": "QWTyNDq0dC4=",
	"v1/persistentvolumes":      "HN/zwEC+JgM=",
	"v1/pods":                   "xPOwRZ+Yhw8=",
	"v1/podtemplates":           "LIXB2x4IFpk=",
	"v1/replicationcontrollers": "Jond2If31h0=",
	"v1/resourcequotas":         "8uhSgffRX6w=",
	"v1/secrets":                "S6u1pOWzb84=",
	"v1/serviceaccounts":        "pbx9ZvyFpBE=",
	"v1/services":               "0/CO1lhkEBI=",
	"autoscaling/v1/horizontalpodautoscalers": "qwQve8ut294=",
	"autoscaling/v2/horizontalpodautoscalers": "qwQve8ut294=",
	"batch/v1/jobs":     "mudhfqk/qZY=",
	"batch/v1/cronjobs": "sd5LIXh4Fjs=",
	"certificates.k8s.io/v1/certificatesigningrequests":                 "95fRKMXA+00=",
	"coordination.k8s.io/v1/leases":                                     "gqkMMb/YqFM=",
	"discovery.k8s.io/v1/endpointslices":                                "qgS0xkrxYAI=",
	"networking.k8s.io/v1/networkpolicies":                              "YpfwF18m1G8=",
	"networking.k8s.io/v1/ingresses":                                    "39NQlfNR+bo=",
	"networking.k8s.io/v1/ingressclasses":                               "l/iqIbDgFyQ=",
	"networking.k8s.io/v1/ipaddresses":                                  "O4H8VxQhW5Y=",
	"networking.k8s.io/v1/servicecidrs":                                 "8ufAXOnr3Yg=",
	"node.k8s.io/v1/runtimeclasses":                                     "WQTu1GL3T2Q=",
	"policy/v1/poddisruptionbudgets":                                    "EVWiDmWqyJw=",
	"rbac.authorization.k8s.io/v1/clusterrolebindings":                  "48tpQ8gZHFc=",
	"rbac.authorization.k8s.io/v1/clusterroles":                         "bYE5ZWDrJ44=",
	"rbac.authorization.k8s.io/v1/rolebindings":                         "eGsCzGH6b1g=",
	"rbac.authorization.k8s.io/v1/roles":                                "7FuwZcIIItM=",
	"scheduling.k8s.io/v1/priorityclasses":                              "1QwjyaZjj3Y=",
	"storage.k8s.io/v1/csidrivers":                                      "hL6j/rwBV5w=",
	"storage.k8s.io/v1/csinodes":                                        "Pe62DkZtjuo=",
	"storage.k8s.io/v1/storageclasses":                                  "K+m6uJwbjGY=",
	"storage.k8s.io/v1/csistoragecapacities":                            "xeVl+2Ly1kE=",
	"storage.k8s.io/v1/volumeattachments":                               "tJx/ezt6UDU=",
	"apps/v1/controllerrevisions":                                       "85nkx63pcBU=",
	"apps/v1/daemonsets":                                                "dd7pWHUlMKQ=",
	"apps/v1/deployments":                                               "8aSe+NMegvE=",
	"apps/v1/replicasets":                                               "P1RzHs8/mWQ=",
	"apps/v1/statefulsets":                                              "H+vl74LkKdo=",
	"admissionregistration.k8s.io/v1/mutatingwebhookconfigurations":     "Sqi0GUgDaX0=",
	"admissionregistration.k8s.io/v1/validatingwebhookconfigurations":   "B0wHjQmsGNk=",
	"admissionregistration.k8s.io/v1/validatingadmissionpolicies":       "6OxvlMmQ6is=",
	"admissionregistration.k8s.io/v1/validatingadmissionpolicybindings": "v9715VZqakg=",
	"events.k8s.io/v1/events":                                           "r2yiGXH7wu8=",
	"flowcontrol.apiserver.k8s.io/v1/flowschemas":                       "GJVAJZSZBIw=",
	"flowcontrol.apiserver.k8s.io/v1/prioritylevelconfigurations":       "Kir5PVfvNeI=",
}
