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

package master

const (
	DummyDeployment = `
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  labels:
    app: dummy
  name: dummy
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dummy
  template:
    metadata:
      labels:
        app: dummy
    spec:
      containers:
      - image: {{ .ImageRepository }}/pause-{{ .Arch }}:3.0
        name: dummy
      hostNetwork: true
`
)
