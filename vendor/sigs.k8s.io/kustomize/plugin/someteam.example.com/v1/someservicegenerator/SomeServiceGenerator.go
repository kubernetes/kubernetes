// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"text/template"

	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/yaml"
)

// A simple generator example.  Makes one service.
type plugin struct {
	rf   *resmap.Factory
	Name string `json:"name,omitempty" yaml:"name,omitempty"`
	Port string `json:"port,omitempty" yaml:"port,omitempty"`
}

//noinspection GoUnusedGlobalVariable
//nolint: golint
var KustomizePlugin plugin

const tmpl = `
apiVersion: v1
kind: Service
metadata:
  labels:
    app: dev
  name: {{.Name}}
spec:
  ports:
  - port: {{.Port}}
  selector:
    app: dev
`

func (p *plugin) Config(
	_ ifc.Loader, rf *resmap.Factory, config []byte) error {
	p.rf = rf
	return yaml.Unmarshal(config, p)
}

func (p *plugin) Generate() (resmap.ResMap, error) {
	var buf bytes.Buffer
	temp := template.Must(template.New("tmpl").Parse(tmpl))
	err := temp.Execute(&buf, p)
	if err != nil {
		return nil, err
	}
	return p.rf.NewResMapFromBytes(buf.Bytes())
}
