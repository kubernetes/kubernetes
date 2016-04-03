// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package discovery

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/appc/spec/schema/common"
	"github.com/appc/spec/schema/types"
)

type App struct {
	Name   types.ACIdentifier
	Labels map[types.ACIdentifier]string
}

func NewApp(name string, labels map[types.ACIdentifier]string) (*App, error) {
	if labels == nil {
		labels = make(map[types.ACIdentifier]string, 0)
	}
	acn, err := types.NewACIdentifier(name)
	if err != nil {
		return nil, err
	}
	return &App{
		Name:   *acn,
		Labels: labels,
	}, nil
}

// NewAppFromString takes a command line app parameter and returns a map of labels.
//
// Example app parameters:
// 	example.com/reduce-worker:1.0.0
// 	example.com/reduce-worker,channel=alpha,label=value
// 	example.com/reduce-worker:1.0.0,label=value
//
// As can be seen in above examples - colon, comma and equal sign have
// special meaning. If any of them has to be a part of a label's value
// then consider writing your own string to App parser.
func NewAppFromString(app string) (*App, error) {
	var (
		name   string
		labels map[types.ACIdentifier]string
	)

	preparedApp, err := prepareAppString(app)
	if err != nil {
		return nil, err
	}
	v, err := url.ParseQuery(preparedApp)
	if err != nil {
		return nil, err
	}
	labels = make(map[types.ACIdentifier]string, 0)
	for key, val := range v {
		if len(val) > 1 {
			return nil, fmt.Errorf("label %s with multiple values %q", key, val)
		}
		if key == "name" {
			name = val[0]
			continue
		}
		labelName, err := types.NewACIdentifier(key)
		if err != nil {
			return nil, err
		}
		labels[*labelName] = val[0]
	}
	a, err := NewApp(name, labels)
	if err != nil {
		return nil, err
	}
	return a, nil
}

func prepareAppString(app string) (string, error) {
	if err := checkColon(app); err != nil {
		return "", err
	}

	app = "name=" + strings.Replace(app, ":", ",version=", 1)
	return common.MakeQueryString(app)
}

func checkColon(app string) error {
	firstComma := strings.IndexRune(app, ',')
	firstColon := strings.IndexRune(app, ':')
	if firstColon > firstComma && firstComma > -1 {
		return fmt.Errorf("malformed app string - colon may appear only right after the app name")
	}
	if strings.Count(app, ":") > 1 {
		return fmt.Errorf("malformed app string - colon may appear at most once")
	}
	return nil
}

func (a *App) Copy() *App {
	ac := &App{
		Name:   a.Name,
		Labels: make(map[types.ACIdentifier]string, 0),
	}
	for k, v := range a.Labels {
		ac.Labels[k] = v
	}
	return ac
}

// String returns the URL-like image name
func (a *App) String() string {
	img := a.Name.String()
	for n, v := range a.Labels {
		img += fmt.Sprintf(",%s=%s", n, v)
	}
	return img
}
