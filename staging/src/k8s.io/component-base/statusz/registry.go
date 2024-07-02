/*
Copyright 2024 The Kubernetes Authors.

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

package statusz

import (
	"context"
	"fmt"
	"html/template"
	"io"
	"sync"
	"time"
)

type statuszRegistry struct {
	lock     sync.Mutex
	options  Options
	sections []section
}

type section struct {
	Title string
	Func  func(context.Context, io.Writer) error
}

func Register(opts Options) *statuszRegistry {
	registry := &statuszRegistry{
		options: opts,
	}
	registry.addSection("default", defaultSection)
	return registry
}

func (reg *statuszRegistry) addSection(title string, f func(ctx context.Context, wr io.Writer, opts Options) error) error {
	reg.lock.Lock()
	defer reg.lock.Unlock()
	reg.sections = append(reg.sections, section{
		Title: title,
		Func: func(ctx context.Context, wr io.Writer) error {
			err := f(ctx, wr, reg.options)
			if err != nil {
				failErr := template.Must(template.New("").Parse("<code>invalid HTML: {{.}}</code>")).Execute(wr, err)
				if failErr != nil {
					return fmt.Errorf("go/server: couldn't execute the error template for %q: %v (couldn't get HTML fragment: %v)", title, failErr, err)
				}
				return err
			}
			return nil
		},
	})
	return nil
}

func defaultSection(ctx context.Context, wr io.Writer, opts Options) error {
	var data struct {
		ServerName string
		StartTime  string
		Uptime     string
	}

	data.ServerName = opts.ComponentName
	data.StartTime = opts.StartTime.Format(time.RFC1123)
	uptime := int64(time.Since(opts.StartTime).Seconds())
	data.Uptime = fmt.Sprintf("%d hr %02d min %02d sec",
		uptime/3600, (uptime/60)%60, uptime%60)

	if err := defaultTmp.Execute(wr, data); err != nil {
		return fmt.Errorf("couldn't execute template: %v", err)
	}

	return nil
}

var defaultTmp = template.Must(template.New("").Parse(`
<!DOCTYPE html>
<html>
<head>
	<title>Status for {{.ServerName}}</title>
		<style>
		body {
			font-family: sans-serif;
		}
		h1 {
			clear: both;
			width: 100%;
			text-align: center;
			font-size: 120%;
			background: #eef;
		}
		.lefthand {
			float: left;
			width: 80%;
		}
		.righthand {
			text-align: right;
		}
		td {
		  background-color: rgba(0, 0, 0, 0.05);
		}
	</style>
</head>

<body>
<h1>Status for {{.ServerName}}</h1>

<div>
	<div class=lefthand>
		Started: {{.StartTime}}<br>
		Up {{.Uptime}}<br>
</body>
</html>
`))
