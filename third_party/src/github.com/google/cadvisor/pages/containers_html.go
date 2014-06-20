// Copyright 2014 Google Inc. All Rights Reserved.
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

package pages

const containersHtmlTemplate = `
<html>
<head>
  <title>cAdvisor - Container {{.ContainerName}}</title>
  <!-- Latest compiled and minified CSS -->
  <link rel="stylesheet" href="//netdna.bootstrapcdn.com/bootstrap/3.1.1/css/bootstrap.min.css">

  <!-- Optional theme -->
  <link rel="stylesheet" href="//netdna.bootstrapcdn.com/bootstrap/3.1.1/css/bootstrap-theme.min.css">

  <link rel="stylesheet" href="/static/containers.css">

  <!-- Latest compiled and minified JavaScript -->
  <script src="//ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"></script>
  <script src="//netdna.bootstrapcdn.com/bootstrap/3.1.1/js/bootstrap.min.js"></script>
  <script type="text/javascript" src="https://www.google.com/jsapi"></script>

  <script type="text/javascript" src="/static/containers.js"></script>
</head>
<body>
<div class="container theme-showcase" >
  <div class="col-sm-12" id="logo">
  </div>
  <div class="col-sm-12">
  <div class="page-header">
    <h1>{{.ContainerName}}</h1>
  </div>
  <ol class="breadcrumb">
    {{range $parentContainer := .ParentContainers}}
      <li>{{containerLink $parentContainer true ""}}</li>
    {{end}}
  </ol>
  </div>
  {{if .Subcontainers}}
  <div class="col-sm-12">
    <div class="page-header">
      <h3>Subcontainers</h3>
    </div>
    <div class="list-group">
      {{range $subcontainer := .Subcontainers}}
        {{containerLink $subcontainer false "list-group-item"}}
      {{end}}
    </div>
  </div>
  {{end}}
  {{if .ResourcesAvailable}}
  <div class="col-sm-12">
    <div class="page-header">
      <h3>Isolation</h3>
    </div>
    {{if .CpuAvailable}}
      <ul class="list-group">
        <li class="list-group-item active isolation-title panel-title">CPU</li>
        {{if .Spec.Cpu.Limit}}
          <li class="list-group-item"><span class="stat-label">Limit</span> {{printCores .Spec.Cpu.Limit}} <span class="unit-label">cores</span></li>
        {{end}}
        {{if .Spec.Cpu.MaxLimit}}
          <li class="list-group-item"><span class="stat-label">Max Limit</span> {{printCores .Spec.Cpu.MaxLimit}} <span class="unit-label">cores</span></li>
        {{end}}
        {{if .Spec.Cpu.Mask}}
          <li class="list-group-item"><span class="stat-label">Allowed Cores</span> {{printMask .Spec.Cpu.Mask .MachineInfo.NumCores}}</li>
        {{end}}
      </ul>
    {{end}}
    {{if .MemoryAvailable}}
      <ul class="list-group">
        <li class="list-group-item active isolation-title panel-title">Memory</li>
        {{if .Spec.Memory.Reservation}}
          <li class="list-group-item"><span class="stat-label">Reservation</span> {{printMegabytes .Spec.Memory.Reservation}} <span class="unit-label">MB</span></li>
        {{end}}
        {{if .Spec.Memory.Limit}}
          <li class="list-group-item"><span class="stat-label">Limit</span> {{printMegabytes .Spec.Memory.Limit}} <span class="unit-label">MB</span></li>
        {{end}}
        {{if .Spec.Memory.SwapLimit}}
          <li class="list-group-item"><span class="stat-label">Swap Limit</span> {{printMegabytes .Spec.Memory.SwapLimit}} <span class="unit-label">MB</span></li>
        {{end}}
      </ul>
    {{end}}
  </div>
  <div class="col-sm-12">
    <div class="page-header">
      <h3>Usage</h3>
    </div>
      <div class="panel panel-primary">
        <div class="panel-heading">
          <h3 class="panel-title">Overview</h3>
        </div>
        <div id="usage-gauge" class="panel-body">
        </div>
      </div>
    {{if .CpuAvailable}}
      <div class="panel panel-primary">
        <div class="panel-heading">
          <h3 class="panel-title">CPU</h3>
        </div>
        <div class="panel-body">
          <h4>Total Usage</h4>
	  <div id="cpu-total-usage-chart"></div>
          <h4>Usage per Core</h4>
	  <div id="cpu-per-core-usage-chart"></div>
          <h4>Usage Breakdown</h4>
	  <div id="cpu-usage-breakdown-chart"></div>
        </div>
      </div>
    {{end}}
    {{if .MemoryAvailable}}
      <div class="panel panel-primary">
        <div class="panel-heading">
          <h3 class="panel-title">Memory</h3>
        </div>
        <div class="panel-body">
          <h4>Total Usage</h4>
	  <div id="memory-usage-chart"></div>
          <br/>
          <div class="row col-sm-12">
            <h4>Usage Breakdown</h4>
            <div class="col-sm-9">
              <div class="progress">
                <div class="progress-bar progress-bar-danger" style="width: {{getHotMemoryPercent .Spec .Stats .MachineInfo}}%">
                  <span class="sr-only">Hot Memory</span>
                </div>
                <div class="progress-bar progress-bar-info" style="width: {{getColdMemoryPercent .Spec .Stats .MachineInfo}}%">
                  <span class="sr-only">Cold Memory</span>
                </div>
              </div>
            </div>
            <div class="col-sm-3">
              {{ getMemoryUsage .Stats }} MB ({{ getMemoryUsagePercent .Spec .Stats .MachineInfo}}%)
            </div>
	  </div>
          <h4>Page Faults</h4>
	  <div id="memory-page-faults-chart"></div>
        </div>
      </div>
    {{end}}
  </div>
  {{end}}
</div>
<script type="text/javascript">
  startPage({{.ContainerName}}, {{.CpuAvailable}}, {{.MemoryAvailable}});
</script>
</body>
</html>
`
