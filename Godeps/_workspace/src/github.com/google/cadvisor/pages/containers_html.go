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
    <title>cAdvisor - {{.DisplayName}}</title>
    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="{{.Root}}static/bootstrap-3.1.1.min.css">

    <!-- Optional theme -->
    <link rel="stylesheet" href="{{.Root}}static/bootstrap-theme-3.1.1.min.css">

    <link rel="stylesheet" href="{{.Root}}static/containers.css">

    <!-- Latest compiled and minified JavaScript -->
    <script src="{{.Root}}static/jquery-1.10.2.min.js"></script>
    <script src="{{.Root}}static/bootstrap-3.1.1.min.js"></script>
    <script type="text/javascript" src="{{.Root}}static/google-jsapi.js"></script>

    <script type="text/javascript" src="{{.Root}}static/containers.js"></script>
  </head>
  <body>
    <div class="container theme-showcase" >
      <a href="{{.Root}}" class="col-sm-12" id="logo">
      </a>
      <div class="col-sm-12">
	<div class="page-header">
	  <h1>{{.DisplayName}}</h1>
	</div>
	<ol class="breadcrumb">
	  {{range $parentContainer := .ParentContainers}}
	  <li><a href="{{$parentContainer.Link}}">{{$parentContainer.Text}}</a></li>
	  {{end}}
	</ol>
      </div>
      {{if .IsRoot}}
      <div class="col-sm-12">
        <h4><a href="../docker">Docker Containers</a></h4>
      </div>
      {{end}}
      {{if .Subcontainers}}
      <div class="col-sm-12">
	<div class="page-header">
	  <h3>Subcontainers</h3>
	</div>
	<div class="list-group">
	  {{range $subcontainer := .Subcontainers}}
	  <a href="{{$subcontainer.Link}}" class="list-group-item">{{$subcontainer.Text}}</a>
	  {{end}}
	</div>
      </div>
      {{end}}
     {{if .DockerStatus}}
      <div class="col-sm-12">
	<div class="page-header">
	  <h3>Driver Status</h3>
	</div>
	<ul class="list-group">
	  {{range $dockerstatus := .DockerStatus}}
	  <li class ="list-group-item"><span class="stat-label">{{$dockerstatus.Key}}</span> {{$dockerstatus.Value}}</li>
	  {{end}}
	  {{if .DockerDriverStatus}}
		<li class ="list-group-item"><span class="stat-label">Storage<br></span>
		<ul class="list-group">
		{{range $driverstatus := .DockerDriverStatus}}
		<li class="list-group-item"><span class="stat-label">{{$driverstatus.Key}}</span> {{$driverstatus.Value}}</li>
		{{end}}
		</ul>
		</li>
	  </ul>
	  {{end}}
	</div>
      {{end}}
      {{if .DockerImages}}
      <div class="col-sm-12">
          <div class="page-header">
            <h3>Images</h3>
          </div>
       <div id="docker-images"></div>
       <br><br>
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
          <li class="list-group-item"><span class="stat-label">Shares</span> {{printShares .Spec.Cpu.Limit}} <span class="unit-label">shares</span></li>
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
          <li class="list-group-item"><span class="stat-label">Reservation</span> {{printSize .Spec.Memory.Reservation}} <span class="unit-label">{{printUnit .Spec.Memory.Reservation}}</span></li>
          {{end}}
          {{if .Spec.Memory.Limit}}
          <li class="list-group-item"><span class="stat-label">Limit</span> {{printSize .Spec.Memory.Limit}} <span class="unit-label">{{printUnit .Spec.Memory.Limit}}</span></li>
          {{end}}
          {{if .Spec.Memory.SwapLimit}}
          <li class="list-group-item"><span class="stat-label">Swap Limit</span> {{printSize .Spec.Memory.SwapLimit}} <span class="unit-label">{{printUnit .Spec.Memory.SwapLimit}}</span></li>
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
          <div id="usage-gauge" class="panel-body"></div>
	</div>
	<div class="panel panel-primary">
          <div class="panel-heading">
            <h3 class="panel-title">Processes</h3>
          </div>
          <div id="processes-top" class="panel-body"></div>
	</div>
	{{if .CpuAvailable}}
	<div class="panel panel-primary">
          <div class="panel-heading">
            <h3 class="panel-title">CPU</h3>
          </div>
          <div class="panel-body">
            <h4>Total Usage</h4>
	    <div id="cpu-total-usage-chart"></div>
	    <!-- <h4>CPU Load Average</h4>
	    <div id="cpu-load-chart"></div> -->
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
                  <div class="progress-bar progress-bar-danger" id="progress-hot-memory">
                    <span class="sr-only">Hot Memory</span>
                  </div>
                  <div class="progress-bar progress-bar-info" id="progress-cold-memory">
                    <span class="sr-only">Cold Memory</span>
                  </div>
		</div>
              </div>
              <div class="col-sm-3" id="memory-text"></div>
	    </div>
          </div>
	</div>
	{{end}}
	{{if .NetworkAvailable}}
	<div class="panel panel-primary">
	  <div class="panel-heading">
            <h3 class="panel-title">Network</h3>
	  </div>
	  <div class="panel-body">
            <h4>Throughput</h4>
            <div id="network-bytes-chart"></div>
	  </div>
	  <div class="panel-body">
            <h4>Errors</h4>
	    <div id="network-errors-chart"></div>
	  </div>
	</div>
        {{end}}
	{{if .FsAvailable}}
	<div class="panel panel-primary">
          <div class="panel-heading">
            <h3 class="panel-title">Filesystem</h3>
          </div>
          <div id="filesystem-usage" class="panel-body">
          </div>
        </div>
	{{end}}
      </div>
      {{end}}
    </div>
    <script type="text/javascript">
      startPage({{.ContainerName}}, {{.CpuAvailable}}, {{.MemoryAvailable}}, {{.Root}}, {{.IsRoot}});
      drawImages({{.DockerImages}});
    </script>
  </body>
</html>
`
