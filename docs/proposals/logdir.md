<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/proposals/logdir.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# LogDir

## Abstract

A proposal for implementing `LogDir`. A `LogDir` can preserve and manage a pod's log files.

Several existing issues and PRs were already created regarding that particular subject:
* Collect logfiles inside the container [#12567](https://github.com/kubernetes/kubernetes/issues/12567)
* Add LogDir [#13010](https://github.com/kubernetes/kubernetes/pull/13010#issuecomment-133997188)

Author: WuLonghui (@wulonghui)

## Motivation

Some applications will write log to files, so need to add  `LogDir`. It can add some useful infomation to make logging agent easy to collect the logs, also we can manage the log files(like log rotation and maximum log file sizes).

## Implementation

Create `LogDir` as a policy  to `api.VolumeMount` struct :

```
// VolumeMount describes a mounting of a Volume within a container.
type VolumeMount struct {
  // Required: This must match the Name of a Volume [above].
  Name string `json:"name"`
  // Optional: Defaults to false (read-write).
  ReadOnly bool `json:"readOnly,omitempty"`
  // Required.
  MountPath string `json:"mountPath"`
  // Optional: Policies of VolumeMount.
  Policy *VolumeMountPolicy `json:"policy,omitempty"`
}

// VolumeMountPolicy describes policies of a VolumeMount.
type VolumeMountPolicy struct {
  // Optional: LogDir policy.
  LogDir *LogDirPolicy `json:"logDir,omitempty"`
}

// LogDirPolicy describes a policy of logDir, include log rotation and maximum log file size.
type LogDirPolicy struct {
  // Optional: Glob pattern of log files.
  Glob string `json:"glob,omitempty"`
  // Optional: Log rotation.
  Rotate string `json:"rotate,omitempty"`
  // Optional: Maximum log file size.
  MaxFileSize int `json:"maxFileSize,omitempty"`
}

```

If users set the LogDir policy of the VolumeMount in a container:

```
apiVersion: v1
kind: Pod
metadata:
  name: my-app
spec:
  containers:
  - name: container1
    image: ubuntu:14.04
    command: ["bash", "-c", "i=\"0\"; while true; do echo \"`hostname`_container1: $i \"; date --rfc-3339 ns >> /varlog/container1.log; sleep 4; i=$[$i+1]; done"]
    volumeMounts:
    - name: log-storage
      mountPath: /varlog
      policy:
        logDir: {}
    securityContext:
        privileged: true
  volumes:
  - name: log-storage
    emptyDir: {}
```

Kubelet will create a symbolic link to the volume path in `/var/log/containers`.

```
/var/log/containers/<pod_full_name>_<contianer_name> => <volume_path>
```

Then the logging agent(e.g.Fluentd) can watch `/var/log/containers` on host to collect log files, add tag by `<pod_full_name>_<contianer_name>`, which can be used for search terms in Elasticsearch or for
labels for Cloud Logging.


## Integrated with Fluentd

We can use Fluentd to collect log files in `LogDir`. Fluentd should be installed on each Kubernetes node, and it will watch LogDir `/var/log/containers`, the Fluentd's configuration as follows:

```
<source>
  type tail
  format none
  time_key time
  path /var/log/containers/*/**/*.log
  pos_file /lib/pods.log.pos
  time_format %Y-%m-%dT%H:%M:%S
  tag reform.*
  read_from_head true
</source>

<match reform.**>
  type record_reformer
  enable_ruby true
  tag kubernetes.${hostname}.${tag_suffix[4]}
</match>

<match **>
  type stdout
</match>
```

The Fluentd will tail any files in the LogDir `/var/log/containers`, and add tag `kubernetes.<node_host_name>.<pod_full_name>_<container_name>.<file_name>`. We only print the logs to stdout,  also can forward to logging storage endpoint(e.g Elasticsearch).

Then the Fluentd prints the logs to stdout:

```
2015-09-11 11:00:10 +0000 kubernetes.8f5cd4af528a.my-app_default_container1.container1.log: {"message":"2015-09-11 10:59:53.331748730+00:00"}
2015-09-11 11:00:10 +0000 kubernetes.8f5cd4af528a.my-app_default_container1.container1.log: {"message":"2015-09-11 10:59:57.335719322+00:00"}
2015-09-11 11:00:10 +0000 kubernetes.8f5cd4af528a.my-app_default_container1.container1.log: {"message":"2015-09-11 11:00:01.339536181+00:00"}
```

## Future work

*  Be able to limit maximum log file sizes
*  Be able to support log rotation

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/logdir.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
