<!-- markdownlint-disable first-line-h1 no-inline-html -->

[![Build](https://github.com/vmware/govmomi/actions/workflows/govmomi-build.yaml/badge.svg)][ci-build]
[![Tests](https://github.com/vmware/govmomi/actions/workflows/govmomi-go-tests.yaml/badge.svg)][ci-tests]
[![Go Report Card](https://goreportcard.com/badge/github.com/vmware/govmomi)][go-report-card]
[![Latest Release](https://img.shields.io/github/release/vmware/govmomi.svg?logo=github&style=flat-square)][latest-release]
[![Go Reference](https://pkg.go.dev/badge/github.com/vmware/govmomi.svg)][go-reference]
[![go.mod Go version](https://img.shields.io/github/go-mod/go-version/vmware/govmomi)][go-version]

# govmomi

A Go library for interacting with VMware vSphere APIs (ESXi and/or vCenter Server).

In addition to the vSphere API client, this repository includes:

* [govc][govc] - vSphere CLI
* [vcsim][vcsim] - vSphere API mock framework
* [toolbox][toolbox] - VM guest tools framework

## Compatibility

This library supports vCenter Server and ESXi versions following the [VMware Product Lifecycle Matrix][reference-lifecycle].

Product versions that are end of support may work, but are not officially supported.

## Documentation

The APIs exposed by this library closely follow the API described in the [VMware vSphere API Reference Documentation][reference-api]. Refer to the documentation to become familiar with the upstream API.

The code in the `govmomi` package is a wrapper for the code that is generated from the vSphere API description. It primarily provides convenience functions for working with the vSphere API. See [godoc.org][reference-godoc] for documentation.

## Installation

### govmomi (Package)

```bash
go get -u github.com/vmware/govmomi
```

### Binaries and Docker Images for `govc` and `vcsim`

Installation instructions, released binaries, and Docker images are documented in the respective README files of [`govc`][govc] and [`vcsim`][vcsim].

## Discussion

The project encourages the community to collaborate using GitHub [issues][govmomi-github-issues], GitHub [discussions][govmomi-github-discussions], and [Slack][slack-channel].

> **Note**
> Access to Slack requires a free [VMware {code}][slack-join] developer program membership.

## Status

Changes to the API are subject to [semantic versioning][reference-semver].

Refer to the [CHANGELOG][govmomi-changelog] for version to version changes.

## Notable Projects Using govmomi

* [collectd-vsphere][project-travisci-collectd-vsphere]
* [Docker LinuxKit][project-docker-linuxKit]
* [Elastic Agent VMware vSphere integration][project-elastic-agent]
* [Gru][project-gru]
* [Juju][project-juju]
* [Jupiter Brain][project-travisci-jupiter-brain]
* [Kubernetes vSphere Cloud Provider][project-k8s-cloud-provider]
* [Kubernetes Cluster API][project-k8s-cluster-api]
* [OPS][project-nanovms-ops]
* [Packer Plugin for VMware vSphere][project-hashicorp-packer-plugin-vsphere]
* [Rancher][project-rancher]
* [Terraform Provider for VMware vSphere][project-hashicorp-terraform-provider-vsphere]
* [Telegraf][project-influxdata-telegraf]
* [VMware Event Broker Appliance][project-vmware-veba]
* [VMware vSphere Integrated Containers Engine][project-vmware-vic]
* [VMware vSphere 7.0][project-vmware-vsphere]

## Related Projects

* [go-vmware-nsxt][reference-go-vmware-nsxt]
* [pyvmomi][reference-pyvmomi]
* [rbvmomi][reference-rbvmomi]

## License

govmomi is available under the [Apache 2 License][govmomi-license].

## Name

Pronounced: _go·​v·​mom·​ie_

Follows pyvmomi and rbvmomi: language prefix + the vSphere acronym "VM Object Management Infrastructure".

[//]: Links

[ci-build]: https://github.com/vmware/govmomi/actions/workflows/govmomi-build.yaml
[ci-tests]: https://github.com/vmware/govmomi/actions/workflows/govmomi-go-tests.yaml
[latest-release]: https://github.com/vmware/govmomi/releases/latest
[govc]: govc/README.md
[govmomi-github-issues]: https://github.com/vmware/govmomi/issues
[govmomi-github-discussions]: https://github.com/vmware/govmomi/discussions
[govmomi-changelog]: CHANGELOG.md
[govmomi-license]: LICENSE.txt
[go-reference]: https://pkg.go.dev/github.com/vmware/govmomi
[go-report-card]: https://goreportcard.com/report/github.com/vmware/govmomi
[go-version]: https://github.com/vmware/govmomi
[project-docker-linuxKit]: https://github.com/linuxkit/linuxkit/tree/master/src/cmd/linuxkit
[project-elastic-agent]: https://github.com/elastic/integrations/tree/main/packages/vsphere
[project-gru]: https://github.com/dnaeon/gru
[project-hashicorp-packer-plugin-vsphere]: https://github.com/hashicorp/packer-plugin-vsphere
[project-hashicorp-terraform-provider-vsphere]: https://github.com/hashicorp/terraform-provider-vsphere
[project-influxdata-telegraf]: https://github.com/influxdata/telegraf/tree/master/plugins/inputs/vsphere
[project-juju]: https://github.com/juju/juju
[project-k8s-cloud-provider]: https://github.com/kubernetes/cloud-provider-vsphere
[project-k8s-cluster-api]: https://github.com/kubernetes-sigs/cluster-api-provider-vsphere
[project-nanovms-ops]: https://github.com/nanovms/ops
[project-rancher]: https://github.com/rancher/rancher/blob/master/pkg/api/norman/customization/vsphere/listers.go
[project-travisci-collectd-vsphere]: https://github.com/travis-ci/collectd-vsphere
[project-travisci-jupiter-brain]: https://github.com/travis-ci/jupiter-brain
[project-vmware-veba]: https://github.com/vmware-samples/vcenter-event-broker-appliance/tree/development/vmware-event-router
[project-vmware-vic]: https://github.com/vmware/vic
[project-vmware-vsphere]: https://docs.vmware.com/en/VMware-vSphere/7.0/rn/vsphere-esxi-vcenter-server-7-vsphere-with-kubernetes-release-notes.html
[reference-api]: https://developer.vmware.com/apis/968/vsphere
[reference-godoc]: http://godoc.org/github.com/vmware/govmomi
[reference-go-vmware-nsxt]: https://github.com/vmware/go-vmware-nsxt
[reference-lifecycle]: https://lifecycle.vmware.com
[reference-pyvmomi]: https://github.com/vmware/pyvmomi
[reference-rbvmomi]: https://github.com/vmware/rbvmomi
[reference-semver]: http://semver.org
[slack-join]: https://developer.vmware.com/join/
[slack-channel]: https://vmwarecode.slack.com/messages/govmomi
[toolbox]: toolbox/README.md
[vcsim]: vcsim/README.md
