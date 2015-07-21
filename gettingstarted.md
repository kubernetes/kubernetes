---
layout: gettingstarted
title: Getting Started
permalink: /gettingstarted/
show_in_nav: true
slug: gettingstarted

hero:
    title: Getting Started
    text: Get started running, deploying, and using Kubernetes.
    img: /img/desktop/getting_started/hero_icon.svg

steps:
  - title: Hosted Services
    text: "Don't want to go through the hassle of setting up your own cluster and the infrastructure associated with it? These services offer managed Kubernetes to make it that much easier to get going."
    slug: hosted_services
  - title: Installation
    text: "First, lets get you up and running by starting our first Kubernetes cluster. Kubernetes can run almost anywhere so choose the configuration you're most comfortable with:"
    slug: installation
  - title: Your First Application
    text: "Now we're ready to run our first real application! A simple multi-tiered guestbook."
    slug: first_app
  - title: Releases
    text: Releases of Kubernetes
    slug: releases
  - title: Technical Details
    text: "Interested in taking a peek inside Kubernetes? You should start by reading the <a href=\"v1.0/docs/design/README.html\" onclick=\"trackOutboundLink('v1.0/docs/design/README.html'); return false;\">design overview</a> which introduces core Kubernetes concepts and components. After that, you probably want to take a look at the API documentation and learn about the kubecfg command line tool."
    slug: techdetails

hostedservices:
  - label: Google Container Engine
    url: https://cloud.google.com/container-engine/

installguides:
  - label: Google Compute Engine
    url: v1.0/docs/getting-started-guides/gce.html
  - label: Vagrant
    url: v1.0/docs/getting-started-guides/vagrant.html
  - label: Fedora (Ansible)
    url: v1.0/docs/getting-started-guides/fedora/fedora_ansible_config.html
  - label: Fedora (Manual)
    url: v1.0/docs/getting-started-guides/fedora/fedora_manual_config.html
  - label: Local
    url: v1.0/docs/getting-started-guides/locally.html
  - label: Microsoft Azure
    url: v1.0/docs/getting-started-guides/azure.html
  - label: Rackspace
    url: v1.0/docs/getting-started-guides/rackspace.html
  - label: CoreOS
    url: v1.0/docs/getting-started-guides/coreos.html
  - label: vSphere
    url: v1.0/docs/getting-started-guides/vsphere.html
  - label: Amazon Web Services
    url: v1.0/docs/getting-started-guides/aws.html
  - label: Mesos
    url: v1.0/docs/getting-started-guides/mesos.html

firstapp:
    label: Run Now
    url: v1.0/examples/guestbook/README.html

releases:
    label: Releases
    url: https://github.com/GoogleCloudPlatform/kubernetes/releases

techdetails:
    api:
        label: API Documentation
        url: http://kubernetes.io/third_party/swagger-ui/
    kubecfg:
        label: Kubectl Command Tool
        url: v1.0/docs/user-guide/kubectl/kubectl.html
---
