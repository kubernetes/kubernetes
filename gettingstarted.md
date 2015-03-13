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
  - title: Binary Releases
    text: <em>Coming Soon &hellip;</em>
  - title: Technical Details
    text: "Interested in taking a peek inside Kubernetes? You should start by reading the <a href=\"https://github.com/GoogleCloudPlatform/kubernetes/blob/master/DESIGN.md\">design overview</a> which introduces core Kubernetes concepts and components. After that, you probably want to take a look at the API documentation and learn about the kubecfg command line tool."
    slug: technical

hostedservices: 
  - label: Google Container Engine
    url: https://cloud.google.com/container-engine/

installguides: 
  - label: Google Compute Engine
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/gce.md
  - label: Vagrant
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/vagrant.md
  - label: Fedora (Ansible)
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/fedora/fedora_ansible_config.md
  - label: Fedora (Manual)
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/fedora/fedora_manual_config.md
  - label: Local
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/locally.md
  - label: Microsoft Azure
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/azure.md
  - label: Rackspace
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/rackspace.md
  - label: CoreOS
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/coreos.md
  - label: vSphere
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/vsphere.md
  - label: Amazon Web Services
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/aws.md
  - label: Mesos
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/getting-started-guides/mesos.md

firstapp:
    label: Run Now
    url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/guestbook/README.md

techdetails:
    api: 
        label: API Documentation
        url: http://kubernetes.io/third_party/swagger-ui/
    kubecfg:
        label: Kubectl Command Tool
        url: https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/kubectl.md
---
