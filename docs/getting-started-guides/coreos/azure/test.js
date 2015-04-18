#!/usr/bin/env node

var kube = require('./lib/deployment_logic/kubernetes.js');

kube.create_etcd_cloud_config(1);