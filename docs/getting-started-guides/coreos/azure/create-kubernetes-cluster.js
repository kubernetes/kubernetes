#!/usr/bin/env node

var azure = require('./lib/azure_wrapper.js');
var kube = require('./lib/deployment_logic/kubernetes.js');

azure.create_config('kubernetes', { 'etcd': 3, 'kube': 3 });

azure.run_task_queue([
  azure.queue_default_network(),
  azure.queue_machines('etcd', 'stable',
    kube.create_etcd_cloud_config),
  azure.queue_machines('kube', 'stable',
    kube.create_node_cloud_config),
]);
