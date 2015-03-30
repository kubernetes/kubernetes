#!/usr/bin/env node

var azure = require('./lib/azure_wrapper.js');
var lmktfy = require('./lib/deployment_logic/lmktfy.js');

azure.create_config('lmktfy', { 'etcd': 3, 'lmktfy': 3 });

azure.run_task_queue([
  azure.queue_default_network(),
  azure.queue_storage_if_needed(),
  azure.queue_machines('etcd', 'stable',
    lmktfy.create_etcd_cloud_config),
  azure.queue_machines('lmktfy', 'stable',
    lmktfy.create_node_cloud_config),
]);
