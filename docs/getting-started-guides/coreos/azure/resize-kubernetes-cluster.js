#!/usr/bin/env node

var azure = require('./lib/azure_wrapper.js');
var kube = require('./lib/deployment_logic/kubernetes.js');

azure.load_state_for_resizing(process.argv[2], 'kube', 2);

azure.run_task_queue([
  azure.queue_machines('kube', 'stable', kube.create_node_cloud_config),
]);
