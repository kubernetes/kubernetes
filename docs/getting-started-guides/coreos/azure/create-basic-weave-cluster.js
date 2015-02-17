#!/usr/bin/env node

var azure = require('./lib/azure_wrapper.js');
var weave = require('./lib/deployment_logic/weave.js');

azure.create_config('weave-cluster-example', { 'core': 3 });

azure.run_task_queue([
  azure.queue_default_network(),
  azure.queue_machines('core', 'stable',
    weave.create_basic_cloud_config),
]);
