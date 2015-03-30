#!/usr/bin/env node

var azure = require('./lib/azure_wrapper.js');
var lmktfy = require('./lib/deployment_logic/lmktfy.js');

azure.load_state_for_resizing(process.argv[2], 'lmktfy', parseInt(process.argv[3] || 1));

azure.run_task_queue([
  azure.queue_machines('lmktfy', 'stable', lmktfy.create_node_cloud_config),
]);
