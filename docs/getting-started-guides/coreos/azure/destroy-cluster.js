#!/usr/bin/env node

var azure = require('./lib/azure_wrapper.js');

azure.destroy_cluster(process.argv[2]);

console.log('The cluster had been destroyed, you can delete the state file now.');
