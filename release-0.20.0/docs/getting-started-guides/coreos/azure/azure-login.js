#!/usr/bin/env node

require('child_process').fork('node_modules/azure-cli/bin/azure', ['login'].concat(process.argv));
