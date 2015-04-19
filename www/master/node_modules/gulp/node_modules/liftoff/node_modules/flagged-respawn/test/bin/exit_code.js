#!/usr/bin/env node

const flaggedRespawn = require('../../');

flaggedRespawn(['--harmony'], process.argv, function (ready) {

  if (ready) {
    setTimeout(function () {
      process.exit(100);
    }, 100);
  }

});
