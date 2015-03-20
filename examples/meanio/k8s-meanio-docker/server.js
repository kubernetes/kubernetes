'use strict';

/*
var cl = console.log;
console.log = function(){
  console.trace();
  cl.apply(console,arguments);
};
*/

// Requires meanio .
var mean = require('meanio');
var cluster = require('cluster');


// Code to run if we're in the master process or if we are not in debug mode/ running tests

if ((cluster.isMaster) && (process.execArgv.indexOf('--debug') < 0) && (process.env.NODE_ENV!=='test') && (process.execArgv.indexOf('--singleProcess')<0)) {
//if (cluster.isMaster) {

    console.log('for real!');
    // Count the machine's CPUs
    var cpuCount = require('os').cpus().length;

    // Create a worker for each CPU
    for (var i = 0; i < cpuCount; i += 1) {
        console.log ('forking ',i);
        cluster.fork();
    }

    // Listen for dying workers
    cluster.on('exit', function (worker) {
        // Replace the dead worker, we're not sentimental
        console.log('Worker ' + worker.id + ' died :(');
        cluster.fork();

    });

// Code to run if we're in a worker process
} else {

    var workerId = 0;
    if (!cluster.isMaster)
    {
        workerId = cluster.worker.id;
    }
// Creates and serves mean application
    mean.serve({ workerid: workerId /* more options placeholder*/ }, function (app) {
      var config = app.config.clean;
        var port = config.https && config.https.port ? config.https.port : config.http.port;
        console.log('Mean app started on port ' + port + ' (' + process.env.NODE_ENV + ') cluster.worker.id:', workerId);
    });
}
