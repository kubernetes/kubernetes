var WebSocket = require('../lib/faye/websocket'),
    pace      = require('pace');

var host  = 'ws://localhost:9001',
    agent = 'Node ' + process.version,
    cases = 0,
    skip  = [];

var socket = new WebSocket.Client(host + '/getCaseCount'),
    progress;

socket.onmessage = function(event) {
  console.log('Total cases to run: ' + event.data);
  cases = parseInt(event.data);
  progress = pace(cases);
};

socket.onclose = function() {
  var runCase = function(n) {
    progress.op();

    if (n > cases) {
      socket = new WebSocket.Client(host + '/updateReports?agent=' + encodeURIComponent(agent));
      socket.onclose = process.exit;

    } else if (skip.indexOf(n) >= 0) {
      runCase(n + 1);

    } else {
      socket = new WebSocket.Client(host + '/runCase?case=' + n + '&agent=' + encodeURIComponent(agent));
      socket.pipe(socket);
      socket.on('close', function() { runCase(n + 1) });
    }
  };

  runCase(1);
};
