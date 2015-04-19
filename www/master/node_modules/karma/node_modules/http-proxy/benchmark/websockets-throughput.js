var crypto = require('crypto'),
    WebSocket = require('ws'),
    async = require('async'),
    httpProxy = require('../');

var SERVER_PORT = 8415,
    PROXY_PORT = 8514;

var testSets = [
  {
    size: 1024 * 1024, // 1 MB
    count: 128         // 128 MB
  },
  {
    size: 1024,        // 1 KB,
    count: 1024        // 1 MB
  },
  {
    size: 128,         // 128 B
    count: 1024 * 8    // 1 MB
  }
];

testSets.forEach(function (set) {
  set.buffer = new Buffer(crypto.randomBytes(set.size));

  set.buffers = [];
  for (var i = 0; i < set.count; i++) {
    set.buffers.push(set.buffer);
  }
});

function runSet(set, callback) {
  function runAgainst(port, callback) {
    function send(sock) {
      sock.send(set.buffers[got++]);
      if (got === set.count) {
        t = new Date() - t;

        server.close();
        proxy.close();

        callback(null, t);
      }
    }

    var server = new WebSocket.Server({ port: SERVER_PORT }),
        proxy = httpProxy.createServer(SERVER_PORT, 'localhost').listen(PROXY_PORT),
        client = new WebSocket('ws://localhost:' + port),
        got = 0,
        t = new Date();

    server.on('connection', function (ws) {
      send(ws);

      ws.on('message', function (msg) {
        send(ws);
      });
    });

    client.on('message', function () {
      send(client);
    });
  }

  async.series({
    server: async.apply(runAgainst, SERVER_PORT),
    proxy: async.apply(runAgainst, PROXY_PORT)
  }, function (err, results) {
    if (err) {
      throw err;
    }

    var mb = (set.size * set.count) / (1024 * 1024);
    console.log(set.size / (1024) + ' KB * ' + set.count + ' (' + mb + ' MB)');

    Object.keys(results).forEach(function (key) {
      var t = results[key],
          throughput = mb / (t / 1000);

      console.log('  ' + key + ' took ' + t + ' ms (' + throughput + ' MB/s)');
    });

    callback();
  });
}

async.forEachLimit(testSets, 1, runSet);
