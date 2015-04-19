importScripts('/socket.io/socket.io.js');

self.onmessage = function (ev) {
  var url = ev.data
    , socket = io.connect(url);

  socket.on('done', function () {
    self.postMessage('done!');
  });

  socket.on('connect_failed', function () {
    self.postMessage('connect failed');
  });

  socket.on('error', function () {
    self.postMessage('error');
  });

  socket.send('woot');
}
