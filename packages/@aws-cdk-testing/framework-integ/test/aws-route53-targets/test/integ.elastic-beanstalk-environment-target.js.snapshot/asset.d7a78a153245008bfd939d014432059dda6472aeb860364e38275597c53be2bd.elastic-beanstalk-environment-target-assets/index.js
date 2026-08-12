var express = require('express');
var app = express();
var port = 8080;

app.listen(port, function () {
  console.log('Server running at http://127.0.0.1:', port);
});
