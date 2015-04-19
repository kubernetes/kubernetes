var fs = require('fs'),
    path = require('path'),
    union = require('../../lib');

var server = union.createServer({
  before: [ function (req,res) {
    if (req.url === "/foo") {
      res.text(201, "foo");
    }
  } ],
  after: [
    function LoggerStream() {
        var stream   = new union.ResponseStream();

        stream.once("pipe", function (req) {
          console.log({res: this.res.statusCode, method: this.req.method});
        });

        return stream;
    }
  ]
});

server.listen(9080);
console.log('union running on 9080');

