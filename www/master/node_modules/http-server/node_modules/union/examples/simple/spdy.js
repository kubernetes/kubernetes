// In order to run this example you need to
// generate local ssl certificate
var union = require('../../lib'),
    director = require('director');

var router = new director.http.Router();

var server = union.createServer({
  before: [
    function (req, res) {
      var found = router.dispatch(req, res);
      if (!found) {
        res.emit('next');
      }
    }
  ],
  spdy :{
    key: './certs/privatekey.pem',
    cert: './certs/certificate.pem'
  }
});

router.get(/foo/, function () {
  this.res.writeHead(200, { 'Content-Type': 'text/plain' })
  this.res.end('hello world\n');
});

server.listen(9090, function () {
  console.log('union with director running on 9090 with SPDY');
});