
//
// just to make these example a little bit interesting, 
// make a little key value store with an http interface
// (see couchbd for a grown-up version of this)
//
// API:
// GET / 
// retrive list of keys
//
// GET /[url]
// retrive object stored at [url]
// will respond with 404 if there is nothing stored at [url]
//
// POST /[url]
// 
// JSON.parse the body and store it under [url]
// will respond 400 (bad request) if body is not valid json.
//
// TODO: cached map-reduce views and auto-magic sharding.
//
var Store = module.exports = function Store () {
  this.store = {};
};

Store.prototype = {
  get: function (key) {
    return this.store[key]
  },
  set: function (key, value) {
    return this.store[key] = value
  },
  handler:function () {
    var store = this
    return function (req, res) {
      function send (obj, status) {
        res.writeHead(200 || status,{'Content-Type': 'application/json'})
        res.write(JSON.stringify(obj) + '\n')
        res.end()
      }
      var url = req.url.split('?').shift()
      if (url === '/') {
        console.log('get index')
        return send(Object.keys(store.store))
      } else if (req.method == 'GET') {
        var obj = store.get (url)
        send(obj || {error: 'not_found', url: url}, obj ? 200 : 404)
      } else {
        //post: buffer body, and parse.
        var body = '', obj
        req.on('data', function (c) { body += c})
        req.on('end', function (c) {
          try {
            obj = JSON.parse(body)
          } catch (err) {
            return send (err, 400)
          }
          store.set(url, obj)
          send({ok: true})
        })
      } 
    }
  }
}
