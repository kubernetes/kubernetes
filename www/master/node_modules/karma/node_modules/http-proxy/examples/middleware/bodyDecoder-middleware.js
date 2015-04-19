
var Store = require('../helpers/store')
  , http = require('http')

http.createServer(new Store().handler()).listen(7531, function () {
//try these commands:
// get index:
// curl localhost:7531
// []
//
// get a doc:
// curl localhost:7531/foo
// {"error":"not_found"}
//
// post an doc:
// curl -X POST localhost:7531/foo -d '{"content": "hello", "type": "greeting"}'
// {"ok":true}
//
// get index (now, not empty)
// curl localhost:7531
// ["/foo"]
//
// get doc 
// curl localhost:7531/foo
// {"content": "hello", "type": "greeting"}

//
// now, suppose we wanted to direct all objects where type == "greeting" to a different store 
// than where type == "insult"
//
// we can use connect connect-bodyDecoder and some custom logic to send insults to another Store.

//insult server:

  http.createServer(new Store().handler()).listen(2600, function () {

  //greetings -> 7531, insults-> 2600 

  // now, start a proxy server.

    var bodyParser = require('connect/lib/middleware/bodyParser')
    //don't worry about incoming contont type
    //bodyParser.parse[''] = JSON.parse

    require('../../lib/node-http-proxy').createServer(
      //refactor the body parser and re-streamer into a separate package
      bodyParser(),
      //body parser absorbs the data and end events before passing control to the next
      // middleware. if we want to proxy it, we'll need to re-emit these events after 
      //passing control to the middleware.
      require('connect-restreamer')(),
      function (req, res, proxy) {
        //if your posting an obect which contains type: "insult"
        //it will get redirected to port 2600.
        //normal get requests will go to 7531 nad will not return insults.
        var port = (req.body && req.body.type === 'insult' ? 2600 : 7531)
        proxy.proxyRequest(req, res, {host: 'localhost', port: port})
      }
    ).listen(1337, function () {
      var request = require('request')
      //bodyParser needs content-type set to application/json
      //if we use request, it will set automatically if we use the 'json:' field.
      function post (greeting, type) {
        request.post({
          url: 'http://localhost:1337/' + greeting,
          json: {content: greeting, type: type || "greeting"}
        })
      }
      post("hello")
      post("g'day")
      post("kiora")
      post("houdy")
      post("java", "insult")

      //now, the insult should have been proxied to 2600
      
      //curl localhost:2600
      //["/java"]

      //but the greetings will be sent to 7531

      //curl localhost:7531
      //["/hello","/g%27day","/kiora","/houdy"]

    })
  })
})
